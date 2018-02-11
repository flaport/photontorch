''' Network Module '''

#############
## Imports ##
#############

## Torch
import torch
from torch.nn import Module
from torch.autograd import Variable

## Others
import warnings
import functools
import numpy as np
import matplotlib.pyplot as plt

## Relative
from .connector import Connector
from ..components.component import Component
from ..components.terms import Term
from ..components.terms import Detector
from ..torch_ext.autograd import block_diag
from ..torch_ext.tensor import where
from ..constants import pi, c


#############
## Network ##
#############

class Network(Component):
    def __init__(self, *args, **kwargs):
        '''
        Initialization of the network.

        Parameters
        ----------
        There are three accepted forms for the arguments of a new network:

        1. First option:
        s = args[0] of type str
        components = args[1:] of type component

        s is a string specifying how the components are connected. It follows
        the einstein summation convention.
        e.g.
        nw = Network('ij,jklm,mn', wg1, dircoup, wg2)
        makes a connection between two waveguides and a directional coupler.
        The connection is made where equal indices occur:
            last port of wg1 is connected to first port of dircoup
            last port of dircoup is connected to first port of wg2.

        2. Second option:
        args is a list of list with args[i][0] of type component and args[i][1] of type
        str. Also follows the einstein summation convention.
        e.g.
        nw = Network(
            (wg1, 'ij'),
            (dircoup, 'jklm'),
            (wg3, 'mn')
        )

        3. Third option:
        args[0] is a Connector object that resulted from multiplication (connecting) of
        indexed components:
        e.g.
        nw = Network(wg1['ij']*dircoup['jklm']*wg2['mn'])

        Note
        ----
        The initializer of the network does not check if the number of indices
        given corresponds to the number of ports in the component.
        '''

        Component.__init__(self, name=kwargs.pop('name','nw'))
        # parse arguments
        self.s, self.components = self._parse_args(args)

        self.num_ports = np.sum(comp.num_ports for comp in self.components)
        self.initialized = False

    def terminate(self, term=None):
        ''' Add Terms to open connections '''
        if term is None:
            term = Term()
        connector = Connector(self.s, self.components)
        idxs = connector.idxs
        for i in idxs:
            connector = connector*term[i]
        return Network(connector, name=self.name)

    def cuda(self):
        ''' Transform Network to live on the GPU '''
        new = self.copy()
        new.components = tuple([comp.cuda() for comp in new.components])
        new.is_cuda = True
        return new

    def cpu(self):
        ''' Transform Network to live on the CPU '''
        new = self.copy()
        new.components = tuple([comp.cpu() for comp in new.components])
        new.is_cuda = False
        return new

    def initialize(self, env):
        '''
        Initializer of the network. The initializer should be called before
        doing the forward pass through the network. It creates all the internal variables
        necessary.

        The Initializer should in principle also be called after every training Epoch to
        update the parameters of the network.
        '''
        ## begin initialization:
        self.initialized = False

        ## Initialize components in the network
        for comp in self.components:
            comp.initialize(env)

        ## Initialize network
        super(Network, self).initialize(env)

        ## Check if network is fully connected
        C = self.C
        fully_connected = ((self.C.sum(0) > 0) | (self.C.sum(1) > 0)).all()
        if not fully_connected:
            def forward(*args, **kwargs):
                raise ValueError('Network not Fully Connected')
            self.forward = forward
            return # Stop initialization here.

        ## delays
        # delays can be turned off for frequency calculations
        # with constant input sources
        delays_in_seconds = self.delays * float(self.env.use_delays)
        # resulting delays in terms of the simulation timestep:
        delays = (delays_in_seconds/self.env.dt + 0.5).int()
        # Check if simulation timestep is too big:
        if (delays[delays_in_seconds>0] < 10).any(): # This bound is rather arbitrary...
            warnings.warn('Simulation timestep might be too large, resulting'
                          'in too short delays. Try using a smaller timestep')

        ## detector locations
        detectors_at = self.detectors_at

        ## source locations
        sources_at = self.sources_at


        ## locations of memory-containing and memory-less nodes:

        mc = (sources_at | detectors_at | (delays > 0)) # memory-containing nodes:
        ml = mc.ne(1) # negation of mc: memory-less nodes
        self.nmc = nmc = int(mc.sum()) # number of memory-containing nodes:
        nml = int(ml.sum()) # number of memory-less nodes:

        # This extra step is necessary for CUDA.
        # CUDA does not allow matrix multiplication of ByteTensors.
        if self.is_cuda:
            mc = mc.float()
            ml = ml.float()

        # combined locations:
        mcmc = (mc.unsqueeze(1)).mm(mc.unsqueeze(0))
        mcml = (mc.unsqueeze(1)).mm(ml.unsqueeze(0))
        mlmc = (ml.unsqueeze(1)).mm(mc.unsqueeze(0))
        mlml = (ml.unsqueeze(1)).mm(ml.unsqueeze(0))

        # This extra step is necessary for CUDA:
        # Indexing has to happen with ByteTensors.
        if self.is_cuda:
            mc = mc.byte()
            ml = ml.byte()
            mcmc = mcmc.byte()
            mcml = mcml.byte()
            mlmc = mlmc.byte()
            mlml = mlml.byte()


        ## S-matrix subsets

        # subsets of scattering matrix:
        rS,iS = self.rS, self.iS
        rSmcmc = rS[mcmc].view(nmc,nmc)
        rSmlml = rS[mlml].view(nml,nml)
        iSmcmc = iS[mcmc].view(nmc,nmc)
        iSmlml = iS[mlml].view(nml,nml)

        # subsets of connection matrix:
        Cmcmc = C[mcmc].view(nmc,nmc)
        Cmcml = C[mcml].view(nmc,nml)
        Cmlmc = C[mlmc].view(nml,nmc)
        Cmlml = C[mlml].view(nml,nml)


        if nml: # Only do the following steps if there is at least one ml node:
            ## helper matrices
            # P = I - Cmlml@Smlml
            rP = self.new_variable(np.eye(nml),'float') - (Cmlml).mm(rSmlml)
            iP = -(Cmlml).mm(iSmlml)

            ## reduced connection matrix

            # C = Cmcml@Smlml@inv(P)@Cmlmc + Cmcmc (we do this in 5 steps)

            # 1. Calculation of inv(P) = X + i*Y
            # note that real(inv(P)) != inv(real(P)) in most cases!
            # for a matrix P = rP + i*iP, with rP invertible it is easy to check that
            # the inverse is given by inv(P) = X + i*Y, with
            # X = inv(rP + iP@inv(rP)@iP)
            # Y = -X@iP@inv(rP)
            inv_rP = torch.inverse(rP)
            X = torch.inverse(rP + (iP).mm(inv_rP).mm(iP))
            Y = -(X).mm(iP).mm(inv_rP)

            # 2. x = Cmcml@Smlml
            rx, ix = (Cmcml).mm(rSmlml), (Cmcml).mm(iSmlml)

            # 3. x = x@invP [rx and ix calculated at the same time: they depend on each other]
            rx, ix = (rx).mm(X) - (ix).mm(Y), (rx).mm(Y) + (ix).mm(X)

            # 4. x = x@Cmlmc
            rx, ix = (rx).mm(Cmlmc), (ix).mm(Cmlmc)

            # 5. C = x + Cmcmc
            rC = rx + Cmcmc
            iC = ix
        else:
            rC = Cmcmc
            iC = torch.zeros_like(Cmcmc)

        ## other locations
        others_at = where((sources_at | detectors_at)[mc].ne(1).data)

        ## locations and number of detectors
        self.num_detectors = int(torch.sum(detectors_at))
        detectors_at = where(detectors_at[mc].data)

        ## locations and number of sources
        self.num_sources = int(torch.sum(sources_at))
        sources_at = where(sources_at[mc].data)

        ## Create and reorder reduced matrices.
        ## The reordering yields a small performance upgrade in the forward pass
        new_order = torch.cat((sources_at, others_at, detectors_at))
        self._delays = delays[mc][new_order] # Reduced delay vector
        self._rS = rSmcmc[new_order][:,new_order] # real part of reduced S-matrix
        self._iS = iSmcmc[new_order][:,new_order] # imag part of reduced S-matrix
        self._rC = rC[new_order][:,new_order] # real part of reduced C-matrix
        self._iC = iC[new_order][:,new_order] # imag part of reduced C-matrix

        # Create buffermask
        buffermask = self.zeros((1, int(self._delays.max())+2, self.nmc))
        for i, d in enumerate(self._delays):
            buffermask[0, int(d), i] = 1.0
        self.buffermask = Variable(buffermask)

        self.initialized = True

    def require_initialization(func):
        ''' Some functions require the Network to be initialized '''
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            if not self.initialized:
                raise ValueError('Network not fully initialized. Is the network fully terminated?')
            return func(self, *args, **kwargs)
        return wrapped

    @require_initialization
    def new_buffer(self, num_batches=1):
        '''
        Create buffer to keep the hidden states of the Network (RNN)
        The buffer has shape (# batches, # time, # mc nodes)
        '''
        buffer = self.zeros((num_batches, self.buffermask.size(1), self.buffermask.size(2)))
        rbuffer = Variable(buffer.clone())
        ibuffer = Variable(buffer)
        return rbuffer, ibuffer

    @require_initialization
    def new_source(self, source=None, add_phase=False):
        '''
        Create a Source FloatTensor with size (# batches, # time, # sources, 2)

        Parameters
        ----------
        source : source amplitude or time evolution of source. Accepted forms are:
                 accepted:
                   * None or (int/float) or (np.ndarray with ndim==1):
                     A new FloatTensor with a single batch and the same input for all
                     source locations will be created (None-> amplitude 1)
                   * (np.ndarray with ndim==2):
                     A new FloatTensor with multiple batches (first dimension) will be
                     created. Same input for all source locations
                   * (np.ndarray with ndim==3):
                     A new FloatTensor with multiple batches and different inputs at each
                     source location will be created.
        add_phase = False : wether to add a phase that varies with time according to the
                            frequency of the light. This is more realistic.
        Note
        ----
        In the case an np.ndarray is provided, then then make sure the number of timesteps
        correspond to the number of timesteps in the environment.
        '''

        if source is None: # We will make a source with amplitude 1
            source = 1
        if isinstance(source, int): # We will make a source with amplitude source
            source = float(source)
        if isinstance(source, float): # We will make a source with amplitude source
            source = source*np.ones_like(self.env.t)

        # Create the FloatTensor
        type = 'torch.cuda.FloatTensor' if self.is_cuda else 'torch.FloatTensor'
        rsource = self.new_variable(np.real(source))
        isource = self.new_variable(np.imag(source))

        if rsource.dim() == 1:
            # Only the time evolution of the source is given, we make a source with
            # one single batch:
            rsource.unsqueeze_(0)
            isource.unsqueeze_(0)
        if rsource.dim() == 2:
            # No different input is given for the (different) source locations.
            # We will use the same source amplitude at each source location.
            rsource.unsqueeze_(-1)
            isource.unsqueeze_(-1)
            rsource = torch.cat([rsource]*self.num_sources, dim=-1)
            isource = torch.cat([isource]*self.num_sources, dim=-1)
        if add_phase:
            # We add the phase introduced by the time evolution of the source:
            rphase = self.new_variable(np.cos(2*pi*(c/self.env.wl)*self.env.t))
            iphase = self.new_variable(np.sin(2*pi*(c/self.env.wl)*self.env.t))
            rphase.unsqueeze_(0).unsqueeze_(-1)
            iphase.unsqueeze_(0).unsqueeze_(-1)
            rsource, isource = rphase*rsource - iphase*isource, rphase*isource + iphase*rsource

        # Concatenate real and imaginary part and return the source:
        return torch.cat((rsource.unsqueeze_(-1), isource.unsqueeze_(-1)), dim=-1)

    @require_initialization
    def forward(self, source):
        '''
        Forward pass of the network.
        source should be a FloatTensor of size (# batches, # time, # sources).
        '''
        type = 'torch.cuda.FloatTensor' if self.is_cuda else 'torch.FloatTensor'
        _source = Variable(self.zeros((source.size(0), source.size(1), self.nmc, 2)))
        _source[:,:,:self.num_sources] = source

        detected = self.new_variable(self.zeros((source.size(0), self.env.num_timesteps, self.nmc)))

        ## Get new buffer
        rbuffer, ibuffer = self.new_buffer(source.size(0))

        # solve
        for i in range(self.env.num_timesteps):

            # get state
            rx = (torch.sum(self.buffermask*rbuffer, dim=1) + _source[:,i,:,0]).t()
            ix = (torch.sum(self.buffermask*ibuffer, dim=1) + _source[:,i,:,1]).t()

            # add source
            #rx = rx + source[:,i,:,0].t()
            #ix = rx + source[:,i,:,1].t()

            # connect memory-less components
            # rx and ix need to be calculated at the same time because of dependencies on each other
            rx, ix = (self._rC).mm(rx) - (self._iC).mm(ix), (self._rC).mm(ix) + (self._iC).mm(rx)

            # get output state
            detected[:,i,:] = (torch.pow(rx,2) + torch.pow(ix,2)).t()

            # connect memory-containing components
            # rx and ix need to be calculated at the same time because of dependencies on each other
            rx, ix = (self._rS).mm(rx) - (self._iS).mm(ix), (self._rS).mm(ix) + (self._iS).mm(rx)

            # update buffer
            rbuffer = torch.cat((rx.t().unsqueeze(1), rbuffer[:,0:-1,:]), dim=1)
            ibuffer = torch.cat((ix.t().unsqueeze(1), ibuffer[:,0:-1,:]), dim=1)

        return detected[:,:,-self.num_detectors:]


    def parameters(self):
        '''
        Generator of the parameters of the network. Emulates the behavior of
        a normal torch.nn.Module.parameters() call.
        '''
        for comp in self.components:
            for p in comp.parameters():
                yield p

    def plot(self, x, detected, label='', type='time'):
        if isinstance(detected, Variable):
            detected = detected.data.cpu().numpy()
        if isinstance(detected, torch.Tensor):
            detected = detected.cpu().numpy()
        if len(detected.shape) == 1:
            detected = detected[:,None]
        f = (int(np.log10(max(x))+0.5)//3)*3-3
        x = x*10**-f # no inplace operation, since that would change the original x...
        prefix = {12:'T',9:'G',6:'M',3:'k',0:'',-3:'m',-6:'$\mu$',-9:'n',-12:'p',-15:'f'}[f]
        plots = plt.plot(x, detected)
        if type=='time':
            plt.xlabel("time [%ss]"%prefix)
        elif type in ['wl','wls','wavelength','wavelengths']:
            plt.xlabel('wavelength [%sm]'%prefix)
        plt.ylabel("intensity [a.u.]")
        names = [comp.name for comp in self.components if isinstance(comp, Detector)]
        for i, (name, plot) in enumerate(zip(names, plots)):
            if name == '' or name is None:
                name = str(i)
            plot.set_label(label + ': ' + name if label != '' else name)
        plt.legend()
        return plots


    @property
    def delays(self):
        return torch.cat([comp.delays for comp in self.components])

    @property
    def detectors_at(self):
        return torch.cat([comp.detectors_at for comp in self.components])

    @property
    def sources_at(self):
        return torch.cat([comp.sources_at for comp in self.components])

    @property
    def rS(self):
        ''' Combined real part of the S-matrix of all the components in the network '''
        return block_diag(*(comp.rS for comp in self.components))

    @property
    def iS(self):
        ''' Combined imaginary part of the S-matrix of all the components in the network '''
        return block_diag(*(comp.iS for comp in self.components))

    @property
    def C(self):
        Ns = np.cumsum([0]+[comp.num_ports for comp in self.components])
        free_idxs = [comp.free_idxs for comp in self.components]

        C = block_diag(*(comp.C for comp in self.components))

        # add loops
        for k, j1, j2 in self._parse_loops():
            idxs = free_idxs[k]
            i = Ns[k] + idxs[j1]
            j = Ns[k] + idxs[j2]
            C[i,j] = C[j,i] = 1.0

        # add connections
        for i1, j1, i2, j2  in self._parse_connections():
            idxs1 = free_idxs[i1]
            idxs2 = free_idxs[i2]
            i = Ns[i1] + idxs1[j1]
            j = Ns[i2] + idxs2[j2]
            C[i,j] = C[j,i] = 1.0

        return C

    @staticmethod
    def _parse_args(args):
        if isinstance(args[0], str):
            s = args[0]
            components = tuple(args[1:])
        elif isinstance(args[0], Connector):
            s = args[0].s
            components = tuple(args[0].components)
        else:
            components, s = zip(*args)
            s = ','.join(s)
        return s, components

    def _parse_loops(self):
        S = self.s.split(',')
        loops = []
        for i, s in enumerate(S):
            for j1, c1 in enumerate(s[:-1]):
                for j2, c2 in enumerate(s[j1+1:], start=j1+1):
                    if c1 == c2:
                        loops += [(i,j1,j2)]
        return loops

    def _parse_connections(self):
        S = self.s.split(',')
        connections = []
        for i1, s1 in enumerate(S[:-1]):
            for i2, s2 in enumerate(S[i1+1:], start=i1+1):
                for j1, c1 in enumerate(s1):
                    for j2, c2 in enumerate(s2):
                        if c1==c2:
                            connections += [(i1,j1,i2,j2)]
        return connections
