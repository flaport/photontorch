''' Network Module '''

#############
## Imports ##
#############

## Torch
import torch
from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable

## Others
import warnings
import numpy as np
import matplotlib.pyplot as plt

## Relative
from .connector import Connector
from ..components.component import Component
from ..components.terms import Detector
from ..utils.autograd import block_diag
from ..utils.tensor import zeros
from ..utils.tensor import where


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

    def cuda(self):
        ''' Transform Network to live on the GPU '''
        components = [comp.cuda() for comp in self.components]
        new = self.__class__(','.join(self.s), *components)
        new._cuda = True
        return new

    def cpu(self):
        ''' Transform Network to live on the CPU '''
        components = [comp.cpu() for comp in self.components]
        new = self.__class__(','.join(self.s), *components)
        new._cuda = False
        return new

    def initialize(self, env):
        '''
        Initializer of the network. The initializer should be called before
        doing the forward pass through the network. It creates all the internal variables
        necessary.

        The Initializer should in principle also be called after every training Epoch to
        update the parameters of the network.
        '''
        ## Initialize components in the network
        for comp in self.components:
            comp.initialize(env)

        ## Initialize network
        super(Network, self).initialize(env)

        ## Check if network is fully connected
        C = self.C
        fully_connected = (self.C.sum(0) > 0).all() or (self.C.sum(1) > 0).all()
        if not fully_connected:
            def forward(*args, **kwargs):
                raise ValueError('Network not Fully Connected')
            self.forward = forward
            return # Stop initialization here.

        ## gradients
        self.zero_grad()

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
        if self._cuda:
            mc = mc.float()
            ml = ml.float()

        # combined locations:
        mcmc = (mc.unsqueeze(1)).mm(mc.unsqueeze(0))
        mcml = (mc.unsqueeze(1)).mm(ml.unsqueeze(0))
        mlmc = (ml.unsqueeze(1)).mm(mc.unsqueeze(0))
        mlml = (ml.unsqueeze(1)).mm(ml.unsqueeze(0))

        # This extra step is necessary for CUDA:
        # Indexing has to happen with ByteTensors.
        if self._cuda:
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

        ## other locations
        others_at = where((sources_at | detectors_at)[mc].ne(1).data)

        ## locations and number of detectors
        self.num_detectors = int(torch.sum(detectors_at))
        detectors_at = where(detectors_at[mc].data)#list(np.where(detectors_at[mc].data.cpu().numpy())[0])

        ## locations and number of sources
        self.num_sources = int(torch.sum(sources_at))
        sources_at = where(sources_at[mc].data)#list(np.where(sources_at[mc].data.cpu().numpy())[0])

        ## Create and reorder reduced matrices.
        ## The reordering yields a small performance upgrade in the forward pass
        new_order = torch.cat((sources_at, others_at, detectors_at))
        self._delays = delays[mc][new_order] # Reduced delay vector
        self._rS = rSmcmc[new_order][:,new_order] # real part of reduced S-matrix
        self._iS = iSmcmc[new_order][:,new_order] # imag part of reduced S-matrix
        self._rC = rC[new_order][:,new_order] # real part of reduced C-matrix
        self._iC = iC[new_order][:,new_order] # imag part of reduced C-matrix

        # Create buffermask
        type = 'torch.cuda.FloatTensor' if self._cuda else 'torch.FloatTensor'
        buffermask = zeros(1, int(self._delays.max())+2, self.nmc, type=type)
        for i, d in enumerate(self._delays):
            buffermask[0, int(d), i] = 1.0
        self.buffermask = Variable(buffermask)

    def new_buffer(self, num_batches=1):
        '''
        Create buffer to keep the hidden states of the Network (RNN)
        The buffer has shape (# batches, # time, # mc nodes)
        '''
        type = 'torch.cuda.FloatTensor' if self._cuda else 'torch.FloatTensor'
        buffer = zeros(num_batches, self.buffermask.size(1), self.buffermask.size(2), type=type)
        rbuffer = Variable(buffer.clone())
        ibuffer = Variable(buffer)
        return rbuffer, ibuffer

    def handle_source(self, source):
        '''
        Make a FloatTensor with size (# batches, # time, # sources)
        '''
        type = 'torch.cuda.FloatTensor' if self._cuda else 'torch.FloatTensor'
        _rsource = self.new_variable(np.real(source))
        _isource = self.new_variable(np.imag(source))
        if _rsource.dim() == 1:
            # batch with size 1:
            _rsource = _rsource.unsqueeze(0)
            _isource = _isource.unsqueeze(0)
        if _rsource.dim() == 2:
            # same input for all sources
            _rsource = torch.cat([_rsource.unsqueeze(-1)]*self.num_sources, dim=-1)
            _isource = torch.cat([_isource.unsqueeze(-1)]*self.num_sources, dim=-1)

        source = Variable(zeros(_rsource.size(0), _rsource.size(1), self.nmc, 2, type=type))
        source[:,:,:self.num_sources,0] = _rsource
        source[:,:,:self.num_sources,1] = _isource
        return source


    def forward(self, source):
        '''
        Forward pass of the network.
        source should be a FloatTensor of size (# batches, # time, # sources).
        '''

        detected = self.new_variable(np.zeros((source.size(0), self.env.num_timesteps, self.nmc)))

        ## Get new buffer
        rbuffer, ibuffer = self.new_buffer(source.size(0))

        # solve
        for i in range(self.env.num_timesteps):

            # get input state
            rx = (torch.sum(self.buffermask*rbuffer, dim=1) + source[:,i,:,0]).t()
            ix = (torch.sum(self.buffermask*ibuffer, dim=1) + source[:,i,:,1]).t()

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

    def plot(self, t, detected, label='', type='time'):
        if isinstance(detected, Variable):
            detected = detected.data.cpu().numpy()
        if isinstance(detected, torch.Tensor):
            detected = detected.cpu().numpy()
        if len(detected.shape) == 1:
            detected = detected[:,None]
        f = (int(np.log10(max(t))+0.5)//3)*3-3
        t *= 10**-f
        prefix = {12:'T',9:'G',6:'M',3:'k',0:'',-3:'m',-6:'$\mu$',-9:'n',-12:'p',-15:'f'}[f]
        plots = plt.plot(t, detected)
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
        for i, j1, j2 in self._parse_loops(self.s):
            idxs = free_idxs[i]
            i = Ns[i] + idxs[j1]
            j = Ns[i] + idxs[j2]
            C[i,j] = C[j,i] = 1.0

        # add connections
        for i1, j1, i2, j2  in self._parse_connections(self.s):
            idxs1 = free_idxs[i1]
            idxs2 = free_idxs[i2]
            i = Ns[i1] + idxs1[j1]
            j = Ns[i2] + idxs2[j2]
            C[i,j] = C[j,i] = 1.0

        return C

    @staticmethod
    def _parse_args(args):
        if isinstance(args[0], str):
            s = args[0].split(',')
            components = args[1:]
        elif isinstance(args[0], Connector):
            s = args[0].s.split(',')
            components = args[0].components
        else:
            components, s = zip(*args)
        return s, components

    @staticmethod
    def _parse_loops(S):
        loops = []
        for i, s in enumerate(S):
            for j1, c1 in enumerate(s[:-1]):
                for j2, c2 in enumerate(s[j1+1:], start=j1+1):
                    if c1 == c2:
                        loops += [(i,j1,j2)]
        return loops

    @staticmethod
    def _parse_connections(S):
        connections = []
        for i1, s1 in enumerate(S[:-1]):
            for i2, s2 in enumerate(S[i1+1:], start=i1+1):
                for j1, c1 in enumerate(s1):
                    for j2, c2 in enumerate(s2):
                        if c1==c2:
                            connections += [(i1,j1,i2,j2)]
        return connections
