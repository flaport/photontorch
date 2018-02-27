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
from ..torch_ext.autograd import batch_block_diag
from ..torch_ext.tensor import where
from ..sources.inject import SourceInjector


#############
## Network ##
#############

class Network(Component, SourceInjector):
    ''' a Network (circuit) of Components '''
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

        Component.__init__(self, name=kwargs.pop('name', None))

        # Add all the possible sources to this network:
        self.inject_sources()


        # parse arguments
        self.s, self.components = self._parse_args(args)

        # PyTorch requires submodules to be registered as attributes
        # For the parameters to be found by autograd:
        for comp in self.components:
            i = 0
            name = comp.name + ' '
            while hasattr(self, name.strip()):
                k = 1 if i == 0 else int(np.log10(float(i))) + 1
                i += 1
                name = name[:-k] + str(i)
            if i == 0:
                name = name[:-1]
            setattr(self, name, comp)

        # flag to see if the network is initialized with an environment.
        self.initialized = False

    def copy(self):
        components = [comp.copy() for comp in self.components]
        new = self.__class__(self.s, *components)
        if self.initialized:
            new.initialize(self.env.copy())
        return new

    @property
    def num_ports(self):
        return np.sum(comp.num_ports for comp in self.components)

    def terminate(self, term=None):
        ''' Add Terms to open connections '''
        if term is None:
            term = Term()
        if self.is_cuda:
            term = term.cuda()
        connector = Connector(self.s, self.components)
        idxs = connector.idxs
        for i in idxs:
            term = term.copy()
            term.name = i
            connector = connector*term[i]
        return Network(connector, name=self.name)

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
                ''' Forward function for not fully connected Nework '''
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
        if (delays[delays_in_seconds.data > 0] < 10).any(): # This bound is rather arbitrary...
            warnings.warn('Simulation timestep might be too large, resulting'
                          'in too short delays. Try using a smaller timestep')

        ## detector locations
        detectors_at = self.detectors_at

        ## source locations
        sources_at = self.sources_at


        ## locations of memory-containing and memory-less nodes:

        mc = (sources_at | detectors_at | (delays > 0)) # memory-containing nodes:
        ml = mc.ne(1) # negation of mc: memory-less nodes
        self.nmc = nmc = int(mc.int().sum()) # number of memory-containing nodes:
        nml = int(ml.int().sum()) # number of memory-less nodes:

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

        # batched combined locations:
        bmcmc = torch.cat([mcmc.unsqueeze(0)]*self.env.num_wl, dim=0)
        bmcml = torch.cat([mcml.unsqueeze(0)]*self.env.num_wl, dim=0)
        bmlmc = torch.cat([mlmc.unsqueeze(0)]*self.env.num_wl, dim=0)
        bmlml = torch.cat([mlml.unsqueeze(0)]*self.env.num_wl, dim=0)

        # This extra step is necessary for CUDA:
        # Indexing has to happen with ByteTensors.
        if self.is_cuda:
            mc = mc.byte()
            ml = ml.byte()
            mcmc = mcmc.byte()
            mcml = mcml.byte()
            mlmc = mlmc.byte()
            mlml = mlml.byte()
            bmcmc = bmcmc.byte()
            bmcml = bmcml.byte()
            bmlmc = bmlmc.byte()
            bmlml = bmlml.byte()


        ## S-matrix subsets

        # MC subsets of scattering matrix:
        rS, iS = self.rS, self.iS
        rSmcmc = rS[bmcmc].view(-1, nmc, nmc)
        iSmcmc = iS[bmcmc].view(-1, nmc, nmc)

        # MC subset of Connection matrix
        Cmcmc = C[mcmc].view(nmc, nmc)



        if nml: # Only do the following steps if there is at least one ml node:

            # ML subsets of scattering matrix:
            rSmlml = rS[bmlml].view(-1, nml, nml)
            iSmlml = iS[bmlml].view(-1, nml, nml)

            # ML subsets of connection matrix:
            Cmcml = C[mcml].view(nmc, nml)
            Cmlmc = C[mlmc].view(nml, nmc)
            Cmlml = C[mlml].view(nml, nml)

            ## helper matrices
            # P = I - Cmlml@Smlml
            def cmm(C, S):
                ''' Batch multiply with normal matrix '''
                return torch.matmul(S.permute(0,2,1),C.t()).permute(0,2,1)

            eye = torch.cat([self.new_variable(np.eye(nml), 'float').unsqueeze_(0)]*self.env.num_wl, dim=0)
            rP = eye - cmm(Cmlml, rSmlml)
            iP = -cmm(Cmlml, iSmlml)

            ## reduced connection matrix

            # C = Cmcml@Smlml@inv(P)@Cmlmc + Cmcmc (we do this in 5 steps)

            # 1. Calculation of inv(P) = X + i*Y
            # note that real(inv(P)) != inv(real(P)) in most cases!
            # for a matrix P = rP + i*iP, with rP invertible it is easy to check that
            # the inverse is given by inv(P) = X + i*Y, with
            # X = inv(rP + iP@inv(rP)@iP)
            # Y = -X@iP@inv(rP)
            inv_rP = torch.cat([torch.inverse(rp).unsqueeze(0) for rp in rP], dim=0)
            X = torch.cat([torch.inverse(rp + (ip).mm(inv_rp).mm(ip)).unsqueeze(0) for rp, ip, inv_rp in zip(rP, iP, inv_rP)], dim=0)
            Y = -(X).bmm(iP).bmm(inv_rP)

            # 2. x = Cmcml@Smlml
            rx, ix = cmm(Cmcml, rSmlml), cmm(Cmcml, iSmlml)

            # 3. x = x@invP [rx and ix calculated at the same time: they depend on each other]
            rx, ix = (rx).bmm(X) - (ix).bmm(Y), (rx).bmm(Y) + (ix).bmm(X)

            # 4. x = x@Cmlmc
            rx, ix = torch.matmul(rx, Cmlmc), torch.matmul(ix, Cmlmc)

            # 5. C = x + Cmcmc
            rC = rx + Cmcmc.unsqueeze(0)
            iC = ix
        else:
            rC = torch.cat([Cmcmc.unsqueeze(0)]*self.env.num_wl, dim=0)
            iC = torch.zeros_like(rC)

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
        self._rS = rSmcmc[:, new_order, :][:, :, new_order] # real part of reduced S-matrix
        self._iS = iSmcmc[:, new_order, :][:, :, new_order] # imag part of reduced S-matrix
        self._rC = rC[:, new_order, :][:, :, new_order] # real part of reduced C-matrix
        self._iC = iC[:, new_order, :][:, :, new_order] # imag part of reduced C-matrix

        # Create buffermask (# time, # wavelengths =1, # mc nodes, # batches = 1)
        buffermask = self.zeros((int(self._delays.max())+2, 1, self.nmc, 1))
        for i, d in enumerate(self._delays):
            buffermask[int(d), 0, i, 0] = 1.0
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
    def new_buffer(self):
        '''
        Create buffer to keep the hidden states of the Network (RNN)
        The buffer has shape (# time, # wavelengths, # mc nodes, # batches)
        '''
        buffer = self.zeros((self.buffermask.size(0), self.env.num_wl, self.nmc, self.env.num_batches))
        rbuffer = Variable(buffer.clone())
        ibuffer = Variable(buffer)
        return rbuffer, ibuffer

    @require_initialization
    def forward(self, source):
        '''
        Forward pass of the network.

        Arguments
        ---------
        source : should be a FloatTensor of size
                 (#2 = (real|imag), # time, # wavelength, # mc nodes, # batches)
              OR a Source object from photontorch.sources that returns a FloatTensor when
                 indexed.
        '''

        detected = self.new_variable(self.zeros((self.env.num_timesteps, self.env.num_wl, self.nmc, self.env.num_batches)))

        ## Get new buffer
        rbuffer, ibuffer = self.new_buffer()

        # solve
        for i in range(self.env.num_timesteps):

            # get state
            rx = torch.sum(self.buffermask*rbuffer, dim=0) + source[0,i].clone()
            ix = torch.sum(self.buffermask*ibuffer, dim=0) + source[1,i].clone()

            # connect memory-less components
            # rx and ix need to be calculated at the same time because of dependencies on each other
            rx, ix = (self._rC).bmm(rx) - (self._iC).bmm(ix), (self._rC).bmm(ix) + (self._iC).bmm(rx)

            # get output state
            detected[i] = torch.pow(rx, 2) + torch.pow(ix, 2)

            # connect memory-containing components
            # rx and ix need to be calculated at the same time because of dependencies on each other
            rx, ix = (self._rS).bmm(rx) - (self._iS).bmm(ix), (self._rS).bmm(ix) + (self._iS).bmm(rx)

            # update buffer
            rbuffer = torch.cat((rx.unsqueeze(0), rbuffer[0:-1]), dim=0)
            ibuffer = torch.cat((ix.unsqueeze(0), ibuffer[0:-1]), dim=0)

        return detected[:, :, -self.num_detectors:]

    def plot(self, type, detected, **kwargs):
        ''' Plot detected power versus time or wavelength '''
        label = kwargs.pop('label','')
        if isinstance(detected, Variable):
            detected = detected.data.cpu().numpy()
        if isinstance(detected, torch.Tensor):
            detected = detected.cpu().numpy()
        if detected.ndim == 1:
            detected = detected[:, None]
        type = {'time':'t', 't':'t',
                'wl':'wls', 'wls':'wls',
                'f':'wls', 'freq':'wls','frequency':'wls'}[type]
        x = (self.env.__dict__[type])[:detected.shape[0]]
        detected = detected[:x.shape[0]]
        f = (int(np.log10(max(x))+0.5)//3)*3-3
        x = x*10**-f # no inplace operation, since that would change the original x...
        prefix = {12:'T', 9:'G', 6:'M', 3:'k', 0:'', -3:'m',
                  -6:r'$\mu$', -9:'n', -12:'p', -15:'f'}[f]
        plots = plt.plot(x, detected, **kwargs)
        if type in ['time', 't']:
            plt.xlabel("time [%ss]"%prefix)
        elif type in ['wl', 'wls', 'wavelength', 'wavelengths']:
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
        return batch_block_diag(*(comp.rS for comp in self.components))

    @property
    def iS(self):
        ''' Combined imaginary part of the S-matrix of all the components in the network '''
        return batch_block_diag(*(comp.iS for comp in self.components))

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
            C[i, j] = C[j, i] = 1.0

        # add connections
        for i1, j1, i2, j2  in self._parse_connections():
            idxs1 = free_idxs[i1]
            idxs2 = free_idxs[i2]
            i = Ns[i1] + idxs1[j1]
            j = Ns[i2] + idxs2[j2]
            C[i, j] = C[j, i] = 1.0

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
                        loops += [(i, j1, j2)]
        return loops

    def _parse_connections(self):
        S = self.s.split(',')
        connections = []
        for i1, s1 in enumerate(S[:-1]):
            for i2, s2 in enumerate(S[i1+1:], start=i1+1):
                for j1, c1 in enumerate(s1):
                    for j2, c2 in enumerate(s2):
                        if c1 == c2:
                            connections += [(i1, j1, i2, j2)]
        return connections
