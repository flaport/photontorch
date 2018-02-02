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
from ..utils.autograd import block_diag


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

        # gradients
        self.zero_grad()

        # delays
        # delays can be turned off for frequency calculations
        # with constant input sources
        delays_in_seconds = self.delays * float(self.env.use_delays)
        # resulting delays in terms of the simulation timestep:
        delays = (delays_in_seconds/self.env.dt + 0.5).int()
        # Check if simulation timestep is too big:
        if (delays[delays_in_seconds>0] < 10).any(): # This bound is rather arbitrary...
            warnings.warn('Simulation timestep might be too large, resulting'
                          'in too short delays. Try using a smaller timestep')

        # detector locations
        detectors_at = self.detectors_at

        # source locations
        sources_at = self.sources_at

        # location of memory-containing nodes:
        mc = (sources_at | detectors_at | (delays > 0))
        # location of memory-less nodes:
        ml = mc.ne(1) # negation of mc
        # number of memory-containing nodes:
        nmc = int(mc.sum())
        # number of memory-less nodes:
        nml = int(ml.sum())

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

        # helper matrices
        # P = I - Cmlml@Smlml
        rP = self.new_variable(np.eye(nml),'float') - (Cmlml).mm(rSmlml)
        iP = -(Cmlml).mm(iSmlml)

        # reduced connection matrix
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
        self._rC = rx + Cmcmc
        self._iC = ix

        # reduced scattering matrix
        self._rS = rSmcmc
        self._iS = iSmcmc

        # reduced delays vector
        self._delays = delays[mc]

        # reduced location of detectors
        self._detectors_at = detectors_at[mc]

        # reduced location of sources
        self._sources_at = sources_at[mc]

        # source input weights
        self._rsourceweights = self._sources_at.float().unsqueeze(1)
        self._isourceweights = torch.zeros_like(self._rsourceweights)

        # buffer
        buffer = np.zeros((int(self._delays.max())+2, int(mc.sum())))
        buffermask = np.zeros_like(buffer)
        for i, d in enumerate(self._delays):
            buffermask[int(d), i] = 1.0
        self._rbuffer = self.new_variable(buffer.copy())
        self._ibuffer = self.new_variable(buffer.copy())
        self._buffermask  = self.new_variable(buffermask)

    def forward(self, source):
        # handle input
        rsource, isource = (
            self.new_variable(np.real(source)).unsqueeze(0),
            self.new_variable(np.imag(source)).unsqueeze(0),
        )
        rsource, isource = (
            ((self._rsourceweights).mm(rsource) - (self._isourceweights).mm(isource)).transpose(1,0),
            ((self._rsourceweights).mm(isource) + (self._isourceweights).mm(rsource)).transpose(1,0),
        )

        # initialize return vector
        detected = torch.zeros_like(rsource)

        # solve
        for i, (r_s, i_s) in enumerate(zip(rsource, isource)):
            # get input state
            rx, ix = (
                torch.sum(self._buffermask*self._rbuffer, dim=0) + r_s,
                torch.sum(self._buffermask*self._ibuffer, dim=0) + i_s,
            )

            # connect memory-less components
            # r_x and i_x need to be calculated at the same time because of dependencies on each other
            # the .mm function does not work here (2Dx1D not supported). Using torch.matmul instead
            rx, ix = (
                torch.matmul(self._rC,rx) - torch.matmul(self._iC,ix),
                torch.matmul(self._rC,ix) + torch.matmul(self._iC,rx),
            )

            # get output state
            detected[i] = torch.pow(rx,2) + torch.pow(ix,2)

            # connect memory-containing components
            # r_x and i_x need to be calculated at the same time because of dependencies on each other
            # the .mm function does not work here (2Dx1D not supported). Using torch.matmul instead
            rx, ix = (
                torch.matmul(self._rS,rx) - torch.matmul(self._iS,ix),
                torch.matmul(self._rS,ix) + torch.matmul(self._iS,rx),
            )

            # update buffer
            self._rbuffer = torch.cat((rx.unsqueeze(0), self._rbuffer[0:-1]), dim=0)
            self._ibuffer = torch.cat((ix.unsqueeze(0), self._ibuffer[0:-1]), dim=0)

        return detected[self._detectors_at].view(-1,int(torch.sum(self.detectors_at)))


    def parameters(self):
        '''
        Generator of the parameters of the network. Emulates the behavior of
        a normal torch.nn.Module.parameters() call.
        '''
        for comp in self.components:
            for p in comp.parameters():
                yield p

    def plot(self, detected):
        if isinstance(detected, Variable):
            detected = detected.data.cpu().numpy()
        if isinstance(detected, torch.Tensor):
            detected = detected.cpu().numpy()
        if len(detected.shape) == 1:
            detected = detected[:,None]
        t = np.arange(detected.shape[0])*self.env.dt
        f = (int(np.log10(max(t))+0.5)//3)*3-3
        t *= 10**-f
        prefix = {12:'T',9:'G',6:'M',3:'k',0:'',-3:'m',-6:'$\mu$',-9:'n',-12:'p',-15:'f'}[f]
        plt.plot(t, detected)
        plt.xlabel("time [%ss]"%prefix)
        plt.ylabel("intensity [a.u.]")
        labels = ['output %i'%i for i in range(detected.shape[1])]
        plt.legend(labels)

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
