r'''
# PhotonTorch Network

The Network is the core of Photontorch. This is where everything comes together.
The Network is a special kind of torch.nn.Module, where all subcomponents are
automatically initialized and connected in the right way.

## A note on the S- and C-matrix reduction performed during Network.initialize:
Each component can be described in terms of it's S matrix. For such a component,
we have that the output fields \(\bf x_{\rm out}\) are connected to the input fields
\(x_{\rm in}\) through a scattering matrix:
```math
x_{\rm out} = S \cdot x_{\rm in}
```
For a network of components, the field vectors \(x\) will just be stacked on top of each other,
and the S-matrix will just be the block-diagonal matrix of the S-matrices of the
individual components. However, to connect the output fields to each other, we need
a connection matrix, which connects the output fields of the individual components
to input fields of other components in the fields vector \(x\):
```math
x_{\rm in} = C \cdot x_{\rm out}
```
a simulation (without delays) can thus simply be described by:
```math
x(t+1) = C\cdot S\cdot x(t)
```
However, when delays come in the picture, the situation is a bit more complex. We then
split the fields vector \(x\) in a memory_containing part (mc) and a memory-less part (ml):
```math
\begin{pmatrix}x^{\rm mc} \\x^{\rm ml} \end{pmatrix}(t+1) =
\begin{pmatrix} C^{\rm mcmc} & C^{\rm mcml} \\ C^{\rm mlmc} & C^{\rm mlml} \end{pmatrix}
\begin{pmatrix} S^{\rm mcmc} & S^{\rm mcml} \\ S^{\rm mlmc} & S^{\rm mlml} \end{pmatrix}
\cdot\begin{pmatrix}x^{\rm mc} \\x^{\rm ml} \end{pmatrix}(t)
```
Usually, we are only interested in the memory-containing nodes, as memory-less nodes
should be connected together and act all at once. After some matrix algebra we arrive at
```math
\begin{align}
x^{\rm mc}(t+1) &= \left( C^{\rm mcmc} + C^{\rm mcml}\cdot S^{\rm mlml}\cdot
\left(1-C^{\rm mlml}S^{\rm mlml}\right)^{-1} C^{\rm mlmc}\right)S^{\rm mcmc} x^{\rm mc}(t) \\
&= C^{\rm red} x^{\rm mc}(t),
\end{align}
```
Which defines the reduced connection matrix used in the simulations.

## A note on complex matrix inverse

PyTorch still does not allow complex valued Tensors. Therefore, the above equation was
completely rewritten with matrices containing the real and imaginary parts.
This would be fairly straightforward if it were not for the matrix inverse in the
reduced connection matrix:
```math
P^{-1} = \left(1-C^{\rm mlml}S^{\rm mlml}\right)^{-1}
```

unfortunately for complex matrices \(P^{-1} \neq {\rm real}(P)^{-1} + i{\rm imag}(P)^{-1}\),
the actual case is a bit more complicated.

It is however, pretty clear from the equations that the \({\rm real}(P)^{-1}\) will always
exist, and thus we can write for the real and imaginary part of \(P^{-1}\):


```math
\begin{align}
{\rm real}(P^{-1}) &= \left({\rm real}(P) + {\rm imag}(P)\cdot {\rm real}(P)^{-1} \cdot {\rm imag}(P)\right)^{-1}\\
{\rm real}(P^{-1}) &= -{\rm real}(P^{-1})\cdot {\rm imag}(P) \cdot {\rm real}(P)^{-1}
\end{align}
```
This equation is valid, even if \({\rm imag}(P)^{-1}\) does not exist.

'''

#############
## Imports ##
#############

# Standard library
import warnings
import functools

## Torch
import torch
from torch.autograd import Variable

## Others
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
    ''' a Network (circuit) of Components

    The Network is the core of Photontorch. This is where everything comes together.
    The Network is a special kind of torch.nn.Module, where all subcomponents are
    automatically initialized and connected in the right way.

    It's two core method's are `initialize` and `forward`.

    '''
    def __init__(self, *args, **kwargs):
        '''
        Initialization of the network.

        Args:
            There are three accepted forms for the arguments of a new network:

            1. First option:
            s = args[0] of type str
            components = args[1:] of type component

            s is a string specifying how the components are connected. It follows
            the einstein summation convention.
            e.g.
            nw = Network('ij,jklm,mn', wg1, dc, wg2)
            makes a connection between two waveguides and a directional coupler.
            The connection is made where equal indices occur:
                last port of wg1 is connected to first port of dc
                last port of dc is connected to first port of wg2.

            2. Second option:
            args is a list of list with args[i][0] of type component and args[i][1] of type
            str. Also follows the einstein summation convention.
            e.g.
            nw = Network(
                (wg1, 'ij'),
                (dc, 'jklm'),
                (wg3, 'mn')
            )

            3. Third option:
            args[0] is a Connector object that resulted from multiplication (connecting) of
            indexed components:
            e.g.
            nw = Network(wg1['ij']*dc['jklm']*wg2['mn'])

        Note:
            The initializer of the network does not check if the number of indices
            given corresponds to the number of ports in the component.
        '''

        Component.__init__(self, name=kwargs.pop('name', None))


        # parse arguments
        self.s, self.components = self._parse_args(args)

        # Add all the possible sources to this network:
        self._inject_sources()

        # register components as attributes:
        self._register_components()

        # flag to see if the network is initialized with an environment.
        self.initialized = False

    def _register_components(self):
        ''' Register Components as submodules of the network

        Pytorch requires submodules to be registered as attributes of a module.
        This tries to registers all components in self.components under the component's
        name. If the attribute name is already taken, an integer suffix is added.

        Note:
            This Method should only be called one single in the __init__.
        '''
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

    def copy(self):
        ''' create a (deep) copy of the network '''
        components = [comp.copy() for comp in self.components]
        new = self.__class__(self.s, *components)
        if self.initialized:
            new.initialize(self.env.copy())
        return new

    @property
    def num_ports(self):
        ''' Get the number of ports in the network '''
        return np.sum(comp.num_ports for comp in self.components)

    def terminate(self, term=None):
        '''
        Terminate open conections with the term of your choice

        Args (Term): Which term to use. Defaults to Term.
        '''
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
        r''' Initialize
        Initializer of the network. The initializer should be called before
        doing any simulation (forward pass through the network). It creates all the
        internal variables necessary.

        Args:
            env (Environment): Simulation environment.

        The Initializer should in principle also be called after every training Epoch to
        update the parameters of the network.

        The goal of this initialization is to split the network into memory-containing nodes
        (nodes that introduce delay, sources, detectors, ...) and memory-less nodes.
        A matrix reduction method is then applied to remove the memory-less nodes from the
        S-matrix and C-matrix. It is with these reduced matrices that the simulation will
        then be performed.

        The resulting reduced matrices will also be reordered, such that source nodes
        will be the first nodes and detector nodes will be the last nodes. This is done
        purely for ease of access afterwards.

        Note:
            during initialization, the reduced connection matrix is calculated for all
            batches and for all wavelengths. This is a quite lengthy calculation, because
            the matrices need to be (batch) multiplied in the right way. Below, you can find
            the equation for the reduced connection matrix if you get lost:
            ```math
            C^{\rm red} = \left(1-C^{\rm mlml}S^{\rm mlml}\right)^{-1} C^{\rm mlmc}
            ```

        Note:
            During initialization, it will be checked if the network is fully connected,
            i.e. self.C.sum(0) > 0 everywhere and self.C.sum(1) > 0 everywhere. If the
            network is not fully connected, the initialization will not be finalized.
            (and the self.initialized flag will remain False)
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
        '''Decorator.
        Some functions require the Network to be initialized.
        '''
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            if not self.initialized:
                raise ValueError('Network not fully initialized. Is the network fully terminated?')
            return func(self, *args, **kwargs)
        return wrapped

    @require_initialization
    def new_buffer(self):
        ''' Create buffer to keep the hidden states of the Network (RNN)
        The buffer has shape (# time, # wavelengths, # mc nodes, # batches)

        Note:
            obviously, this is a different buffer than Model buffers
        '''
        buffer = self.zeros((self.buffermask.size(0), self.env.num_wl, self.nmc, self.env.num_batches))
        rbuffer = Variable(buffer.clone())
        ibuffer = Variable(buffer)
        return rbuffer, ibuffer

    @require_initialization
    def forward(self, source):
        '''
        Forward pass of the network.

        Args:
            source : should be a FloatTensor of size
                    (2 = (real|imag), # time, # wavelength, # mc nodes, # batches)
             OR     a Source object from photontorch.sources that returns a FloatTensor when
                     indexed.

        Returns:
            detected (torch.Floattensor): a tensor with shape (# time, # wavelengths, # detectors, # batches)
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
        ''' Plot detected power versus time or wavelength

        Args:
            type ('t' or 'wl'): type of the plot.
            detected (np.ndarray | torch.Tensor): Detected signal
            **kwargs: matplotlib plot keyword arguments.
        '''
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
        if len(names) == 1 and len(plots) > 1:
            names = [names[0] + str(i) for i, _ in enumerate(plots)]
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
        ''' Combined Connection matrix of all the components in the network

        Returns:
            torch.FloatTensors with only 1's and 0's.

        Note:
            To create the connection matrix, the connection string is parsed.
        '''
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
