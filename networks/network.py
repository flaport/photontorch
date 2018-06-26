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
from copy import copy
from collections import OrderedDict

## Torch
import torch

## Others
import numpy as np

## Relative
from .connector import Connector
from ..components.component import Component
from ..components.terms import Term
from ..components.terms import Source
from ..sources.sources import Source as InputSource
from ..components.terms import Detector
from ..torch_ext.autograd import block_diag
from ..torch_ext.autograd import batch_block_diag
from ..torch_ext.tensor import where
from ..torch_ext.nn import Buffer


#############
## Network ##
#############

class Network(Component):
    ''' a Network (circuit) of Components

    The Network is the core of Photontorch. This is where everything comes together.
    The Network is a special kind of torch.nn.Module, where all subcomponents are
    automatically initialized and connected in the right way.

    It's two core method's are `initialize` and `forward`.

    '''

    # components and connections are defined here for easy subclassing
    components = None
    connections = None

    # dedicated plotting function for plotting detected power
    from .plot import plot

    # for adding the different source classes to the network
    from ..sources.network import add_sources

    def __init__(self, components=None, connections=None, name=None):
        ''' Network initialization

        Args:
            components (dict): a dictionary containing the components of the network.
                keys (str): new names for the components (will override the component name)
                values (Component): component
            connections (list[str]): a list containing the connections of the network.
                the connection string can have two formats:
                    1. "comp1:port1:comp2:port2": signifying a connection between ports (always reflexive)
                    2. "comp:port:output_port": signifying a connection to an output port index
            name (str): name of the network

        Note:
            be careful not to repeat keys in your component dictionary.

        Note:
            the order of the components is determined by the order of the connections
        '''

        if isinstance(components, Connector) and connections is None:
            components, connections = components.parse()

        # Save name of component
        self.name = name

        # Save connections
        if connections is not None:
            self.connections = connections
        else:
            connections = self.connections


        # Get the components that were used in the connections:
        used_components = self.get_used_components()

        # Save components
        if self.components is not None and components is None:
            components = self.components
        if isinstance(components, (tuple, list)):
            components = OrderedDict([(comp.name, comp) for comp in components])

        self.components = OrderedDict()
        for name in used_components:
            comp = copy(components[name]) # shallow copy
            comp.name = name
            self.components[name] = comp

        ## Get component starting indices in the connection matrix
        self.num_ports = sum(comp.num_ports for comp in self.components.values())

        ## We will reorder the obtained C and S matrices:
        self.order = self.get_order()

        ## Initialize
        Component.__init__(self, name=self.name)

        ## Check if network is fully connected
        C = (self.C.detach() > 0).sum(0)
        self.terminated = ((C.sum(0) > 0) | (C.sum(1) > 0)).all()

        # Add all the possible sources to this network:
        if self.terminated:
            self.add_sources()

        # register components as attributes:
        self._register_components()

        # flag to see if the network is initialized with an environment.
        self.initialized = False


    def _register_components(self):
        ''' Register Components as submodules of the network

        Pytorch requires submodules to be registered as attributes of a module.
        This method registers all components in self.components under the component's
        name. If the attribute name is already taken, an error will be raised

        Note:
            This Method should only be called one single in the __init__.
        '''
        for name, comp in self.components.items():
            setattr(self, name, comp)

    def terminate(self, term=None, name=None):
        '''
        Terminate open conections with the term of your choice

        Args (Term | dict): Which term to use. Defaults to Term.
            if dict: specify as many name:terms as free indices in the network
        '''
        n = len(self.free_idxs)
        if term is None:
            term = Term()
        if isinstance(term , Term):
            term = OrderedDict((term.name+'_%i'%i,term.copy()) for i in range(n))
        if isinstance(term, (list, tuple)):
            term = OrderedDict((t.name,t) for t in term)
        if self.is_cuda:
            term = OrderedDict((name,t.cuda()) for name, t in term.items())
        components = OrderedDict([(self.name,self)])
        components.update(term)
        connections = [name+':0:'+self.name+':%i'%i for i, name in enumerate(term)]
        if name is None:
            name = self.name+'_terminated'
        nw = Network(components, connections, name=self.name)
        nw.base = self
        return nw

    def unterminate(self):
        return self.base

    def freeze(self):
        return FrozenNetwork(self)

    def cuda(self, device=None):
        ''' move the parameters and buffers of the network to the gpu '''
        nw = super(Network, self).cuda(device=device)
        if nw._env is not None:
            nw.initialize(nw.env)
        return nw

    def cpu(self):
        ''' move the parameters and buffers of the network to the cpu '''
        nw = super(Network, self).cpu()
        if nw._env is not None:
            nw.initialize(nw.env)
        return nw

    def initialize(self, env=None):
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
            During initialization, it will be checked if the network is fully connected.
            If the network is not fully connected, the initialization will not be finalized.
            (and the self.initialized flag will remain False). This is to speed up
            the initialization of nested networks, where only the top level needs to be
            fully initialized.
        '''
        ## get environment
        if env is None:
            env = self.env

        ## begin initialization:
        self.initialized = False

        ## Initialize components in the network
        for comp in self.components.values():
            comp.initialize(env)

        ## Initialize network
        super(Network, self).initialize(env)

        ## Check if network if fully connected
        if not self.terminated:
            return self# Stop initialization here.

        ## delays
        # delays can be turned off for frequency calculations
        # with constant input sources
        delays_in_seconds = self.delays * float(self.env.use_delays)
        # resulting delays in terms of the simulation timestep:
        delays = (delays_in_seconds/self.env.dt + 0.5).long()


        ## locations of memory-containing and memory-less nodes:

        mc = (self.sources_at | self.detectors_at | (delays > 0)) # memory-containing nodes:
        ml = where(mc.ne(1)) # negation of mc: memory-less nodes
        mc = where(mc)
        self.nmc = len(mc)
        self.nml = len(ml)

        if self.nmc == 0:
            return self # break of initialization; network is probably part of a bigger network

        # Check if simulation timestep is too big:
        if (delays[delays_in_seconds.data > 0] < 10).any(): # This bound is rather arbitrary...
            warnings.warn('Simulation timestep might be too large, resulting '
                          'in too short delays. Try using a smaller timestep')

        ## Source and detector locations
        sources_at = where(self.sources_at[mc])
        detectors_at = where(self.detectors_at[mc])
        self.num_sources = len(sources_at)
        self.num_detectors = len(detectors_at)

        ## New port order
        others_at = where((self.sources_at | self.detectors_at)[mc].ne(1))
        new_order = torch.cat((sources_at, others_at, detectors_at))
        self.mc = mc = mc[new_order]

        ## S-matrix subsets

        # MC subsets of scattering matrix:
        rSmcmc = self.S[0,:,mc,:][:,:,mc]
        iSmcmc = self.S[1,:,mc,:][:,:,mc]

        # MC subset of Connection matrix
        rCmcmc = self.C[0,mc,:][:,mc]
        iCmcmc = self.C[1,mc,:][:,mc]

        if self.nml == 0:
            rC = torch.stack([rCmcmc]*self.env.num_wl, dim=0)
            iC = torch.stack([iCmcmc]*self.env.num_wl, dim=0)
        else:
            # Only do the following steps if there is at least one ml node:

            # ML subsets of scattering matrix:
            rSmlml = self.S[0,:,ml,:][:,:,ml]
            iSmlml = self.S[1,:,ml,:][:,:,ml]

            # ML subsets of connection matrix:
            rCmcml = self.C[0,mc,:][:,ml]
            rCmlmc = self.C[0,ml,:][:,mc]
            rCmlml = self.C[0,ml,:][:,ml]
            iCmcml = self.C[1,mc,:][:,ml]
            iCmlmc = self.C[1,ml,:][:,mc]
            iCmlml = self.C[1,ml,:][:,ml]

            ## Helper function bmm
            def bmm(C, S):
                ''' Batch multiply a normal matrix [C] with a batched matrix [S] '''
                return torch.matmul(S.permute(0,2,1),C.t()).permute(0,2,1)

            ## reduced connection matrix
            # C = Cmcml@Smlml@inv(P)@Cmlmc + Cmcmc
            # (with P a helper matrix defined below)
            # We do this in 6 steps:

            # 1. Calculation of the helper matrix P = I - Cmlml@Smlml
            rP = torch.stack([self.tensor(np.eye(self.nml), 'float')]*self.env.num_wl, dim=0)
            rP = rP - bmm(rCmlml, rSmlml) + bmm(iCmlml, iSmlml)
            iP = -bmm(rCmlml, iSmlml) - bmm(iCmlml, rSmlml)

            # 2. Calculation of inv(P) = X + Y
            # note that real(inv(P)) != inv(real(P)) in most cases!
            # for a matrix P = rP + i*iP, with rP invertible it is easy to check that
            # the inverse is given by inv(P) = X + i*Y, with
            # X = inv(rP + iP@inv(rP)@iP)
            # Y = -X@iP@inv(rP)
            inv_rP = torch.stack([torch.inverse(rp) for rp in rP], dim=0)
            X = torch.stack([torch.inverse(rp + (ip).mm(inv_rp).mm(ip)) for rp, ip, inv_rp in zip(rP, iP, inv_rP)], dim=0)
            Y = -(X).bmm(iP).bmm(inv_rP)

            # 3. calculation of x = Cmcml@Smlml
            rx, ix = bmm(rCmcml, rSmlml) - bmm(iCmcml,iSmlml), bmm(rCmcml, iSmlml) + bmm(iCmcml, rSmlml)

            # 4. Calculation of x@inv(P) := x@(X + i*Y)
            rx, ix = (rx).bmm(X) - (ix).bmm(Y), (rx).bmm(Y) + (ix).bmm(X)

            # 5. C = x@Cmlmc
            rC, iC = (torch.matmul(rx, rCmlmc) - torch.matmul(ix, iCmlmc),
                      torch.matmul(ix, rCmlmc) + torch.matmul(rx, iCmlmc))

            # 6. C = x + Cmcmc
            rC = rC + rCmcmc[None]
            iC = iC + iCmcmc[None]

        ## Save the reduced matrices
        self._delays = delays[mc]
        self._rS = rSmcmc
        self._iS = iSmcmc
        self._rC = rC
        self._iC = iC

        # Create buffermask (# time, # wavelengths = 1, # mc nodes, # batches = 1)
        self.buffermask = self.zeros((int(self._delays.max())+1, 1, self.nmc, 1))
        self.buffermask[self._delays,:,range(self.nmc),:] = 1.0

        self.initialized = True

        # finish initialization:
        return self

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
    def simulation_buffer(self, num_batches):
        ''' Create buffer to keep the hidden states of the Network (RNN)
        The buffer has shape (# time, # wavelengths, # mc nodes, # batches)

        Note:
            obviously, this is a different buffer than Model buffers
        '''
        rbuffer = self.zeros((int(self._delays.max())+1, self.env.num_wl, self.nmc, num_batches))
        ibuffer = rbuffer.clone()
        return rbuffer, ibuffer

    @require_initialization
    def forward(self, source=1.0, power=True, detector=None):
        '''
        Forward pass of the network.

        Args:
            source : should be a FloatTensor of size
                (2 = (real|imag), # time, # wavelength, # mc nodes, # batches) OR a
                Source object from photontorch.sources that returns a FloatTensor when
                indexed.
            power=True (bool): Wether to return a real-valued power or the
                complex-valued field.
            detector=None (Detector): Pass an extra detector instance to detect the fields (or power)

        Returns:
            detected (torch.Floattensor): a tensor with shape
            (# time, # wavelengths, # detectors, # batches) if power==True
            (2 = (real|imag), # time, # wavelengths, # detectors, # batches) if power==False
        '''

        if not (torch.is_tensor(source) or isinstance(source, InputSource)):
            # try to get the source into a version that photontorch understands
            source = np.asarray(source)
            if source.ndim == 0:
                source = self.ConstantSource(source)
            elif source.shape[0] == self.env.num_timesteps:
                source = self.Source(source)
            elif source.ndim < 2 or (source.ndim==2 and source.shape[0] == self.num_sources):
                source = self.ConstantSource(source)
            else:
                source = self.Source(source)

        num_batches = source.shape[-1]

        if power:
            detected = self.tensor(self.zeros((self.env.num_timesteps, self.env.num_wl, self.num_detectors, num_batches)))
            def update_detected():
                detected[i] = (torch.pow(rx, 2) + torch.pow(ix, 2))[:,-self.num_detectors:]
        else:
            detected = self.tensor(self.zeros((2, self.env.num_timesteps, self.env.num_wl, self.num_detectors, num_batches)))
            def update_detected():
                detected[0,i] = rx[:,-self.num_detectors:]
                detected[1,i] = ix[:,-self.num_detectors:]

        ## Get new buffer
        rbuffer, ibuffer = self.simulation_buffer(num_batches)

        # solve
        for i in range(self.env.num_timesteps):

            # get state
            rx = torch.sum(self.buffermask*rbuffer, dim=0)
            ix = torch.sum(self.buffermask*ibuffer, dim=0)

            # add source
            rx = rx + source[0,i]
            ix = ix + source[1,i]

            # connect memory-less components
            # rx and ix need to be calculated at the same time because of dependencies on each other
            rx, ix = (self._rC).bmm(rx) - (self._iC).bmm(ix), (self._rC).bmm(ix) + (self._iC).bmm(rx)

            update_detected()

            # connect memory-containing components
            # rx and ix need to be calculated at the same time because of dependencies on each other
            rx, ix = (self._rS).bmm(rx) - (self._iS).bmm(ix), (self._rS).bmm(ix) + (self._iS).bmm(rx)

            # update buffer
            rbuffer = torch.cat((rx[None], rbuffer[0:-1]), dim=0)
            ibuffer = torch.cat((ix[None], ibuffer[0:-1]), dim=0)

        if detector is not None:
            detected = detector(detected)

        return detected

    def get_used_components(self):
        # Check which components are actually used
        def is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False
        used_components = [conn.split(':') for conn in self.connections]
        # here we would like to use an ordered set, but this does not exist.
        # therefore, we use an OrderedDict, for which we don't care about
        # the values. [all for python 2 compatibility...]
        used_components = OrderedDict([(comp,None) for tup in used_components for comp in tup if not is_int(comp)])
        return list(used_components.keys())

    def get_delays(self):
        ''' get all the delays in the network '''
        return torch.cat([comp.delays for comp in self.components.values()])[self.order]

    def get_detectors_at(self):
        ''' get the locations of the detectors in the network '''
        return torch.cat([comp.detectors_at for comp in self.components.values()])[self.order]

    def get_sources_at(self):
        ''' get the locations of the sources in the network '''
        return torch.cat([comp.sources_at for comp in self.components.values()])[self.order]

    def get_S(self):
        ''' Combined S-matrix of all the components in the network '''
        rS = batch_block_diag(*(comp.S[0] for comp in self.components.values()))[:,self.order,:][:,:,self.order]
        iS = batch_block_diag(*(comp.S[1] for comp in self.components.values()))[:,self.order,:][:,:,self.order]
        return torch.stack([rS,iS])

    def get_order(self):
        ''' Yields reordering indices for the S matrix '''
        start_idxs = list(np.cumsum([0]+[comp.num_ports for comp in self.components.values()])[:-1])
        start_idxs = OrderedDict(zip(self.components.keys(), start_idxs))
        free_idxs = [comp.free_idxs for comp in self.components.values()]
        free_idxs = OrderedDict(zip(self.components.keys(), free_idxs))

        def parse_connection(conn):
            conn_list = conn.split(':')
            if len(conn_list) == 4:
                return None, None
            comp, i, j = conn_list
            if comp not in self.components:
                raise ValueError('Component %s not found during reordering'%comp)
            i, j = int(i), int(j)
            i = int(start_idxs[comp] + free_idxs[comp][i])
            return i, j

        order_dict = {}
        for conn in self.connections:
            i, j = parse_connection(conn)
            if i is not None:
                order_dict[j] = i

        order = []
        for j in range(len(order_dict)):
            i = order_dict.get(j, None)
            if i is not None:
                order.append(i)

        for i in range(self.num_ports):
            if i not in order_dict.values():
                order.append(i)

        return order

    def get_C(self):
        ''' Combined Connection matrix of all the components in the network

        Returns:
            torch.FloatTensors with only 1's and 0's.

        Note:
            To create the connection matrix, the connection string is parsed.
        '''

        rC = block_diag(*(comp.C[0] for comp in self.components.values()))
        iC = block_diag(*(comp.C[1] for comp in self.components.values()))
        C = torch.stack([rC,iC])

        start_idxs = list(np.cumsum([0]+[comp.num_ports for comp in self.components.values()])[:-1])
        start_idxs = OrderedDict(zip(self.components.keys(), start_idxs))
        free_idxs = [comp.free_idxs for comp in self.components.values()]
        free_idxs = OrderedDict(zip(self.components.keys(), free_idxs))

        def parse_connection(conn):
            conn_list = conn.split(':')
            if len(conn_list) < 4:
                return None, None
            comp1, i1, comp2, i2 = conn_list
            i1, i2 = int(i1), int(i2)
            if comp1 == comp2 and i1 == i2:
                raise IndexError('Cannot connect two equal ports')
            for i, comp in [(i1,comp1), (i2, comp2)]:
                if comp not in self.components:
                    raise ValueError('Component %s not found during connecting'%comp)
                if i >= len(free_idxs[comp]):
                    raise ValueError('Component %s only has %i ports. Port index '
                                     '%i too high'%(comp, len(free_idxs[comp]), i))
            j1 = start_idxs[comp1] + free_idxs[comp1][i1]
            j2 = start_idxs[comp2] + free_idxs[comp2][i2]
            return int(j1), int(j2)

        for conn in self.connections:
            i, j = parse_connection(conn)
            if i is not None:
                C[0,i,j] = C[0,j,i] = 1.0

        return C[:,self.order,:][:,:,self.order]


class FrozenNetwork(Network):
    def __init__(self, nw):
        torch.nn.Module.__init__(self)
        self.name = nw.name
        self._env = nw._env
        self.nmc = nw.nmc
        self.is_cuda = nw.is_cuda
        self.num_detectors = nw.num_detectors
        self.num_sources = nw.num_sources
        self._rC = Buffer(nw._rC.detach())
        self._iC = Buffer(nw._iC.detach())
        self._rS = Buffer(nw._rS.detach())
        self._iS = Buffer(nw._iS.detach())
        self._delays = Buffer(nw._delays).detach()
        self.buffermask = Buffer(nw.buffermask.detach())
        self.initialized = True
        self.add_sources()

    def initialize(self, env=None):
        if env is not None:
            self._env = env
        return self

    def forward(self, source=1.0, power=True, detector=None):
        with torch.no_grad():
            detected = super(FrozenNetwork, self).forward(source=source, power=power, detector=detector)
        return detected
