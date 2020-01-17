r"""
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
{\rm real}(P^{-1}) &= \left({\rm real}(P) + {\rm imag}(P)\cdot {\rm real}(P)^{-1}
\cdot {\rm imag}(P)\right)^{-1}\\
{\rm real}(P^{-1}) &= -{\rm real}(P^{-1})\cdot {\rm imag}(P) \cdot {\rm real}(P)^{-1}
\end{align}
```
This equation is valid, even if \({\rm imag}(P)^{-1}\) does not exist.

"""

#############
## Imports ##
#############

# Standard library
import warnings
import functools
from copy import copy, deepcopy
from collections import OrderedDict, deque

## Torch
import torch

## Others
import numpy as np

## Relative
from ..torch_ext.nn import Buffer
from ..components.component import Component
from ..components.terms import Term
from ..torch_ext.autograd import block_diag
from ..torch_ext.tensor import where
from ..environment import current_environment


#############
## Globals ##
#############
_current_networks = deque()


#############
## Network ##
#############


class Network(Component):
    """ a Network (circuit) of Components

    The Network is the core of Photontorch. This is where everything comes together.
    The Network is a special kind of torch.nn.Module, where all subcomponents are
    automatically initialized and connected in the right way.

    It's two core method's are `initialize` and `forward`.

    """

    # components and connections are defined here for easy subclassing
    components = None
    connections = None

    # dedicated plotting methods for plotting detected power
    from .visualize import plot, graph

    # network properties
    @property
    def num_ports(self):
        return sum(comp.num_ports for comp in self.components.values())

    # Network creation methods
    # ------------------------

    # network initialization method
    def __init__(
        self, components=None, connections=None, name=None, copy_components=False
    ):
        """ Network initialization

        Args:
            components: dict = {}: a dictionary containing the components of the network.
                keys: str: new names for the components (will override the component name)
                values: Component: the component
            connections: list = []: a list containing the connections of the network.
                the connection string can have two formats:
                    1. "comp1:port1:comp2:port2": signifying a connection between ports
                       (always reflexive)
                    2. "comp:port:output_port": signifying a connection to an output port index
            name: str = None: name of the network
            deepcopy: bool = False: if a deepcopy of the components should be taken
                before the components are registered to the network.

        Note:
            be careful not to repeat keys in your component dictionary.

        Note:
            the final order of the components is determined by the order of the connections
        """

        # initial network initialization without calculating the necessary buffers
        super(Network, self).__init__(name=name, _calculate_buffers=False)

        # flag to see if a deepcopy of the component should be
        # made before it's added to the network.
        self.copy_components = copy_components

        # to store the components and connections:
        self.components = OrderedDict()
        self.connections = []  # a list of individual connections
        self.links = []  # a list of complete network links

        # store the components as torch modules:
        if components is not None:
            for name, comp in components.items():
                self.add_component(name, comp)  # add components to the _modules dict

        # register connections:
        if connections is not None:
            self.connections += connections
            self._register_connections()  # add components tot he components dict

    # add a component to the network
    def add_component(self, name, comp, copy=True):
        """ Add a component to the network

        Pytorch requires submodules to be registered as attributes of a module.
        This method register a component as a torch modules in the _modules dictionary.
        """
        if self.components is None:
            raise RuntimeError("Network not yet initialized.")
        if name in self.components:
            raise AttributeError(
                'A component with name "%s" was already added to the network' % name
            )
        if comp.name is not None and comp.name != name:
            raise ValueError(
                'The chosen component has name "%s", but is assigned to attribute "%s". '
                "Please do not specify a name when assigning to an attribute of the "
                "network. The name will be inferred from the attributes name."
                % (comp.name, name)
            )
        if copy and self.copy_components:
            comp = deepcopy(comp)
        self.add_module(name, comp)

    # link components together
    def link(self, *ports):
        """ link components together

        Args:
            *ports: the ports to link together. The first and last port can be an integer
            to specify the ordering of the network ports.

        Note:
            if more than two ports are specified, then the intermediate ports should be
            of the 'double port' type (i.e. "idx1:comp_name:idx2"). The first index will
            connect to the port before; the second index will connect to the port
            after.

        Example:
            >>> with pt.Network() as nw:
            >>>     nw.dc = pt.DirectionalCoupler()
            >>>     nw.wg = pt.Waveguide()
            >>>     nw.link(0, '0:dc:2','0:wg:1','3:dc:1', 1)


        """

        try:
            if ports[0] is not None:
                ports = (int(ports[0]),) + ports[1:]
        except ValueError:
            ports = (None,) + ports

        try:
            if ports[-1] is not None:
                ports = ports[:-1] + (int(ports[-1]),)
        except ValueError:
            ports = ports + (None,)

        ports = [ports[0]] + [tuple(p.split(":")) for p in ports[1:-1]] + [ports[-1]]

        if ports[0] is None and len(ports[1]) == 2:
            ports[1] = (None,) + ports[1]

        if ports[-1] is None and len(ports[-2]) == 2:
            ports[-2] = ports[-2] + (None,)

        self.links.append(tuple(ports))

        if len(ports) < 3:
            raise ValueError("At least one port needs to be specified.")

        # add connection to current network:
        for (_, name1, p1), (p2, name2, _) in zip(ports[1:-2], ports[2:-1]):
            connection_string = "%s:%s:%s:%s" % (name1, p1, name2, p2)
            self.connections.append(connection_string)

        # add input and output connection orders:
        p = ports[0]
        q, name, _ = ports[1]
        if p is not None:
            if q is None:
                raise RuntimeError(
                    "Error during linking: output port %s for %s specified, but no port to link to in subcomponent %s."
                    % (p, self.__class__.__name__, name)
                )
            connection_string = "%s:%s:%s" % (name, q, str(p))
            self.connections.append(connection_string)

        p = ports[-1]
        _, name, q = ports[-2]
        if p is not None:
            if q is None:
                raise RuntimeError(
                    "Error during linking: output port %s for %s specified, but no port to link to in subcomponent %s."
                    % (p, self.__class__.__name__, name)
                )
            connection_string = "%s:%s:%s" % (name, q, p)
            self.connections.append(connection_string)

        # register connections made:
        self._register_connections()

    # helper function to link components together
    def _get_used_component_names(self):
        """ Get which components are already used in a connection """

        def is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        used_components = [conn.split(":") for conn in self.connections]
        # here we would like to use an ordered set, but this does not exist.
        # therefore, we use an OrderedDict, for which we don't care about
        # the values. [all for python 2 compatibility...]
        used_components = OrderedDict(
            [
                (comp, None)
                for tup in used_components
                for comp in tup
                if not is_int(comp)
            ]
        )
        return used_components.keys()

    # helper function to link components together:
    def _register_connections(self):
        # get the registered modules from the network:
        modules = self._modules

        # add used components to the components dictionary:
        self.components = OrderedDict()
        for name in self._get_used_component_names():
            self.components[name] = modules[name]

        # calculate buffers
        o = self.order = self.get_order()
        self.C = Buffer(self.get_C()[:, o, :][:, :, o])
        self.sources_at = Buffer(self.get_sources_at()[o])
        self.detectors_at = Buffer(self.get_detectors_at()[o])
        self.actions_at = Buffer(self.get_actions_at()[o])
        self.free_idxs = Buffer(self.get_free_idxs())
        self.terminated = len(self.free_idxs) == 0

    # terminate a network
    def terminate(self, term=None, name=None):
        """
        Terminate open conections with the term of your choice

        Args:
            term: Term | list | dict = None: A dictionary containing the terms to use
                Which term to use. Defaults to Term. If a dictionary or list is specified,
                then one needs to specify as many terms as there are open connections.
        """
        n = len(self.free_idxs)
        if n == 0:
            raise IndexError("no free ports for termination")
        if term is None:
            term = Term()
        if isinstance(term, Term):
            term = OrderedDict(
                (term.__class__.__name__.lower() + "_%i" % i, deepcopy(term))
                for i in range(n)
            )
        if isinstance(term, (list, tuple)):
            term = OrderedDict((t.name, t) for t in term)
        if self.is_cuda:
            term = OrderedDict((name, t.cuda()) for name, t in term.items())
        copied = copy(self)  # shallow copy so we can change name if necessary
        if copied.name is None:
            copied.name = copied.__class__.__name__.lower()
        components = OrderedDict([(copied.name, copied)])
        components.update(term)
        connections = [
            name + ":0:" + copied.name + ":%i" % i for i, name in enumerate(term)
        ]
        if name is None:
            name = copied.name + "_terminated"
        nw = Network(components, connections, name=name)
        nw.base = self
        return nw

    # undo a termination of a network:
    def unterminate(self):
        """ remove termination of network """
        return self.base

    # Methods to prepare for simulation
    # ---------------------------------

    def initialize(self):
        r""" Initialize
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
        """

        ## get current environment
        self._env = env = current_environment()

        ## begin initialization:
        self.initialized = False

        ## Initialize components in the network
        for comp in self.components.values():
            comp.initialize()

        ## Initialize network
        super(Network, self).initialize()

        self.delays = self.delays[self.order]
        self.S = self.S[:, :, self.order, :][:, :, :, self.order]

        ## delays
        # delays can be turned off for frequency domain calculations
        delays_in_seconds = self.delays * float(not self.env.frequency_domain)
        # resulting delays in terms of the simulation timestep:
        delays = (delays_in_seconds / self.env.dt + 0.5).long()

        ## locations of memory-containing and memory-less nodes:

        mc = (
            self.sources_at | self.detectors_at | self.actions_at | (delays > 0)
        )  # memory-containing nodes:
        ml = where(mc.ne(1))  # negation of mc: memory-less nodes
        mc = where(mc)
        self.nmc = len(mc)
        self.nml = len(ml)

        if not self.terminated or self.nmc == 0:
            # break off initialization; network is probably part of a bigger network
            return self

        # Check if simulation timestep is too big:
        if (delays[delays_in_seconds.data > 0] < 1).any():
            warnings.warn(
                "Simulation timestep might be too large, resulting "
                "in too short delays. Try using a smaller timestep",
                RuntimeWarning,
            )

        ## Source and detector locations
        sources_at = where(self.sources_at[mc])
        detectors_at = where(self.detectors_at[mc])
        actions_at = where(self.actions_at[mc])
        self.num_sources = len(sources_at)
        self.num_detectors = len(detectors_at)
        self.num_actions = len(actions_at)

        ## Create action for the network if necessary:
        self.components_with_action = [
            comp for comp in self.components.values() if comp.actions_at.any()
        ]
        self.action_idxs = np.cumsum(
            [self.num_sources]
            + [comp.num_ports for comp in self.components_with_action]
        )

        ## New port order
        others_at = where(
            (self.sources_at | self.actions_at | self.detectors_at)[mc].ne(1)
        )
        new_order = torch.cat((sources_at, actions_at, others_at, detectors_at))
        self.mc = mc = mc[new_order]
        self._sources_at = where(self.sources_at[mc])
        self._detectors_at = where(self.detectors_at[mc])
        self._actions_at = where(self.actions_at[mc])

        ## S-matrix subsets

        # MC subsets of scattering matrix:
        rSmcmc = self.S[0, :, mc, :][:, :, mc]
        iSmcmc = self.S[1, :, mc, :][:, :, mc]

        # MC subset of Connection matrix
        rCmcmc = self.C[0, mc, :][:, mc]
        iCmcmc = self.C[1, mc, :][:, mc]

        if self.nml == 0:
            rC = torch.stack([rCmcmc] * self.env.num_wavelengths, dim=0)
            iC = torch.stack([iCmcmc] * self.env.num_wavelengths, dim=0)
        else:
            # Only do the following steps if there is at least one ml node:

            # ML subsets of scattering matrix:
            rSmlml = self.S[0, :, ml, :][:, :, ml]
            iSmlml = self.S[1, :, ml, :][:, :, ml]

            # ML subsets of connection matrix:
            rCmcml = self.C[0, mc, :][:, ml]
            rCmlmc = self.C[0, ml, :][:, mc]
            rCmlml = self.C[0, ml, :][:, ml]
            iCmcml = self.C[1, mc, :][:, ml]
            iCmlmc = self.C[1, ml, :][:, mc]
            iCmlml = self.C[1, ml, :][:, ml]

            ## Helper function bmm
            def bmm(C, S):
                """ Batch multiply a normal matrix [C] with a batched matrix [S] """
                return torch.matmul(S.permute(0, 2, 1), C.t()).permute(0, 2, 1)

            ## reduced connection matrix
            # C = Cmcml@Smlml@inv(P)@Cmlmc + Cmcmc
            # (with P a helper matrix defined below)
            # We do this in 5 steps:

            # 1. Calculation of the helper matrix P = I - Cmlml@Smlml
            ones = torch.ones((self.env.num_wavelengths, 1, 1), device=self.device)
            rP = ones * torch.eye(self.nml, device=self.device)[None, :, :]
            rP = rP - bmm(rCmlml, rSmlml) + bmm(iCmlml, iSmlml)
            iP = -bmm(rCmlml, iSmlml) - bmm(iCmlml, rSmlml)

            # 2. Calculate inv(P)@Cmlmc [using torch.gesv]
            M = torch.cat([torch.cat([rP, -iP], -1), torch.cat([iP, rP], -1)], -2)
            Cmlmc = ones * torch.cat([ones * rCmlmc[None], ones * iCmlmc[None]], -2)
            x, _ = torch.solve(Cmlmc, M)
            rx, ix = torch.split(x, x.shape[-2] // 2, -2)

            # 3. Calculate Smlml@inv(P)@Cmlmc
            rx, ix = (
                (rSmlml).bmm(rx) - (iSmlml).bmm(ix),
                (iSmlml).bmm(rx) + (rSmlml).bmm(ix),
            )

            # 4. Calculate Cmcml@Smlml@inv(P)@Cmlmc
            rx, ix = (
                bmm(rCmcml, rx) - bmm(iCmcml, ix),
                bmm(rCmcml, ix) + bmm(iCmcml, rx),
            )

            # 5. C = x + Cmcmc
            rC = rx + rCmcmc[None]
            iC = ix + iCmcmc[None]

        ## Save the reduced matrices
        self._delays = delays[mc]
        self._rS = rSmcmc
        self._iS = iSmcmc
        self._rC = rC
        self._iC = iC

        # Create buffermask (# time, # wavelengths = 1, # mc nodes, # batches = 1)
        self.buffermask = torch.zeros(
            (2, int(self._delays.max()) + 1, 1, self.nmc, 1), device=self.device
        )
        self.buffermask[:, self._delays, :, range(self.nmc), :] = 1.0

        self.initialized = True

        # finish initialization:
        return self

    def simulation_buffer(self, num_batches):
        """ Create buffer to keep the hidden states of the Network (RNN)

        Args:
            num_batches: int: number of batches to create the buffer for

        Returns:
            buffer: torch.Tensor[2, #timesteps, #wavelengths, #mc nodes, num_batches]

        Note:
            obviously, this is a different buffer than a Model Buffer.
        """
        buffer = torch.zeros(
            (
                2,
                int(self._delays.max()) + 1,
                self.env.num_wavelengths,
                self.nmc,
                num_batches,
            ),
            device=self.device,
        )
        return buffer

    def handle_source(self, source, axes=None):
        """ bring a source in a usable form

        the .forward method ideally expects an input source of shape
        source.shape == (2, #timesteps, #wavelengths, #mc nodes, #batches)

        however, it can be tedious to bring your input array in exactly this format.
        This method brings the source array (which may be lower dimensional) in the
        expected format.

        Args:
            source: torch.Tensor: the input tensor to be brought in the expected format
                for the forward pass
            axes: str = None: the "priority" order of the axes. If a certain input
                array has to be expanded to more dimensions, it is often ambiguous which
                dimensions (axes) are already present. You can override the default
                priority (which can change with the type of environment used) by specifying
                a list of axes present. Allowed characters to use are:
                    t: time axis
                    w: wavelength axis
                    s: source axis
                    b: batch axis

        Example:
            >>> src = torch.random.randn(num_batches, num_timesteps)
            >>> new_src = nw.handle_source(src, axes="bt")
            >>> new_src.shape == (2, num_timesteps, num_wavelengths, num_mc_nodes, num_batches)
            True

        """
        stacked = False

        # The source should be a tensor
        if not torch.is_tensor(source):
            source = np.asarray(source)
            source = torch.stack(
                [
                    torch.tensor(
                        np.real(source),
                        dtype=torch.get_default_dtype(),
                        device=self.device,
                    ),
                    torch.tensor(
                        np.imag(source),
                        dtype=torch.get_default_dtype(),
                        device=self.device,
                    ),
                ],
                0,
            )
            stacked = True
        elif source.shape == (
            2,
            self.env.num_timesteps,
            self.env.num_wavelengths,
            self.num_sources,
        ):
            source = source[:, :, :, :, None]
            stacked = True
        elif source.shape[:-1] == (
            2,
            self.env.num_timesteps,
            self.env.num_wavelengths,
            self.num_sources,
        ):
            stacked = True

        # The source should be a tensor with ndim > 0
        if len(source.shape) == 0:
            source = source * torch.ones(
                (self.env.num_timesteps, self.env.num_wavelengths, self.num_sources, 1),
                dtype=source.dtype,
                device=source.device,
            )
        if len(source.shape) == 1 and source.shape[0] == 2:
            source = source[:, None, None, None, None] * torch.ones(
                (self.env.num_timesteps, self.env.num_wavelengths, self.num_sources, 1),
                dtype=source.dtype,
                device=source.device,
            )
            stacked = True

        # The source should be a tensor with the first dimension == 2 for the real and imag part
        if source.shape[0] != 2 or (
            not stacked
            and (
                source.shape[0] == self.num_sources
                or source.shape[0] == self.env.num_wavelengths
                or source.shape[0] == self.env.num_timesteps
            )
        ):
            source = torch.stack([source, torch.zeros_like(source)])

        # default ordering of the source axis according to the number of dimensions:
        # the first dimension (real|imag) is ignored
        if axes is None:
            if len(source.shape) - 1 == 1:
                if source.shape[-1] == self.env.num_timesteps:
                    axes = "t"
                elif source.shape[-1] == self.num_sources:
                    axes = "s"
                elif source.shape[-1] == self.env.num_wavelengths:
                    axes = "w"
                else:
                    axes = "b"
            elif len(source.shape) - 1 == 2:
                if source.shape[-1] == self.num_sources:
                    axes = "ts"
                elif source.shape[-1] == self.env.num_wavelengths:
                    axes = "tw"
                else:
                    axes = "tb"
            elif len(source.shape) - 1 == 3:
                axes = "t"
                if source.shape[-2] == self.num_sources:
                    axes = axes + "s"
                elif source.shape[-2] == self.env.num_wavelengths:
                    axes = axes + "w"
                else:
                    axes = axes + "b"
                if source.shape[-1] == self.env.num_wavelengths and "w" not in axes:
                    axes = axes + "w"
                elif source.shape[-1] == self.num_sources and "s" not in axes:
                    axes = axes + "s"
                elif "b" not in axes:
                    axes = axes + "b"
            elif len(source.shape) - 1 == 4:
                axes = "twsb"
            else:
                raise ValueError("invalid input source shape")

        # Iterate over the axis names and add dimensions if axis does not exist
        # In the meantime keep track of the order of the axisses
        for c in "twsb":
            if c not in axes:
                source = source[..., None]
                axes = axes + c

        # Transpose the source to the default order: (2, time, wls, sources, batches)
        order = [0]
        for c in "twsb":
            order.append(axes.index(c) + 1)

        source = source.permute(*order)

        # perform checks on the source tensor:
        _, num_timesteps, num_wl, num_sources, num_batches = source.shape

        # check if the number of wavelengths corresponds to the number of wavelengths in the environment
        if num_wl > 1 and num_wl != self.env.num_wavelengths:
            raise ValueError(
                "Number of wavelengths in the source does not correspond"
                " to the number of wavelengths in the environment."
            )

        # add zeros to the unused mc nodes:
        if num_sources < self.nmc:
            source = torch.cat(
                [
                    source,
                    torch.zeros(
                        (2, num_timesteps, num_wl, self.nmc - num_sources, num_batches),
                        dtype=source.dtype,
                        device=source.device,
                    ),
                ],
                -2,
            )

        # repeat last source value if num_timesteps < env.num_timesteps
        if num_timesteps < self.env.num_timesteps:
            repeated_source = source[:, -1, None] * torch.ones(
                (
                    2,
                    self.env.num_timesteps - num_timesteps,
                    num_wl,
                    self.nmc,
                    num_batches,
                ),
                device=source.device,
                dtype=source.dtype,
            )
            source = torch.cat([source, repeated_source], 1)

        return source

    def forward(self, source=1.0, power=True, detector=None, axes=None):
        """
        Forward pass of the network.

        Args:
            source: torch.Tensor: should ideally be a float tensor of shape
                (2 = (real|imag), #time, #wavelength, #sources, #batches)
                if the tensor shape does not match the ideal shape, photontorch will try
                to be smart enough to cast the tensor in the right shape.

            power: bool = True: choose to return the power as output or the complex
                valued fields. In the latter case, the complex components will be stacked
                along the first dimension (PyTorch does not support complex tensors (yet))

            detector: Detector = None: pass an extra detector instance to detect the fields

        Returns:
            detected: torch.Tensor: a tensor containin the the detected fields. This
                tensor has shape
                    (# time, # wavelengths, # detectors, # batches) if power==True
                 or (2 = (real|imag), # time, # wavelengths, # detectors, # batches) if power==False
        """

        # reinitialize the network if the current environment does not correspond
        # to the previous environment
        if self.env is not current_environment() or torch.is_grad_enabled():
            self.initialize()

        source = self.handle_source(source, axes=axes)

        num_batches = source.shape[-1]

        detected = torch.zeros(
            (
                self.env.num_timesteps,
                self.env.num_wavelengths,
                self.num_detectors,
                num_batches,
            ),
            device=self.device,
        )
        if not power:
            detected = torch.stack([detected, detected], 0)

        ## Get new simulation buffer
        buffer = self.simulation_buffer(num_batches)

        # solve
        for i, t in enumerate(self.env.time):
            det, buffer = self.step(t, source[:, i], buffer)

            if power:
                detected[i] = torch.sum(det ** 2, 0)
            else:
                detected[:, i] = det

        if detector is not None:
            detected = detector(detected)

        return detected

    def step(self, t, srcvalue, buffer):
        """ Single step forward pass through the network

        Args:
            t (float): the time of the simulation
            srcvalue (torch.tensor): The source value at the next timestep
            buffer (torch.tensor): The internal state of the network

        Returns:
            detected (torch.tensor): The detected fields
            buffer (torch.tensor): The internal state of the network after the step
        """
        # get state
        rx, ix = torch.sum(self.buffermask * buffer, dim=1)

        # connect memory-containing components
        # rx and ix need to be calculated at the same time because of dependencies on each other
        rx, ix = (
            (self._rS).bmm(rx) - (self._iS).bmm(ix),
            (self._rS).bmm(ix) + (self._iS).bmm(rx),
        )

        # add sources
        if self.num_actions == 0:
            rx = rx + srcvalue[0]
            ix = ix + srcvalue[1]
        else:
            x = torch.stack([rx, ix], 0)
            x = (x + srcvalue).permute(2, 0, 1, 3)
            x, x_in = torch.zeros_like(x), x
            self.action(t, x_in, x)
            rx, ix = x.permute(1, 2, 0, 3)

        # connect memory-less components
        # rx and ix need to be calculated at the same time because of dependencies on each other
        rx, ix = (
            (self._rC).bmm(rx) - (self._iC).bmm(ix),
            (self._rC).bmm(ix) + (self._iC).bmm(rx),
        )

        # update buffer
        x = torch.stack([rx, ix], 0)
        buffer = torch.cat((x[:, None], buffer[:, 0:-1]), dim=1)

        # get detected
        detected = x[:, :, -self.num_detectors :]

        return detected, buffer

    def action(self, t, x_in, x_out):
        """ The action of the active components on the network """
        x_out[:] = x_in[:]

        idxs = self.action_idxs
        for comp, i, j in zip(self.components_with_action, idxs[:-1], idxs[1:]):
            _x_in = x_in[i:j]
            _x_out = x_out[i:j]
            _x_out[:] = 0
            comp.action(t, _x_in, _x_out)

    def get_delays(self):
        """ get all the delays in the network """
        return torch.cat([comp.delays for comp in self.components.values()])

    def get_detectors_at(self):
        """ get the locations of the detectors in the network """
        return torch.cat([comp.detectors_at for comp in self.components.values()])

    def get_sources_at(self):
        """ get the locations of the sources in the network """
        return torch.cat([comp.sources_at for comp in self.components.values()])

    def get_actions_at(self):
        """ get the locations of the functions in the network """
        return torch.cat([comp.actions_at for comp in self.components.values()])

    def get_S(self):
        """ Get the combined S-matrix of all the components in the network """
        rS = block_diag(*(comp.S[0] for comp in self.components.values()))
        iS = block_diag(*(comp.S[1] for comp in self.components.values()))
        return torch.stack([rS, iS])

    def get_C(self):
        """ Combined Connection matrix of all the components in the network

        Returns:
            torch.FloatTensor with only 1's and 0's.

        Note:
            To create the connection matrix, the connection strings are parsed.
        """

        rC = block_diag(*(comp.C[0] for comp in self.components.values()))
        iC = block_diag(*(comp.C[1] for comp in self.components.values()))
        C = torch.stack([rC, iC])

        start_idxs = list(
            np.cumsum([0] + [comp.num_ports for comp in self.components.values()])[:-1]
        )
        start_idxs = OrderedDict(zip(self.components.keys(), start_idxs))
        free_idxs = [comp.free_idxs for comp in self.components.values()]
        free_idxs = OrderedDict(zip(self.components.keys(), free_idxs))

        def parse_connection(conn):
            conn_list = conn.split(":")
            if len(conn_list) < 4:
                return None, None
            comp1, i1, comp2, i2 = conn_list
            i1, i2 = int(i1), int(i2)
            if comp1 == comp2 and i1 == i2:
                raise IndexError("Cannot connect two equal ports")
            for i, comp in [(i1, comp1), (i2, comp2)]:
                if i >= len(free_idxs[comp]):
                    raise ValueError(
                        "Component %s only has %i ports. Port index "
                        "%i too high" % (comp, len(free_idxs[comp]), i)
                    )
            j1 = start_idxs[comp1] + free_idxs[comp1][i1]
            j2 = start_idxs[comp2] + free_idxs[comp2][i2]
            return int(j1), int(j2)

        for conn in self.connections:
            i, j = parse_connection(conn)
            if i is not None:
                C[0, i, j] = C[0, j, i] = 1.0

        return C

    def get_order(self):
        """ Get the reordering indices for the ports of the network """
        start_idxs = list(
            np.cumsum([0] + [comp.num_ports for comp in self.components.values()])[:-1]
        )
        start_idxs = OrderedDict(zip(self.components.keys(), start_idxs))
        free_idxs = [comp.free_idxs for comp in self.components.values()]
        free_idxs = OrderedDict(zip(self.components.keys(), free_idxs))

        def parse_connection(conn):
            conn_list = conn.split(":")
            if len(conn_list) == 4:
                return None, None
            comp, i, j = conn_list
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

    def __setattr__(self, name, attr):
        if isinstance(attr, Component):
            self.add_component(name, attr)
        else:
            super(Network, self).__setattr__(name, attr)

    def __enter__(self):
        """ enter the with block """
        # and add to _current_networks, so the components and networks will
        # be updated with the "link" function
        _current_networks.appendleft(self)
        return self

    def __exit__(self, error, value, traceback):
        """ exit the with block """
        if error is not None:
            raise  # raise the last error thrown
        del _current_networks[0]
        return self


#####################
## Current Network ##
#####################


def current_network():
    """ get the current network being defined """
    if _current_networks:
        return _current_networks[0]


def link(*ports):
    """ link ports together in the current network

    Args:
        *ports: the ports to link together. The first and last port can be an integer
        to specify the ordering of the network ports.

    Note:
        if more than two ports are specified, then the intermediate ports should be
        of the 'double port' type (i.e. "idx1:comp_name:idx2"). The first index will
        connect to the port before; the second index will connect to the port
        after.

    Note:
        This function can only be used inside a network-definition with-block.

    Example:
        >>> with pt.Network() as nw:
        >>>     nw.dc = pt.DirectionalCoupler()
        >>>     nw.wg = pt.Waveguide()
        >>>     nw.link(0, '0:dc:2','0:wg:1','3:dc:1', 1)


    """
    nw = current_network()
    nw.link(*ports)
