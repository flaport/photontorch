""" Photontorch network

The Network is the core of Photontorch: it is where everything comes together.

A network is created by subclassing it and linking its components together,
like so: ::


    class Circuit(Network):
        def __init__(self):
            self.src = pt.Source()
            self.det = pt.Detector()
            self.wg = pt.Waveguide()
            self.link("src:0", "0:wg:1", "0:det")
    circuit = Circuit()

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
from .visualize import plot, graph
from ..nn.nn import Buffer
from ..components.component import Component
from ..components.terms import Term
from ..nn.autograd import block_diag
from ..nn.tensor import where
from ..environment import current_environment


#############
## Globals ##
#############
_current_networks = deque()
""" a deque of networks currently being defined with a with-block """


#############
## Network ##
#############


class Network(Component):
    """ a Network (circuit) of Components

    The Network is the core of Photontorch. This is where everything comes
    together.  The Network is a special kind of torch.nn.Module, where all
    subcomponents are automatically initialized and connected in the right way.

    """

    # components and connections are defined here for easy subclassing
    components = None
    """ dictionary containing all the components of the network """

    connections = None
    """ list containing all the connection of the network """

    # network properties
    @property
    def num_ports(self):
        """ number of ports in the network """
        return sum(comp.num_ports for comp in self.components.values())

    # Network creation methods
    # ------------------------

    # network initialization method
    def __init__(
        self, components=None, connections=None, name=None, copy_components=False
    ):
        """ Network initialization

        Args:
            components (dict): a dictionary containing the components of the network.
                keys: str: new names for the components (will override the component name)
                values: Component: the component
            connections (list): a list containing the connections of the network.
                the connection string can have two formats: 1. "comp1:port1:comp2:port2": signifying a connection between ports
                (always reflexive) 2. "comp:port:output_port": signifying a connection to an output port index
            name (optional, str): name of the network
            copy_components (bool): create a deepcopy of each component before
                linking them together.

        Note:
            Although it's possible to initialize the network with a list of
            components and a list of connections. It's a lot easier to create a
            network with subclassing. For example::

                class Circuit(Network):
                    def __init__(self):
                        self.src = pt.Source()
                        self.det = pt.Detector()
                        self.wg = pt.Waveguide()
                        self.link("src:0", "0:wg:1", "0:det")
                circuit = Circuit()

        Note:
            If quick creation of a network is desired, a network can also be
            created with a with-block::

                with pt.Network() as circuit:
                    circuit.src = pt.Source()
                    circuit.det = pt.Detector()
                    circuit.wg = pt.Waveguide()
                    circuit.link("src:0", "0:wg:1", "0:det")

            However, creating networks by subclassing is often preferred as
            multiple instances of the network can easily be created after the
            network class is defined.

        """

        # initial network initialization without calculating the necessary buffers
        super(Network, self).__init__(name=name)

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
            self._register_connections()  # add components to the components dict

    def _set_buffers(self):
        pass  # do not calculate buffers here.

    # add a component to the network
    def add_component(self, name, comp, copy=True):
        """ Add a component to the network

        Pytorch requires submodules to be registered as attributes of a module.
        This method register a component as a torch module in the _modules
        dictionary.

        Args:
            name (optional, str): name of the component to add to the network
            copy (bool): copy the component before adding.

        Note:
            the following two lines are equivalent::

                nw.wg = pt.Waveguide()
                nw.add_component("wg", pt.Waveguide())

            The first one is often preferred for simple networks, whereas the
            latter can be useful when components are created in a loop.
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
            *ports[str]: the components and ports to link together.
                The following format is expected: "idx1:component:idx2".
                This format specifies the input port (idx1) and the output
                port (idx2) of a component to use in the chain.
                The chain can be preceded and followed by an integer,
                specifying the port indices of the resulting network.

        Example:
            >>> with pt.Network() as allpass:
            >>>     allpass.dc = pt.DirectionalCoupler()
            >>>     allpass.wg = pt.Waveguide()
            >>>     allpass.link(0, '0:dc:2', '0:wg:1', '3:dc:1', 1)

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
        """ get names of components that are actually used in the network's connections """

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
        """ create connection matrix from the network's list of connections """
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
        """ Terminate open conections with the Term of your choice

        Args:
            term: (Term|list|dict): Which term to use. Defaults to Term. If a
                dictionary or list is specified, then one needs to specify as
                many terms as there are open ports.

        Returns:
            terminated network.

        Note:
            the original (unterminated) network is always available with the
            ``.base`` attribute of the terminated network.
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
        """ remove termination of network

        Returns:
            unterminated network.
        """
        return self.base

    # Methods to prepare for simulation
    # ---------------------------------

    def initialize(self):
        r""" Initializer of the network.

        The goal of this initialization is to split the network into
        memory-containing nodes (nodes that introduce delay, sources,
        detectors, active components...) and memory-less nodes.  A matrix
        reduction method is then applied to remove the memory-less nodes from
        the S-matrix and C-matrix. It is with these reduced matrices that the
        simulation will then be performed.

        The resulting reduced matrices will also be reordered, such that source
        nodes will be the first nodes of the network and detector nodes will be
        the last nodes. This is done for performance reasons during simulation,
        but has the added benefit of easier access to the results afterwards.

        Note:
            during initialization, the reduced connection matrix is calculated
            for all batches and for all wavelengths. This is a quite verbose
            calculation, because the matrices need to be (batch) multiplied in
            the right way. Below, you can find the equation for the reduced
            connection matrix we're trying to calculate (will saved to network
            attribute ``._C``.

            .. math::

                C^{\rm red} = \left(1-C^{\rm mlml}S^{\rm mlml}\right)^{-1} C^{\rm mlmc}

        Note:
            Before any initialization can happen, all the ports of the
            comonents of the network need to be interconnected: the network
            needs to be fully connected (terminated). If the network is not
            fully connected, the initialization will silently fail.  This is to
            speed up the initialization of nested networks, where only the top
            level needs to be fully initialized.

        Note:
            Usually, calling ``initialize`` directly is not necessary.
            ``Network.forward`` calls ``initialize`` automatically. It could
            however be useful to call ``initialize`` if you want access to the
            reduced matrices without needing the response of the network to an
            input signal.
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
        delays_in_seconds = self.delays * float(not self.env.freqdomain)
        # resulting delays in terms of the simulation timestep:
        if self.env.dt is not None:
            delays = (delays_in_seconds / self.env.dt + 0.5).long()
        else:
            delays = torch.zeros_like(delays_in_seconds).long()

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
                "Simulation timestep might be too large, resulting in zero "
                "delays for nonzero lengths. Try using a smaller timestep",
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
            rC = torch.stack([rCmcmc] * self.env.num_wl, dim=0)
            iC = torch.stack([iCmcmc] * self.env.num_wl, dim=0)
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
            ones = torch.ones((self.env.num_wl, 1, 1), device=self.device)
            rP = ones * torch.eye(self.nml, device=self.device)[None, :, :]
            rP = rP - bmm(rCmlml, rSmlml) + bmm(iCmlml, iSmlml)
            iP = -bmm(rCmlml, iSmlml) - bmm(iCmlml, rSmlml)

            # 2. Calculate inv(P)@Cmlmc [using torch.solve]
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
        self.buffermask[
            :, self._delays - 1, :, range(self.nmc), :
        ] = 1.0  # delay == 1 -> index 0

        self.initialized = True

        # finish initialization:
        return self

    def _simulation_buffer(self, num_batches):
        """ Create cyclic buffer to keep the time-delayed states of the network.

        Args:
            num_batches (int): number of batches to create the buffer for

        Returns:
            torch.Tensor[2, #timesteps, #wavelengths, #mc nodes, num_batches]

        """
        buffer = torch.zeros(
            (2, int(self._delays.max()) + 1, self.env.num_wl, self.nmc, num_batches,),
            device=self.device,
        )
        return buffer

    def _handle_source(self, source):
        """ bring a source tensor in a usable form to use in forward pass.

        Args:
            source (Tensor): The source tensor to validate and handle.

        Returns:
            Tensor: The source tensor with shape (2, t, w, n, b) to be used
                during the forward pass. With 'n' being the number of MC nodes in
                the network.

        Note:
             The source tensor should have shape (t, w, s, b), with
               * t: the number of timesteps in the simulation environment.
               * w: the number of wavelengths in the simulation environment.
               * s: the number of sources in the network.
               * b: the number of unrelated input waveforms (the batch size).

             Alternatively, two of such tensors can be stacked together in dimension 0
             to represent the real and imaginary part of a complex tensor,
             resulting in a tensor of shape (2, t, w, s, b).

             Any lower dimensional tensor should have named dimensions to remove any
             ambiguity in the broadcasting rules. Dimensions of a tensor can be named
             with the ``.rename`` method of the PyTorch Tensor class.
             accepted dimension names are 'c', 't', 'w', 's', 'b'.
        """
        _note = self._handle_source.__doc__.split("Note:")[-1]
        _possible_names = ("c", "t", "w", "s", "b")

        if isinstance(source, np.ndarray):
            if source.dtype in (np.complex64, np.complex128, np.complex256):
                source = np.stack([np.real(source), np.imag(source)], axis=0)

        if not torch.is_tensor(source):
            try:
                source = torch.tensor(
                    source, dtype=torch.get_default_dtype(), device=self.device
                )
            except TypeError:
                raise ValueError("Cannot convert source to torch tensor.\n%s" % _note)

        if source.ndim > 5:
            raise ValueError("Source dimensionality too high.\n%s" % _note)

        if not all(source.names):
            if any(source.names):
                raise ValueError(
                    "Network does not accept source tensors with partly named dimensions. "
                    "Either name None of them (in which case a 4D or 5D tensor is expected) "
                    "or name all of them.\n%s" % _note
                )
            if source.ndim == 5:
                if source.shape[0] > 2:
                    raise ValueError(
                        "First dimension of a 5D source should "
                        "maximally be 2, containing the real and"
                        "the imaginary part.\n%s" % _note
                    )
            elif source.ndim == 4:
                source = source[None]
            elif source.ndim == 0:
                source = source[None, None, None, None, None]
            else:
                raise ValueError(
                    "No named dimensions found for low dimensional source\n%s" % _note
                )
            source = source.rename("c", "t", "w", "s", "b")

        for name in source.names:
            if name not in _possible_names:
                raise ValueError(
                    "name '%s' is not an allowed dimension name for a source. "
                    "allowed names are: %s\n%s"
                    % (name, str(_possible_names)[1:-1], _note)
                )

        names = list(source.names)
        source = source.rename(None)
        for name in _possible_names:
            if name not in names:
                source = source[..., None]
                names.append(name)
        source = source.rename(*names).align_to("c", "t", "w", "s", "b").rename(None)

        if source.shape[0] == 1:
            source = torch.cat([source, torch.zeros_like(source)], 0)
        if source.shape[1] > 1 and source.shape[1] != self.env.num_t:
            if source.shape[1] < self.env.num_t:
                source = torch.cat(
                    [
                        source,
                        torch.zeros(
                            (2, self.env.num_t - source.shape[0]) + source.shape[2:],
                            dtype=torch.get_default_dtype(),
                            device=source.device,
                        ),
                    ],
                    1,
                )
            else:
                source = source[:, : self.env.num_t]
        if source.shape[2] > 1 and source.shape[2] != self.env.num_wl:
            raise ValueError(
                "Source is defined for a different number of wavelengths than the number present in the simulation environment.\n%s"
                % _note
            )
        if source.shape[3] > 1 and source.shape[3] != self.num_sources:
            raise ValueError(
                "Source is defined for a different number of source nodes than the number present in the network\n%s"
                % _note
            )

        source_template = torch.empty(
            (2, self.env.num_t, self.env.num_wl, self.num_sources, 1),
            dtype=torch.get_default_dtype(),
            device=source.device,
        )

        source, _ = torch.broadcast_tensors(source, source_template)

        # we want zero source values for all other MC nodes.
        source = torch.cat(
            [
                source,
                torch.zeros(
                    source.shape[:3] + (self.nmc - source.shape[3], source.shape[4]),
                    dtype=torch.get_default_dtype(),
                    device=source.device,
                ),
            ],
            3,
        )

        return source

    def forward(self, source=0.0, power=True, detector=None):
        """ calculate the network's response to an applied source.

        Args:
            source (Tensor): The source tensor to calculate the response for.
            power (bool): Return detected power, otherwise return complex signal.
            detector (callable): Custom detector function to use to detect the signal.

        Returns:
            Tensor: The detected tensor with shape (t, w, s, b) or with
                shape (2, t, w, s, b) in the case of power=False (in that case, dimension
                0 contains the stacked real and imaginary part of the result)

        Note:
             The source tensor should have shape (t, w, s, b), with
               * t: the number of timesteps in the simulation environment.
               * w: the number of wavelengths in the simulation environment.
               * s: the number of sources in the network.
               * b: the number of unrelated input waveforms (the batch size).

             Alternatively, two of such tensors can be stacked together in dimension 0
             to represent the real and imaginary part of a complex tensor,
             resulting in a tensor of shape (2, t, w, s, b).

             Any lower dimensional tensor should have named dimensions to remove any
             ambiguity in the broadcasting rules. Dimensions of a tensor can be named
             with the ``.rename`` method of the PyTorch Tensor class.
             accepted dimension names are 'c', 't', 'w', 's', 'b'.

        """

        # reinitialize the network if the current environment does not correspond
        # to the previous environment
        if self.env is not current_environment() or torch.is_grad_enabled():
            self.initialize()

        source = self._handle_source(source)

        num_batches = source.shape[-1]

        detected = torch.zeros(
            (self.env.num_t, self.env.num_wl, self.num_detectors, num_batches,),
            device=self.device,
        )
        if not power:
            detected = torch.stack([detected, detected], 0)

        ## Get new simulation buffer
        buffer = self._simulation_buffer(num_batches)

        # solve
        for i, t in enumerate(self.env.t):
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
            srcvalue (Tensor): The source value at the next timestep
            buffer (Tensor): The internal state of the network

        Returns:
            detected (Tensor): The detected fields
            buffer (Tensor): The internal state of the network after the step
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
        """ Perform the action of an active components in the network """
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
        """ get the combined S-matrix of all the components in the network """
        rS = block_diag(*(comp.S[0] for comp in self.components.values()))
        iS = block_diag(*(comp.S[1] for comp in self.components.values()))
        return torch.stack([rS, iS])

    def get_C(self):
        """ get the combined connection matrix of all the components in the network

        Returns:
            binary tensor with only 1's and 0's.

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

    def plot(self, detected, **kwargs):
        """ Plot detected power versus time or wavelength

        Args:
            detected (np.ndarray|Tensor): detected power. Allowed shapes:
                * (#timesteps,)
                * (#timesteps, #detectors)
                * (#timesteps, #detectors, #batches)
                * (#timesteps, #wavelengths)
                * (#timesteps, #wavelengths, #detectors)
                * (#timesteps, #wavelengths, #detectors, #batches)
                * (#wavelengths,)
                * (#wavelengths, #detectors)
                * (#wavelengths, #detectors, #batches)
                the plot function should be smart enough to figure out what to plot.
            **kwargs: keyword arguments given to plt.plot

        Note:
            if #timesteps = #wavelengths, the plotting function will choose #timesteps
            as the first dimension

        """
        return plot(self, detected, **kwargs)

    def graph(self, draw=True):
        """ create a graph visualization of the network

        Args:
            draw (bool): draw the graph with matplotlib

        """
        return graph(self, draw=True)

    def __setattr__(self, name, attr):
        """ set attributes of the network """
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
    """ link components together

    Args:
        *ports: the ports to link together. The first and last port can be an integer
            to specify the ordering of the network ports.

    Note:
        if more than two ports are specified, then the intermediate ports
        should be of the 'double port' type (i.e. ``idx1:comp_name:idx2``).
        The first index will connect to the port before; the second index
        will connect to the port after.

    Example:
        >>> with pt.Network() as nw:
        >>>     nw.dc = pt.DirectionalCoupler()
        >>>     nw.wg = pt.Waveguide()
        >>>     nw.link(0, '0:dc:2','0:wg:1','3:dc:1', 1)

    """
    nw = current_network()
    nw.link(*ports)
