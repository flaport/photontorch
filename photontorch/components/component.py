"""
# Photontorch base component

The base component is a parent class meant for subclassing. It should not be used
directly.

Each Component is generally defined by several key attributes defining the behavior
of the component in a network.

    `num_ports`: The number of ports of the components.

    `S`: The scattering matrix of the component.

    `C`: The connection matrix for the component (usually all zero for base components)

    `sources_at`: The location of the sources in the component (usually all zero for
        base components)

    `detectors_at`: The location of the detectors in the component (usually all zero
        for base components)

    `actions_at`: The location of the active nodes in the component (usually all zero
        for passive components)

    `delays`: delays introduced by the nodes of the component.

"""

#############
## Imports ##
#############

## Torch
import torch

## Relative
from ..torch_ext import where
from ..torch_ext.nn import Module
from ..torch_ext.nn import Buffer


###############
## Component ##
###############


class Component(Module):
    """ Generic base component.

        The base component is a parent class meant for subclassing; it should not be
        used directly.

        To define your own component, overwrite the __init__ method and the get_* methods.
    """

    num_ports = 1  # Number of ports of the component

    def __init__(self, name=None):
        """ Component

        Args:
            name: str = None: the name of the component (default: lowercase classname)
        """
        Module.__init__(self)
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name

        # set attributes
        self._env = None
        self.C = Buffer(self.get_C())
        self.free_idxs = Buffer(self.get_free_idxs())
        self.sources_at = Buffer(self.get_sources_at())
        self.detectors_at = Buffer(self.get_detectors_at())
        self.actions_at = Buffer(self.get_actions_at())

    def initialize(self, env=None):
        """ Set the simulation initialization for the component.

        Before a component can be used for simulation, it should be initialized with a
        simulation environment.

        Args:
            env: Environment = None: Simulation environment to initialize the component
                with. If no environment is specified, the component will be initialized
                with the previous environment the component was initialized with.
        """

        if env is None:
            env = self._env
        else:
            self._env = env

        if env.device is not None:
            self.to(env.device)

        self.zero_grad()
        if (self.sources_at & self.detectors_at).any():
            raise ValueError(
                "Sources and Detectors cannot be combined in the same node."
            )

        self.delays = self.get_delays()
        self.S = self.get_S()

        return self  # return the initialized component, so operations can be chained

    @property
    def env(self):
        """ Get the environment object for which the component is initialized """
        return self._env

    def get_S(self):
        """ Calculate the scattering matrix of the component

        Returns:
            S: torch.Tensor[2, #wavelengths, #ports, #ports]: the scattering matrix for
                each wavelength of the simulation.

        Note:
            S[0] is the real part, S[1] is the imaginary part.
        """
        return torch.zeros(
            (2, self.env.num_wavelengths, self.num_ports, self.num_ports),
            device=self.device,
        )

    def get_C(self):
        """ Calculate the connection matrix of the component.

        Returns:
            C: torch.Tensor[2, #ports, #ports]: the connection matrix.

        Note:
            C[0] is the real part, C[0] is the imaginary part
        """
        return torch.zeros((2, self.num_ports, self.num_ports), device=self.device)

    def get_delays(self):
        """ Get the delays introduced by the component.

        Returns:
            delays: torch.Tensor[#ports]: The delays in each node of the component.
        """
        return torch.zeros(self.num_ports, device=self.device)

    def get_sources_at(self):
        """ Get the locations of the sources in the component.

        Returns:
            sources_at: torch.Tensor[#ports]: a uint8 tensor where the locations of the
                sources are denoted by a 1.
        """
        return torch.zeros(self.num_ports, device=self.device, dtype=torch.uint8)

    def get_detectors_at(self):
        """ Get the locations of the detectors in the component.

        Returns:
            detectors_at: torch.Tensor[#ports]: a uint8 tensor where the locations of the
                detectors are denoted by a 1.
        """
        return torch.zeros(self.num_ports, device=self.device, dtype=torch.uint8)

    def get_actions_at(self):
        """ Get the locations of the active nodes in the component.

        Returns:
            actions_at: torch.Tensor[#ports]: a uint8 tensor where the locations of the
                active nodes are denoted by a 1.
        """
        return torch.zeros(self.num_ports, device=self.device, dtype=torch.uint8)

    def get_free_idxs(self):
        """ Calculate locations of the free indices to make a connections to.

        Returns:
            actions_at: torch.Tensor[#ports]: a int64 tensor containing the locations
                of the open ports denoted by a 1.
        """
        C = (abs(self.C) ** 2).sum(0)
        return where(((C.sum(0) > 0) | (C.sum(1) > 0)).ne(1))

    def __repr__(self):
        """ String Representation of the component """
        return self.name

    def __str__(self):
        """ String Representation of the component """
        return self.name

    def __getitem__(self, key):
        """ Get a connector item with certain key

        Each component contains a special __getitem__, which is solely used to connect
        components together. Indexing a component with a string index will create a
        connector with that same index. This connector will connect to any other connector
        where the same string index is used. Connected connectors can be made into a
        network. See Network for more information.

        Args:
            key (str): string key of the connection.

        """

        from ..networks.connector import Connector

        return Connector(key, [self])
