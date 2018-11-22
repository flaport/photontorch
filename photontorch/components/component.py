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
from ..environment import current_environment


###############
## Component ##
###############


class Component(Module):
    """ Generic base component.

        The base component is a parent class meant for subclassing; it should not be
        used directly.

        To define your own component, overwrite the __init__ method and the get_* methods.
    """

    num_ports = 0  # Number of ports of the component

    def __init__(self, name=None, _calculate_buffers=True):
        """ Component

        Args:
            name: str = None: the name of the component
        """
        Module.__init__(self)
        self.name = name

        # add component to current network if a component with that name does not yet exist
        nw = current_network()
        if nw is not None and nw is not self and self.name is not None:
            nw.add_component(self.name, self, copy=False)

        # set environment
        self._env = None

        # calculate buffers
        if _calculate_buffers:
            self.C = Buffer(self.get_C())
            self.sources_at = Buffer(self.get_sources_at())
            self.detectors_at = Buffer(self.get_detectors_at())
            self.actions_at = Buffer(self.get_actions_at())
            self.free_idxs = Buffer(self.get_free_idxs())
            self.terminated = len(self.free_idxs) == 0

    ## The following methods should be overwritten by subclasses:

    def set_S(self, S):
        """ Calculate the scattering matrix of the component

        Returns:
            S: torch.Tensor[2, #wavelengths, #ports, #ports]: the scattering matrix for
                each wavelength of the simulation.

        Note:
            S[0] is the real part, S[1] is the imaginary part.
        """
        pass

    def set_C(self, C):
        """ Calculate the connection matrix of the component.

        Note:
            C[0] is the real part, C[0] is the imaginary part
        """
        pass

    def set_delays(self, delays):
        """ Set the delays introduced by the component. """
        pass

    def set_sources_at(self, sources_at):
        """ Set the locations of the sources in the component. """
        pass

    def set_detectors_at(self, detectors_at):
        """ Set the locations of the detectors in the component. """
        pass

    def set_actions_at(self, actions_at):
        """ Set the locations of the active nodes in the component. """
        pass

    def action(self, t, x_in, x_out):
        """ Nonlinear action of the component on its active nodes

        Args:
            t: float: the current time in the simulation
            x_in: torch.Tensor[#active nodes, 2, #wavelengths, #batches]: the input tensor
                used to define the action
            x_out: torch.Tensor[#active nodes, 2, #wavelengths, #batches]: the output
                tensor. The result of the action should be stored in this tensor.

        Returns:
            None: (the result should be stored in the output tensor and should not be
            returned)
        """
        pass

    ## The following methods should be left alone:

    @property
    def env(self):
        """ Get the environment object for which the component is initialized """
        return self._env

    def initialize(self):
        """ Set the simulation initialization for the component.

        Before a component can be used for simulation, it should be initialized with a
        simulation environment.

        Args:
            env: Environment = None: Simulation environment to initialize the component
                with. If no environment is specified, the component will be initialized
                with the last defined environment (the result of current_environment())
        """
        self._env = env = current_environment()

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

    def get_S(self):
        """ get the scattering matrix of the component

        Returns:
            S: torch.Tensor[2, #wavelengths, #ports, #ports]: the scattering matrix for
                each wavelength of the simulation.

        Note:
            S[0] is the real part, S[1] is the imaginary part.
        """
        S = torch.zeros(
            (2, self.env.num_wavelengths, self.num_ports, self.num_ports),
            device=self.device,
        )
        self.set_S(S)
        return S

    def get_C(self):
        """ get the connection matrix of the component.

        Returns:
            C: torch.Tensor[2, #ports, #ports]: the connection matrix.

        Note:
            C[0] is the real part, C[0] is the imaginary part
        """
        C = torch.zeros((2, self.num_ports, self.num_ports), device=self.device)
        self.set_C(C)
        return C

    def get_delays(self):
        """ Get the delays introduced by the component.

        Returns:
            delays: torch.Tensor[#ports]: The delays in each node of the component.
        """
        delays = torch.zeros(self.num_ports, device=self.device)
        self.set_delays(delays)
        return delays

    def get_sources_at(self):
        """ Get the locations of the sources in the component.

        Returns:
            sources_at: torch.Tensor[#ports]: a uint8 tensor where the locations of the
                sources are denoted by a 1.
        """
        sources_at = torch.zeros(self.num_ports, device=self.device, dtype=torch.uint8)
        self.set_sources_at(sources_at)
        return sources_at

    def get_detectors_at(self):
        """ Get the locations of the detectors in the component.

        Returns:
            detectors_at: torch.Tensor[#ports]: a uint8 tensor where the locations of the
                detectors are denoted by a 1.
        """
        detectors_at = torch.zeros(
            self.num_ports, device=self.device, dtype=torch.uint8
        )
        self.set_detectors_at(detectors_at)
        return detectors_at

    def get_actions_at(self):
        """ Get the locations of the active nodes in the component.

        Returns:
            actions_at: torch.Tensor[#ports]: a uint8 tensor where the locations of the
                active nodes are denoted by a 1.
        """
        actions_at = torch.zeros(self.num_ports, device=self.device, dtype=torch.uint8)
        self.set_actions_at(actions_at)
        return actions_at

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
        s = self.__class__.__name__
        if self.name is not None:
            s = s + '(name="' + self.name + '")'
        return s

    def __str__(self):
        """ String Representation of the component """
        return repr(self)


#############
## Imports ##
#############

# placed here to prevent circular imports

from ..networks.network import current_network
