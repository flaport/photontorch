""" The base component is a parent class meant for subclassing.

It should not be used directly.

"""

#############
## Imports ##
#############

## Torch
import torch

## Relative
from ..nn.nn import Module
from ..nn.nn import Buffer
from ..environment import current_environment


###############
## Component ##
###############


class Component(Module):
    """ Generic base component.

        The base component is a parent class meant for subclassing;
        it should not be used directly.

        To define your own component, subclass Component and overwrite the
        __init__ and the set_* methods.
    """

    num_ports = 0
    """ Number of ports of the component. """

    def __init__(self, name=None):
        """ Component

        Args:
            name (optional, str): the name of the component
        """
        super(Component, self).__init__()
        self.name = name

        # add component to current network if a component with that name does not yet exist
        nw = current_network()
        if nw is not None and nw is not self and self.name is not None:
            nw.add_component(self.name, self)

        # set environment
        self._env = None

        # set buffers.
        self._set_buffers()

    def _set_buffers(self):
        """ create all buffers for the component """
        self.C = Buffer(
            torch.zeros((self.num_ports, self.num_ports), device=self.device)
        )
        self.sources_at = Buffer(
            torch.zeros(self.num_ports, device=self.device, dtype=torch.bool)
        )
        self.detectors_at = Buffer(
            torch.zeros(self.num_ports, device=self.device, dtype=torch.bool)
        )
        self.actions_at = Buffer(
            torch.zeros(self.num_ports, device=self.device, dtype=torch.bool)
        )
        self.set_C(self.C.data)
        self.set_sources_at(self.sources_at.data)
        self.set_detectors_at(self.detectors_at)
        self.set_actions_at(self.actions_at)
        self.free_ports_at = Buffer(((self.C.sum(0) > 0) | (self.C.sum(1) > 0)).ne(1))
        self.terminated = bool(self.free_ports_at.any().ne(1).item())
        self.num_sources = int(self.sources_at.sum())
        self.num_detectors = int(self.detectors_at.sum())
        self.num_actions = int(self.actions_at.sum())
        self.num_free_ports = int(self.free_ports_at.sum())

        if (self.sources_at & self.detectors_at).any():
            raise ValueError("source ports and detector ports cannot be combined.")
        if (self.sources_at & self.actions_at).any():
            raise ValueError("source ports and active ports cannot be combined.")
        if (self.detectors_at & self.actions_at).any():
            raise ValueError("detector ports and active ports cannot be combined.")

        if self.actions_at.any() and not (
            self.actions_at.all() or isinstance(self, Network)
        ):
            raise ValueError(
                "Sources and Detectors cannot be combined in the same node."
                "Active components should have an action defined for all their ports"
            )

    ## The following methods should be overwritten by subclasses:

    def set_S(self, S):
        """ Set the elements of the scattering matrix.

        Args:
            S (Tensor[2, #wavelengths, #ports, #ports]): the empty scattering
                matrix of the component to set the elements for (defined for each
                wavelength of the simulation).  The first dimension of size two
                denotes the stacked real and imaginary part.

        """
        pass

    def set_C(self, C):
        """ Set the connection matrix of the component.

        Args:
            C (Tensor[2, #ports, #ports]): the empty connection matrix for the
                component to set the elements for. The first dimension of size
                two denotes the stacked real and imaginary part.

        Note:
            For most base components the connection matrix should stay empty.
            Only for composite components, like ``Network``, the connection
            matrix will be non-empty.

        """
        pass

    def set_delays(self, delays):
        """ Set the delays introduced by the component.

        Args:
            delays (Tensor[#ports]): the empty delay tensor for the component to
                set the elements for. The delay tensor signifies the delay each
                port of the component introduces.
        """
        pass

    def set_sources_at(self, sources_at):
        """ Set the locations of the source ports in the component.

        Args:
            sources_at (Tensor[#ports]): the empty boolean tensor for the component to
                set the elements for. The ``sources_at`` tensor signifies which
                ports of the component act as a source.
        """
        pass

    def set_detectors_at(self, detectors_at):
        """ Set the locations of the detector ports in the component.

        Args:
            detectors_at (Tensor[#ports]): the empty boolean tensor for the component to
                set the elements for. The ``detectors_at`` tensor signifies which
                ports of the component act as a detector.
        """
        pass

    def set_actions_at(self, actions_at):
        """ Set the locations of the active ports in the component.

        Args:
            actions_at (Tensor[#ports]): the empty boolean tensor for the component to
                set the elements for. The ``actions_at`` tensor signifies which
                ports of the component act actively.
        """
        pass

    def action(self, t, x_in, x_out):
        """ Nonlinear action of the component on its active nodes

        Args:
            t (float): the current time in the simulation
            x_in (torch.Tensor[#active nodes, 2, #wavelengths, #batches]): the input tensor
                used to define the action
            x_out (torch.Tensor[#active nodes, 2, #wavelengths, #batches]): the output
                tensor. The result of the action should be stored in the
                elements of this tensor.

        """
        pass

    ## The following methods should NOT be overwritten with subclassing:

    @property
    def env(self):
        """ Get the environment object for which the component is initialized """
        return self._env

    def initialize(self):
        """ initialize the component with the current simulation environment.

        Before a component can be used for simulation, it should be initialized with a
        simulation environment.

        """
        self._env = env = current_environment()
        return self

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

from ..networks.network import Network, current_network
