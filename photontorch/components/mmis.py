""" An MMI performs a weighted transition from m waveguides to n waveguides

The MMI implemented in this module performs an arbitrary splitting operation specified by
a weight matrix

"""

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .component import Component
from ..torch_ext.nn import Buffer, Parameter


#########
## MMI ##
#########


class Mmi(Component):
    r""" An MMI performs a weighted transition from m waveguides to n waveguides

    Terms::

              m+1
         0___/ m+2
         1____/
          :  :
         m____
              \
               m+n

    """

    num_ports = None  # not yet defined

    def __init__(self, weights=None, trainable=False, name=None):
        """
        Mmi initialization

        Args:
            weights (np.ndarray): interconnection weights between input
                wageguides and output waveguides. If no weights are specified, the
                component defaults to a 1x2 splitter.
            trainable (bool): makes the interconnection weights trainable
            name (str): the name of the component (default: lowercase classname)
        """

        # validate weights
        if weights is None:
            weights = np.sqrt(0.5 * np.ones((1, 2)))

        if len(weights.shape) != 2 and len(weights.shape) != 3:
            raise ValueError(
                "weights should be at least 2D and at most 3D [first index real|imag]."
            )

        if not torch.is_tensor(weights):
            weights = torch.tensor(
                np.stack([np.real(weights), np.imag(weights)], 0),
                dtype=torch.get_default_dtype(),
            )

        if len(weights.shape) == 2:
            weights = torch.stack([weights, torch.zeros_like(weights)], 0)

        # weights should now be 3D: shape = [2=(real|imag), m, n].
        _, m, n = weights.shape
        self.num_ports = m + n

        # after num_ports is defined, we can initialize the component
        super(Mmi, self).__init__(name=name)

        parameter = Parameter if trainable else Buffer
        self.weights = parameter(weights)

    def set_S(self, S):
        weights = self.weights[:, None, :, :]  # same weight for each wavelength
        _, _, m, n = weights.shape
        S[:, :, :m, m:] = weights
        S[:, :, m:, :m] = torch.transpose(weights, -1, -2)


class PhaseArray(Component):
    r""" A Phase Array adds a phase to n inputs

    Terms::

              n+1
         0___/ n+2
         1____/
          :  :
         n____
              \
               2*n

    Note:
        This component is used in bigger networks. On itself it's not immediately useful.

    """

    num_ports = None

    def __init__(self, phases=None, length=1e-5, ng=3.40, trainable=True, name=None):
        """
        Args:
            length (float): Length of the waveguide in meter.
            ng (float): group index of the phase array
            trainable (bool): makes the phases trainable
            name (str): the name of the component (default: lowercase classname)
        """
        if phases is None:
            phases = np.zeros(2)

        if len(phases.shape) != 1:
            raise ValueError("phases should be a 1D array or tensor")

        self.num_ports = 2 * phases.shape[0]

        super(PhaseArray, self).__init__(name=name)

        self.ng = float(ng)
        self.length = float(length)

        parameter = Parameter if trainable else Buffer
        phases = torch.tensor(phases, dtype=torch.float64, device=self.device)
        self.phases = parameter(phases)

    def set_delays(self, delays):
        delays[:] = self.ng * self.length / self.env.c

    def set_S(self, S):
        n = self.num_ports // 2
        cos_phase = torch.diag(torch.cos(self.phases).to(torch.get_default_dtype()))
        sin_phase = torch.diag(torch.sin(self.phases).to(torch.get_default_dtype()))
        phase = torch.stack([cos_phase, sin_phase], 0)[:, None, :, :]
        S[:, :, :n, n:] = phase
        S[:, :, n:, :n] = phase
