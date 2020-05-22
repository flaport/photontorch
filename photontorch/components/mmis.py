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

    def __init__(self, weights=None, trainable=True, name=None):
        """
        Mmi initialization

        Args:
            weights (np.ndarray): interconnection weights between input
                wageguides and output waveguides. If no weights are specified, the
                component defaults to a 1x2 splitter.
            trainable (bool): makes the interconnection weights trainable
            name (optional, str): the name of the component (default: lowercase classname)
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
