""" Mirrors are partly reflecting and partly transmitting connections.  """

#############
## Imports ##
#############

## Torch
import torch

## Relative
from .component import Component
from ..torch_ext.nn import BoundedParameter


############
## Mirror ##
############


class Mirror(Component):
    """ Mirrors are partly reflecting and partly transmitting connections.

    A mirror has one trainable parameter: the reflectivity R.

    Terms::

            |
        0 --|-- 1
            |

    """

    num_ports = 2

    def __init__(self, R=0.5, trainable=True, name=None):
        """
        Args:
            R (float): reflectivity of the mirror (between 0 and 1)
            trainable (bool): makes the reflection trainable
            name (str): the name of the component (default: lowercase classname)
        """
        super(Mirror, self).__init__(name=name)

        self.R = BoundedParameter(
            data=torch.tensor(R, device=self.device),
            bounds=(0, 1),
            requires_grad=trainable,
        )

    def set_S(self, S):
        r = self.R ** 0.5
        t = (1 - self.R) ** 0.5

        # real part
        S[0, :, 0, 0] = r
        S[0, :, 1, 1] = r

        # imag part
        S[1, :, 0, 1] = t
        S[1, :, 1, 0] = t
