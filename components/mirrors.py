"""
# Mirrors

Mirrors are partly reflecting and partly transmitting.

"""

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
    """ A mirror is a memory-less component with one input and one output.

    A mirror has one trainable parameter: the reflectivity R.

    Terms:
            |
        0 --|-- 1
            |
    """

    num_ports = 2

    def __init__(self, R=0.5, trainable=True, name=None):
        """ Mirror

        Args:
            R: float = 0.5: reflectivity of the mirror (between 0 and 1)
            trainable: bool = True: makes the coupling trainable
            name: str = None: the name of the component (default: lowercase classname)
        """
        super(Mirror, self).__init__(name=name)

        self.R = BoundedParameter(
            data=torch.tensor(R, device=self.device),
            bounds=(0, 1),
            requires_grad=trainable,
        )

    def get_S(self):
        r = torch.cat([(self.R ** 0.5).view(1, 1, 1)] * self.env.num_wavelengths, dim=0)
        t = torch.cat(
            [((1 - self.R) ** 0.5).view(1, 1, 1)] * self.env.num_wavelengths, dim=0
        )
        rS = r * torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=self.device)
        iS = t * torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], device=self.device)
        return torch.stack([rS, iS])
