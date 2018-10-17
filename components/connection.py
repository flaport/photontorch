"""
# Connection Module

A Connection is a special kind of component that connects two ports without reflection.

It is generally recommended to make connections with the connection matrix, but if it's
not possible to do so directly, a Connection object can be used.

"""

#############
## Imports ##
#############

## Torch
import torch

## Relative
from .component import Component


################
## Connection ##
################


class Connection(Component):
    """ A connection connects two ports without delays and without reflection.

    Terms:

        0---1

    Note:
        It is generally recommended to make connections with the connection matrix, but if it's
        not possible to do so directly, ac Connection object can be used.

    """

    num_ports = 2

    def get_S(self):
        ones = torch.ones((2, self.env.num_wavelengths, 1, 1), device=self.device)
        eye = torch.eye(2, device=self.device).flip(1)[None, None]
        return ones * eye
