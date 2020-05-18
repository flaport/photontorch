""" A Connection connects two ports without reflection.

It is generally recommended to make direct connections in a network.
But if for some reason, you're unable to do so, a Connection object can
be used.

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

    Terms::

        0---1

    Note:
        It is generally recommended to make direct connections in a network.
        But if for some reason, you're unable to do so, a Connection object can
        be used.

    """

    num_ports = 2

    def set_S(self, S):
        S[0, :, 0, 1] = 1.0
        S[0, :, 1, 0] = 1.0
