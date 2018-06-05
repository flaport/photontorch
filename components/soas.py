
'''
# SOAs

'''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .component import Component


################
## Simple SOA ##
################

class SimpleSoa(Component):
    ''' A Simple SOA is a memory-less component with one input and one output.

    It amplifies a signal instantaneously with the specified amplification factor

    A simple SOA has one trainable parameter: the amplification.

    Connections:
        soa['ij']:

        i ---- j

    '''

    num_ports = 2

    def __init__(self, amplification=0.5, amplification_bounds=None, name=None):
        ''' Mirror

        Args:
            R (float). Reflectivity of the mirror (between 0 and 1)
            R_bounds (tuple): Bounds in which to optimize R. If None, R will not be optimized.
            name (str): name of this specific mirror
        '''
        Component.__init__(self, name=name)

        no_bounds = amplification_bounds is None
        self.amplification = self.new_bounded_parameter(
            data=amplification,
            bounds=amplification_bounds,
            requires_grad=no_bounds or ((not no_bounds) and (amplification_bounds[0]!=amplification_bounds[1])),
        )

    @property
    def rS(self):
        ''' Real part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        a = torch.cat([(1*self.amplification).view(1,1,1)]*self.env.num_wl, dim=0)
        S = self.new_tensor([[[0, 1],
                                [1, 0]]])
        return a*S

    @property
    def iS(self):
        ''' Imag part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        a = torch.cat([(1*self.amplification).view(1,1,1)]*self.env.num_wl, dim=0)
        S = self.new_tensor([[[0, 0],
                                [0, 0]]])
        return a*S
