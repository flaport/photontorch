''' Waveguides Module '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .connection import Connection
from ..constants import pi, c

###############
## Waveguide ##
###############

class Waveguide(Connection):
    '''
    A waveguide is a Component where each of the two nodes
    introduces a delay corresponding to the length of the waveguide.

    A waveguide is not trainable (for now)

    Connections
    -----------
    waveguide['ij']:

    i ---- j

    '''

    def __init__(self, length=1e-6, neff=2.86, loss=0, length_bounds=None, name=None):
        '''
        Waveguide Initialization

        Parameters
        ----------
        length : float. Length of the waveguide in meter.
        neff = 4.0 : float. Effective index of the waveguide
        loss = 0 : float. Loss in the waveguide [dB/m]
        name : str. Name of the specific waveguide
        '''
        Connection.__init__(self, name=name)
        # Handle inputs
        self.neff = float(neff)
        self.loss = loss
        # as the phase is very sensitive on the length, we need double precision to
        # store the length of the waveguide:
        self.length = self.new_bounded_parameter(
            data=length,
            bounds=length_bounds,
            dtype='double',
            requires_grad=(length_bounds is not None) and (length_bounds[0]!=length_bounds[1]),
        )

    @property
    def delays(self):
        ''' The delay per node is given by the propagation time in the waveguide '''
        delay = self.neff*self.length/c
        return torch.cat([delay, delay]).float()

    @property
    def rS(self):
        ''' real part of the scattering matrix '''
        wls = self.new_variable(self.env.wls, dtype='double')
        re = 10**(-self.loss*self.length/10)*torch.cos(2*pi*self.neff*self.length/wls)
        # we can safely convert back to single precision now:
        re = re.float()
        S = self.new_variable([[[0, 1],
                                [1, 0]]])
        return re.view(-1,1,1)*S

    @property
    def iS(self):
        ''' imaginary part of the scattering matrix '''
        wls = self.new_variable(self.env.wls, dtype='double')
        ie = (10**(-self.loss*self.length/10)*torch.sin(2*pi*self.neff*self.length/wls))
        # we can safely convert back to single precision now:
        ie = ie.float()
        S = self.new_variable([[[0, 1],
                                [1, 0]]])
        return ie.view(-1,1,1)*S
