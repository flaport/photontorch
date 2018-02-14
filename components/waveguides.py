''' Waveguides Module '''

#############
## Imports ##
#############

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

    def __init__(self, length=1e-6, neff=2.86, loss=0, name=None):
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
        self.length = float(length)
        self.neff = float(neff)
        self.loss = loss

    @property
    def delays(self):
        ''' The delay per node is given by the propagation time in the waveguide '''
        delay = self.neff*self.length/c
        ones = self.new_variable([1, 1], 'float')
        return delay*ones

    @property
    def rS(self):
        ''' real part of the scattering matrix '''
        re = 10**(-self.loss*self.length/10)*np.cos(2*pi*self.neff*self.length/self.env.wls)
        S = np.array([[[0, 1],
                       [1, 0]]])
        return self.new_variable(re[:,None,None]*S)

    @property
    def iS(self):
        ''' imaginary part of the scattering matrix '''
        # e = exp(2j*pi*neff*length/wl)
        ie = 10**(-self.loss*self.length/10)*np.sin(2*pi*self.neff*self.length/self.env.wls)
        S = np.array([[[0, 1],
                       [1, 0]]])
        return self.new_variable(ie[:,None,None]*S)
