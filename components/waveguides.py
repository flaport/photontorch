''' Waveguides Module '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .component import Component
from ..utils.constants import pi, c


###############
## Waveguide ##
###############

class Waveguide(Component):
    '''
    A waveguide is a Component where each of the two nodes
    introduces a delay corresponding to the length of the waveguide.
    
    A waveguide is not trainable (for now)

    '''
    def __init__(self, length=1e-6, neff=4.0, loss=0, wl=1.55e-6, name=None):
        '''
        Waveguide Initialization

        Parameters
        ----------
        length : float. Length of the waveguide in meter.
        neff = 4.0 : float. Effective index of the waveguide 
        loss = 0 : float. Loss in the waveguide [dB]
        wl = 1.55e-6 : float. Wavelength of the simulation
        name : str. Name of the specific waveguide
        '''
        Component.__init__(self, name=name)
        # Handle inputs
        self.length = float(length)
        self.neff = float(neff)
        self.wl = float(wl)
        self.loss = loss

    @property
    def delays(self):
        ''' The delay per node is given by the propagation time in the waveguide '''
        return (self.neff*self.length/c)*self.new_variable([1,1],'float')

    @property
    def rS(self):
        ''' real part of the scattering matrix '''
        # e = exp(2j*pi*neff*length/wl)
        re = 10**(-self.loss/10)*np.cos(2*pi*self.neff*self.length/self.wl)
        return self.new_variable([[0,1],[1,0]])*re

    @property
    def iS(self):
        ''' imaginary part of the scattering matrix '''
        # e = exp(2j*pi*neff*length/wl)
        ie = 10**(-self.loss/10)*np.sin(2*pi*self.neff*self.length/self.wl)
        return self.new_variable([[0,1],[1,0]])*ie
