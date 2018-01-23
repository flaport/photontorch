
''' Grating Couplers Module '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .component import Component
from ..utils.constants import pi, c, fwhm2sigma


#####################
## Grating Coupler ##
#####################


class GratingCoupler(Component):
    '''
    A Grating Coupler is a memory-less component with one input and one output.

    A Grating Coupler has one trainable parameter: the reflectivity R.

    Connections
    -----------
    gc['ij']:

       i
        \
         \
    _|¯|_|¯|_|¯|___ j
    '''
    def __init__(self, R=0.0, R_in=0.0, Tmax=1.0, BW=0.06e-6, wl0=1.55e-6, wl=1.55e-6, name=None):
        '''
        Mirror initialization

        Parameters (untrainable for now)
        ----------
        R : float. Reflection of the grating coupler (between 0 and 1)
        R_in : float. Incoupling Reflection for the grating coupler
        BW : float. 3dB Bandwidth of the grating coupler
        wl0 : float. Center wavelength of the grating coupler
        wl: float. Wavelength of the simulation
        Tmax : float. Maximum transmission at center wavelength
        name : str. name of this specific mirror
        '''
        Component.__init__(self, name=name)
        self.R = R
        self.R_in = R_in
        self.BW = BW
        self.wl0 = wl0
        self.wl = wl
        self.Tmax = Tmax

    @property
    def rS(self):
        ''' Real part of S matrix '''
        sigma = fwhm2sigma*self.BW
        loss = self.Tmax*np.exp(-(self.wl0-self.wl)**2/(2*sigma**2))
        return self.new_variable([[0,1],[1,0]])*loss

    @property
    def iS(self):
        ''' Imag part of S matrix '''
        return self.new_variable([[0,0],[0,0]])

