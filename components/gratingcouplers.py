'''
# Grating Couplers

Grating couplers are a special kind of 2-port component that simulate the behavior of
coupling light from a fiber onto the chip.

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
from ..constants import fwhm2sigma


#####################
## Grating Coupler ##
#####################


class GratingCoupler(Component):
    r''' A Grating Coupler is a memory-less component with one input and one output.

    A Grating Coupler has one trainable parameter: the reflectivity R.

    Terms:

          0
           \
            \
        _|-|_|-|_|-|___ 1

    Note:
        A Grating Coupler is not trainable (for now).
    '''

    num_ports = 2

    def __init__(self, R=0.0, R_in=0.0, Tmax=1.0, BW=0.06e-6, wl0=1.55e-6, name=None):
        ''' Grating Coupler initialization

        Args:
            R (float): Reflection of the grating coupler (between 0 and 1)
            R_in (float): Incoupling Reflection for the grating coupler
            BW (float): 3dB Bandwidth of the grating coupler
            wl0 (float): Center wavelength of the grating coupler
            Tmax (float): Maximum transmission at center wavelength
            name (str): name of this specific mirror
        '''
        Component.__init__(self, name=name)
        self.R = R
        self.R_in = R_in
        self.BW = BW
        self.wl0 = wl0
        self.Tmax = Tmax

    def get_rS(self):
        ''' Real part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        sigma = fwhm2sigma*self.BW
        loss = np.sqrt(self.Tmax*np.exp(-(self.wl0-self.env.wls)**2/(2*sigma**2)))
        S = np.array([[[0, 1],
                       [1, 0]]])
        return self.tensor(loss[:,None,None]*S)

    def get_iS(self):
        ''' Imag part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        S = self.tensor([[[self.R_in, 0],
                                [0,    self.R]]])
        return torch.cat([S]*self.env.num_wl, dim=0)
