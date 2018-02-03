''' Directional Couplers Module '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .mirrors import Mirror
from .component import Component
from ..utils.constants import pi, c


#########################
## Directional Coupler ##
#########################

class DirectionalCoupler(Mirror):
    '''
    A directional coupler is a memory-less component with 4 ports.

    A directional coupler has one trainable parameter: the coupling R.

    Connections
    ------------
    dircoup['ijkl']:
     j        l
      \______/
      /------\
     i        k

    Note
    ----
     - This directional coupler introduces no delays (for now)
    '''

    num_ports = 4

    @property
    def rS(self):
        t = (1-self.R)**0.5
        return self.new_variable([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])*t

    @property
    def iS(self):
        r = self.R**0.5
        return self.new_variable([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])*r


class RealisticDirectionalCoupler(Component):
    '''
    A directional coupler is a memory-less component with 4 ports.

    Connections
    ------------
    dircoup['ijkl']:
     j        l
      \______/
      /------\
     i        k

    Notes
    -----
     - This directional coupler introduces no delays (for now)
     - This directional coupler is not trainable (for now)
    '''

    num_ports = 4

    def __init__(self,
                 length = 12.8e-6,
                 k0=0.2332,
                 n0=0.0208,
                 de1_k0=1.2435,
                 de1_n0=0.1169,
                 de2_k0=5.3022,
                 de2_n0=0.4821,
                 name=None):
        '''
        Initialization of the Realistic Directional Coupler.

        The default parameters are based on a directional coupler with
           - bend radius : 5um
           - adiabatic angle : 10 degrees
           - gap distance : 250 nm

        Parameters
        ----------
        k0 : bend coupling
        n0 : effective index difference between even and odd mode
        de1_k0 : first derivative of k0 w.r.t. wavelength
        de1_n0 : first derivative of n0 w.r.t. wavelength
        de2_k0 : second derivative of k0 w.r.t. wavelength
        de2_n0 : second derivative of n0 w.r.t. wavelength
        '''

        Component.__init__(name=name)
        self.k0 = k0
        self.de1_k0 = de1_k0
        self.de2_k0 = de2_k0
        self.n0 = n0
        self.de1_n0 = de1_n0
        self.de2_n0 = de2_n0
        self.wl0 = 1.55e-6

    @property
    def rS(self):
        wl = self.env.wl
        dwl = wl - self.wl0
        dn = self.n0 + self.de1_n0*dwl + 0.5*self.de2_n0*dwl**2
        kappa0 = self.k0 + self.de1_k0*dwl + 0.5*self.de2_k0*dwl**2
        kappa1 = pi*dn/wl
        tau = np.cos(kappa0 + kappa1*self.length)
        return self.new_variable([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])*tau

    @property
    def iS(self):
        wl = self.env.wl
        dwl = wl - self.wl0
        dn = self.n0 + self.de1_n0*dwl + 0.5*self.de2_n0*dwl**2
        kappa0 = self.k0 + self.de1_k0*dwl + 0.5*self.de2_k0*dwl**2
        kappa1 = pi*dn/wl
        kappa = -np.sin(kappa0 + kappa1*self.length)
        return self.new_variable([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])*kappa
