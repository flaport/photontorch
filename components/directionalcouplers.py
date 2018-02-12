''' Directional Couplers Module '''

#############
## Imports ##
#############

## Other
import numpy as np

## Relative
from .component import Component
from ..constants import pi


#########################
## Directional Coupler ##
#########################

class DirectionalCoupler(Component):
    r'''
    A directional coupler is a memory-less component with 4 ports.

    A directional coupler has one trainable parameter: the coupling R.

    Connections
    ------------
    dircoup['ijkl']:
     k        l
      \______/
      /------\
     i        j

    Note
    ----
     - This directional coupler introduces no delays (for now)
    '''

    num_ports = 4

    def __init__(self, kappa2=0.5, kappa2_bounds=(0, 1), name=None):
        '''
        Directional Coupler initialization

        Parameters
        ----------
        kappa2 : float. squared coupling of the directional coupler (between 0 and 1)
        kappa2_bounds : tuple of length 2: Bounds in which to optimize the squared coupling.
                        If None, kappa2 will not be optimized.
        name : str. name of this specific directional coupler
        '''
        Component.__init__(self, name=name)

        self.kappa2 = self.new_bounded_parameter(
            data=kappa2,
            bounds=kappa2_bounds,
            requires_grad=True, # trainable between bounds
        )

    @property
    def rS(self):
        t = (1-self.kappa2)**0.5
        S = self.new_variable([[0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]])
        return t*S

    @property
    def iS(self):
        k = self.kappa2**0.5
        S = self.new_variable([[0, 0, 1, 0],
                               [0, 0, 0, 1],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0]])
        return k*S

class RealisticDirectionalCoupler(Component):
    r'''
    A directional coupler is a memory-less component with 4 ports.

    The realistic directional coupler is based on the CapheModel of Umar Khan

    Connections
    ------------
    dircoup['ijkl']:
     k        l
      \______/
      /------\
     i        j

    Notes
    -----
     - This directional coupler introduces no delays (for now)
     - This directional coupler is not trainable (for now)
    '''

    num_ports = 4

    def __init__(self,
                 length=12.8e-6,
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
        length : length of the directional coupler
        k0 : bend coupling
        n0 : effective index difference between even and odd mode
        de1_k0 : first derivative of k0 w.r.t. wavelength
        de1_n0 : first derivative of n0 w.r.t. wavelength
        de2_k0 : second derivative of k0 w.r.t. wavelength
        de2_n0 : second derivative of n0 w.r.t. wavelength
        '''

        Component.__init__(name=name)
        self.length = length
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
        S = self.new_variable([[0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]])
        return tau*S

    @property
    def iS(self):
        wl = self.env.wl
        dwl = wl - self.wl0
        dn = self.n0 + self.de1_n0*dwl + 0.5*self.de2_n0*dwl**2
        kappa0 = self.k0 + self.de1_k0*dwl + 0.5*self.de2_k0*dwl**2
        kappa1 = pi*dn/wl
        kappa = -np.sin(kappa0 + kappa1*self.length)
        S = self.new_variable([[0, 0, 1, 0],
                               [0, 0, 0, 1],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0]])
        return kappa*S
