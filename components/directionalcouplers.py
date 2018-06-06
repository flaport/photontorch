'''
# Directional Couplers

Directional Couplers are 4-port components coupling two waveguides together.

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
from ..constants import pi


#########################
## Directional Coupler ##
#########################

class DirectionalCoupler(Component):
    r''' A directional coupler is a memory-less component with 4 ports.

    A directional coupler has one trainable parameter: the squared coupling kappa2.

    Connections
        dc['ijkl']:
        l        k
        \______/
        /------\
        i        j

    Note:
        This directional coupler introduces no delays (for now)
    '''

    num_ports = 4

    def __init__(self, kappa2=0.5, kappa2_bounds=(0, 1), name=None):
        '''
        Directional Coupler initialization

        Args:
            kappa2 (float): squared coupling of the directional coupler (between 0 and 1)
            kappa2_bounds (tuple): Bounds in which to optimize the squared coupling.
                If None, kappa2 will not be optimized.
            name (str). name of this specific directional coupler
        '''
        Component.__init__(self, name=name)

        self.kappa2 = self.bounded_parameter(
            data=kappa2,
            bounds=kappa2_bounds,
            requires_grad=(kappa2_bounds is not None) and (kappa2_bounds[0]!=kappa2_bounds[1]),
        )

    def get_rS(self):
        ''' Real part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        t = torch.cat([((1-self.kappa2)**0.5).view(1,1,1)]*self.env.num_wl, dim=0)
        S = self.tensor([[[0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]]])
        return t*S

    def get_iS(self):
        '''
        Imag part of the scattering matrix
        shape: (# num wavelengths, # num ports, # num ports)
        '''
        k = torch.cat([(self.kappa2**0.5).view(1,1,1)]*self.env.num_wl, dim=0)
        S = self.tensor([[[0, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 0, 0],
                                [0, 1, 0, 0]]])
        return k*S


#####################################
## Directional Coupler with length ##
#####################################
class DirectionalCouplerWithLength(Component):
    r''' A directional coupler with length is a memory-containing component with 4 ports.

    It is merely a holder of a directional coupler and a waveguide, and combines both
    in a 4-port component.

    Connections:
        dc['ijkl']:
        l        k
        \______/
        /------\
        i        j

    Note:
        This version of a directional coupler is prefered over a wg-wg-wg-wg-dc network
        becuase it only has 4 ports in stead of 12.
    '''

    num_ports = 4

    def __init__(self, dc, wg, name=None):
        ''' Directional Coupler

        Args:
            dc : DirectionalCoupler instance without length
            wg : yields the full length and the resulting delays of the directional coupler
        '''
        Component.__init__(self, name=name)
        self.wg = wg
        self.dc = dc

    def initialize(self, env):
        self.wg.initialize(env)
        self.dc.initialize(env)
        Component.initialize(self, env)

    def get_delays(self):
        ''' Delays of the directional coupler '''
        return torch.cat((self.wg.delays, self.wg.delays))

    def get_rS(self):
        ''' real part of the scattering matrix '''
        k = self.dc.kappa2**0.5 # coupling
        t = (1-self.dc.kappa2)**0.5 # Transmission
        rS_wg_t = self.wg.rS*t
        iS_wg_k = self.wg.iS*k
        rS = self.tensor([[[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]]*self.env.num_wl)
        rS[:,:2, :2] = rS_wg_t # Transmission from i < - > j
        rS[:,2:, 2:] = rS_wg_t # Transmission from k < - > l
        rS[:,::2, ::2] = -iS_wg_k
        rS[:,1::2, 1::2] = -iS_wg_k
        return rS

    def get_iS(self):
        ''' imag part of the scattering matrix '''
        k = self.dc.kappa2**0.5 # coupling
        t = (1-self.dc.kappa2)**0.5 # Transmission
        iS_wg_t = self.wg.iS*t
        rS_wg_k = self.wg.rS*k
        iS = self.tensor([[[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]]*self.env.num_wl)
        iS[:,:2, :2] = iS_wg_t # Transmission from i < - > j
        iS[:,2:, 2:] = iS_wg_t # Transmission from k < - > l
        iS[:,::2, ::2] = rS_wg_k
        iS[:,1::2, 1::2] = rS_wg_k
        return iS


class RealisticDirectionalCoupler(Component):
    r''' A directional coupler is a memory-less component with 4 ports.

    The realistic directional coupler is based on the CapheModel of Umar Khan

    Connections:
        dc['ijkl']:
        l        k
        \______/
        /------\
        i        j

    Notes:
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

        Args:
            length : length of the directional coupler
            k0 : bend coupling
            n0 : effective index difference between even and odd mode
            de1_k0 : first derivative of k0 w.r.t. wavelength
            de1_n0 : first derivative of n0 w.r.t. wavelength
            de2_k0 : second derivative of k0 w.r.t. wavelength
            de2_n0 : second derivative of n0 w.r.t. wavelength
        '''

        Component.__init__(self, name=name)
        self.length = length
        self.k0 = k0
        self.de1_k0 = de1_k0
        self.de2_k0 = de2_k0
        self.n0 = n0
        self.de1_n0 = de1_n0
        self.de2_n0 = de2_n0
        self.wl0 = 1.55e-6

    def get_rS(self):
        ''' Real part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        wl = self.env.wls
        dwl = wl - self.wl0
        dn = self.n0 + self.de1_n0*dwl + 0.5*self.de2_n0*dwl**2
        kappa0 = self.k0 + self.de1_k0*dwl + 0.5*self.de2_k0*dwl**2
        kappa1 = pi*dn/wl
        tau = self.tensor(np.cos(kappa0 + kappa1*self.length)).view(-1,1,1)
        S = self.tensor([[[0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]]])
        return tau*S

    def get_iS(self):
        ''' Imag part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        wl = self.env.wls
        dwl = wl - self.wl0
        dn = self.n0 + self.de1_n0*dwl + 0.5*self.de2_n0*dwl**2
        kappa0 = self.k0 + self.de1_k0*dwl + 0.5*self.de2_k0*dwl**2
        kappa1 = pi*dn/wl
        kappa = self.tensor(-np.sin(kappa0 + kappa1*self.length)).view(-1,1,1)
        S = self.tensor([[[0, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 0, 0],
                                [0, 1, 0, 0]]])
        return kappa*S
