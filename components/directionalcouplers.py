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
from .waveguides import Waveguide
from ..constants import pi
from ..torch_ext.nn import BoundedParameter


#########################
## Directional Coupler ##
#########################

class DirectionalCoupler(Component):
    r''' A directional coupler is a memory-less component with 4 ports.

    A directional coupler has one trainable parameter: the squared coupling coupling.

    Terms:
       3        2
        \______/
        /------\
       0        1

    Note:
        This directional coupler introduces no delays (for now)
    '''

    num_ports = 4

    def __init__(self, coupling=0.5, trainable=True, name=None):
        '''
        Directional Coupler initialization

        Args:
            coupling (float): squared coupling of the directional coupler (between 0 and 1)
            trainable (bool): whether the coupling of the directional coupler is trainable
            name (str). name of this specific directional coupler
        '''
        Component.__init__(self, name=name)

        self.coupling = BoundedParameter(
            data=torch.tensor(coupling, device=self.device),
            bounds=(0,1),
            requires_grad=trainable,
        )

    def get_S(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''
        t = torch.cat([((1-self.coupling)**0.5).view(1,1,1)]*self.env.num_wl, dim=0)
        k = torch.cat([(self.coupling**0.5).view(1,1,1)]*self.env.num_wl, dim=0)
        rS = t*torch.tensor([[[0, 1, 0, 0], # real part
                              [1, 0, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0]]],
                            device=self.device,
                            dtype=torch.get_default_dtype()) # usually float32
        iS = k*torch.tensor([[[0, 0, 1, 0], # imag part
                              [0, 0, 0, 1],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0]]],
                            device=self.device,
                            dtype=torch.get_default_dtype()) # usually float32
        return torch.stack([rS, iS])


#####################################
## Directional Coupler with length ##
#####################################
class DirectionalCouplerWithLength(Component):
    r''' A directional coupler with length is a memory-containing component with 4 ports.

    It is merely a holder of a directional coupler and a waveguide, and combines both
    in a 4-port component.

    Terms:
        3        2
        \______/
        /------\
        0        1

    Note:
        This version of a directional coupler is prefered over a wg-wg-wg-wg-dc network
        becuase it only has 4 ports in stead of 12.
    '''

    num_ports = 4

    def __init__(self, dc=None, wg=None, name=None):
        ''' Directional Coupler

        Args:
            dc : DirectionalCoupler instance without length
            wg : yields the full length and the resulting delays of the directional coupler
        '''
        if dc is None:
            dc = DirectionalCoupler()
        if wg is None:
            wg = Waveguide()
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

    def get_S(self):
        '''Scattering matrix with shape (2, # wavelengths, # ports, # ports)'''
        k = self.dc.coupling**0.5 # coupling
        t = (1-self.dc.coupling)**0.5 # Transmission

        # Helper matrices
        rS_wg_t = self.wg.rS*t
        iS_wg_k = self.wg.iS*k
        iS_wg_t = self.wg.iS*t
        rS_wg_k = self.wg.rS*k

        # Real part
        rS = torch.tensor([[[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]]*self.env.num_wl,
                          device=self.device,
                          dtype=torch.get_default_dtype())
        rS[:,:2, :2] = rS_wg_t # Transmission from i < - > j
        rS[:,2:, 2:] = rS_wg_t # Transmission from k < - > l
        rS[:,::2, ::2] = -iS_wg_k
        rS[:,1::2, 1::2] = -iS_wg_k

        # Imag Part
        iS = torch.tensor([[[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]]*self.env.num_wl,
                          device=self.device,
                          dtype=torch.get_default_dtype())

        iS[:,:2, :2] = iS_wg_t # Transmission from i < - > j
        iS[:,2:, 2:] = iS_wg_t # Transmission from k < - > l
        iS[:,::2, ::2] = rS_wg_k
        iS[:,1::2, 1::2] = rS_wg_k

        # Return
        return torch.stack([rS, iS])


class RealisticDirectionalCoupler(Component):
    r''' A directional coupler is a memory-less component with 4 ports.

    The realistic directional coupler is based on the CapheModel of Umar Khan

    Terms:
        3       2
        \______/
        /------\
        0       1

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

    def get_S(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''
        wl = self.env.wls
        dwl = wl - self.wl0
        dn = self.n0 + self.de1_n0*dwl + 0.5*self.de2_n0*dwl**2
        kappa0 = self.k0 + self.de1_k0*dwl + 0.5*self.de2_k0*dwl**2
        kappa1 = pi*dn/wl
        tau = torch.tensor(np.cos(kappa0 + kappa1*self.length), device=self.device).view(-1,1,1)
        kappa = torch.tensor(-np.sin(kappa0 + kappa1*self.length), device=self.device).view(-1,1,1)
        rS = tau*torch.tensor([[[0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]]],
                              device=self.device,
                              dtype=torch.get_default_dtype())
        iS = kappa*torch.tensor([[[0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0],
                                  [0, 1, 0, 0]]],
                                device=self.device,
                                dype=torch.get_default_dtype())
        return torch.stack([rS, iS])
