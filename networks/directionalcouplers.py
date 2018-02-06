''' Ring Networks '''

#############
## Imports ##
#############

## Other
import torch

## Relative
from .network import Network
from ..components.component import Component

#####################
## All Pass Filter ##
#####################

class DirectionalCouplerWithLength(Component):
    '''
    A directional coupler with length is a memory-containing component with 4 ports.

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
     * This version of a directional coupler is prefered over a wg-wg-wg-wg-dc network
       becuase it only has 4 ports in stead of 12.
     * This Component is part of the networks module, because it acts as a network.
    '''

    num_ports = 4

    def __init__(self, dircoup, wg, name=None):
        '''
        Directional Coupler initialization

        Parameters
        ----------
        dircoup : DirectionalCoupler instance without length
        wg : gives the full length of the directional coupler
        '''
        Component.__init__(self, name=None)
        self.wg = wg
        self.dircoup = dircoup

    def initialize(self, env):
        ''' Initialize Component '''
        self.wg.initialize(env)
        self.dircoup.initialize(env)

    @property
    def delays(self):
        return torch.cat((self.wg.delays, self.wg.delays))

    @property
    def rS(self):
        ''' real part of the scattering matrix '''
        k = self.dircoup.R**0.5 # coupling
        t = (1-self.dircoup.R)**0.5 # Transmission
        rS_wg_t = self.wg.rS*t
        iS_wg_k = self.wg.iS*k
        rS = self.new_variable([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        rS[:2,:2] = rS_wg_t # Transmission from i < - > j
        rS[2:,2:] = rS_wg_t # Transmission from k < - > l
        rS[0::3,0::3] = -iS_wg_k
        rS[1:3,1:3] = -iS_wg_k
        return rS

    @property
    def iS(self):
        ''' imag part of the scattering matrix '''
        k = self.dircoup.R**0.5 # coupling
        t = (1-self.dircoup.R)**0.5 # Transmission
        iS_wg_t = self.wg.iS*t
        rS_wg_k = self.wg.rS*k
        iS = self.new_variable([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        iS[:2,:2] = iS_wg_t # Transmission from i < - > j
        iS[2:,2:] = iS_wg_t # Transmission from k < - > l
        iS[0::3,0::3] = rS_wg_k
        iS[1:3,1:3] = rS_wg_k
        return iS
