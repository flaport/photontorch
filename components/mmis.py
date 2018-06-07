'''
# MMIs

MMI Splitter and combiners

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


#########
## MMI ##
#########

class Mmi21(Component):
    r''' A 2x1 MMI is a transition from two waveguides to one

    Terms:
              1
         0___/
             \
              2

    Note:
        This MMI introduces no delays (for now)
    '''

    num_ports = 3

    def __init__(self, name=None):
        '''
        MMI2x1 initialization

        Args:
            name (str). name of this specific directional coupler
        '''
        Component.__init__(self, name=name)

    def get_rS(self):
        '''
        Real part of the scattering matrix
        shape: (# wavelengths, # ports, # ports)
        '''
        t = (1.0/2.0)**0.5*self.tensor(np.ones_like(self.env.wls)).view(-1,1,1)
        S = self.tensor([[[0, 1, 1],
                                [1, 0, 0],
                                [1, 0, 0]]])
        return t*S


    def get_iS(self):
        '''
        Imag part of the scattering matrix
        shape: (# wavelengths, # ports, # ports)
        '''
        t = (1.0/2.0**0.5)*self.tensor(np.zeros_like(self.env.wls)).view(-1,1,1)
        S = self.tensor([[[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]]])
        return t*S


#############
## MMI 3x3 ##
#############

class Mmi33(Component):
    r''' A 3x3 MMI mixes 3 inputs with 3 outputs.

    Connections
        mmi33['ijklmn']:

         0___  ___3
         1___\/___4
         2___/\___5

    Note:
        This MMI introduces no delays (for now)
    '''

    num_ports = 6

    def __init__(self, name=None):
        '''
        MMI3x3 initialization

        Args:
            name (str). name of this specific directional coupler
        '''
        Component.__init__(self, name=name)

    def get_rS(self):
        '''
        Real part of the scattering matrix
        shape: (# wavelengths, # ports, # ports)
        '''
        t = (1.0/3.0**0.5)*self.tensor(np.ones_like(self.env.wls)).view(-1,1,1)
        S = self.tensor([[[0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 1, 1],
                                [1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0]]])
        return t*S


    def get_iS(self):
        '''
        Imag part of the scattering matrix
        shape: (# wavelengths, # ports, # ports)
        '''
        t = (1.0/3.0**0.5)*self.tensor(np.zeros_like(self.env.wls)).view(-1,1,1)
        S = self.tensor([[[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]])
        return t*S
