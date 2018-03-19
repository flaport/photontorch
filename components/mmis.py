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

    Connections
        mmi21['ijk']:
              j
         i___/
             \
              k

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

    @property
    def rS(self):
        '''
        Real part of the scattering matrix
        shape: (# wavelengths, # ports, # ports)
        '''
        t = (1.0/2.0)**0.5*self.new_variable(np.ones_like(self.env.wls)).view(-1,1,1)
        S = self.new_variable([[[0, 1, 1],
                                [1, 0, 0],
                                [1, 0, 0]]])
        return t*S


    @property
    def iS(self):
        '''
        Imag part of the scattering matrix
        shape: (# wavelengths, # ports, # ports)
        '''
        t = (1.0/2.0**0.5)*self.new_variable(np.zeros_like(self.env.wls)).view(-1,1,1)
        S = self.new_variable([[[0, 0, 0],
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

         i___  ___l
         j___\/___m
         k___/\___n

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

    @property
    def rS(self):
        '''
        Real part of the scattering matrix
        shape: (# wavelengths, # ports, # ports)
        '''
        t = (1.0/3.0**0.5)*self.new_variable(np.ones_like(self.env.wls)).view(-1,1,1)
        S = self.new_variable([[[0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 1, 1],
                                [1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0]]])
        return t*S


    @property
    def iS(self):
        '''
        Imag part of the scattering matrix
        shape: (# wavelengths, # ports, # ports)
        '''
        t = (1.0/3.0**0.5)*self.new_variable(np.zeros_like(self.env.wls)).view(-1,1,1)
        S = self.new_variable([[[0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]])
        return t*S
