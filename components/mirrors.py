''' Mirrors Module '''

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
from ..utils.functions import inv_sigmoid


############
## Mirror ##
############

class Mirror(Component):
    '''
    A mirror is a memory-less component with one input and one output.

    A mirror has one trainable parameter: the reflectivity R.

    Connections
    -----------
    mirror['ij']:
        |
    i --|-- j
        |
    '''

    num_ports = 2

    def __init__(self, R=0.5, R_bounds=(0,1), name=None):
        '''
        Mirror initialization

        Parameters
        ----------
        R : float. Reflectivity of the mirror (between 0 and 1)
        R_bounds : tuple. Bounds in which to optimize R.
                   If None, R will not be optimized.
        name : str. name of this specific mirror
        '''
        Component.__init__(self, name=name)
        if R_bounds is None:
            self.R_min = self.R_max = R
        else:
            self.R_min = R_bounds[0]
            self.R_max = R_bounds[1]

        if self.R_min != self.R_max:
            _R = (R-self.R_min)/(self.R_max-self.R_min)
            self.W_R = self.new_parameter([inv_sigmoid(_R)], dtype='float')

    @property
    def R(self):
        ''' Reflectivity of the Mirror '''
        if self.R_min == self.R_max:
            return self.R_min
        return (self.R_max-self.R_min)*torch.sigmoid(self.W_R) + self.R_min
    @R.setter
    def R(self, value):
        ''' Set Reflectivity of the mirror manually (not recommended) '''
        _R = (value-self.R_min)/(self.R_max-self.R_min)
        self.W_R = self.new_parameter([inv_sigmoid(_R)], dtype='float')

    @property
    def rS(self):
        r = self.R**0.5
        return self.new_variable([[1,0],[0,1]])*r

    @property
    def iS(self):
        t = (1-self.R)**0.5
        return self.new_variable([[0,1],[1,0]])*t


####################
## Slanted Mirror ##
####################

class SlantedMirror(Mirror):
    '''
    A slanted mirror is a memory-less component with 4 ports.

    A slanted mirror has one trainable parameter: the reflectivity R.


    Connections
    -----------
    slanted_mirror['ijkl']:
           j
           |/
       i --/-- k
          /|
           l
    '''

    num_ports = 4

    @property
    def rS(self):
        r = self.R**0.5
        return self.new_variable([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])*r

    @property
    def iS(self):
        t = (1-self.R)**0.5
        return self.new_variable([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])*t
