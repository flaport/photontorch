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
            self.W_R = self.new_variable([-np.log(1/R-1)], 'float')
            self.R_min = 0
            self.R_max = 1
            self.trainable = False
        else:
            self.W_R = self.new_parameter([-np.log(1/R-1)],'float')
            self.R_min = self.new_variable([R_bounds[0]], 'float')
            self.R_max = self.new_variable([R_bounds[1]], 'float')
            self.trainable = True

    @property
    def R(self):
        ''' Reflectivity of the Mirror '''
        return (self.R_max-self.R_min)*torch.sigmoid(self.W_R) + self.R_min
    @R.setter
    def R(self, value):
        ''' Reflectivity of the mirror '''
        R_bounds = (self.R_min, self.R_max) if self.trainable else None
        self.__init__(self.R.data.numpy()[0], R_bounds, self.name)

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

    @property
    def rS(self):
        r = self.R**0.5
        return self.new_variable([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])*r

    @property
    def iS(self):
        t = (1-self.R)**0.5
        return self.new_variable([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])*t
