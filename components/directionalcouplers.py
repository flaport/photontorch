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
    '''

    @property
    def rS(self):
        t = (1-self.R)**0.5
        return self.new_variable([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])*t

    @property
    def iS(self):
        r = self.R**0.5
        return self.new_variable([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])*r