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
from ..torch_ext.nn import Buffer, Parameter


#########
## MMI ##
#########

class Mmi(Component):
    r''' A MMI performs a weighted transition from m waveguides to n waveguides

    Terms:
              m+1
         0___/ m+2
         1____/
          :  :
         m____
              \
               n

    Note:
        This MMI introduces no delays
    '''

    num_ports = None # not yet defined

    def __init__(self, weights=None, trainable=False, name=None):
        '''
        MMI2x1 initialization

        Args:
            weights (array). interconnection weights between input and output if MMI
            trainable (bool). if the weights are trainable or not
            name (str). name of this specific directional coupler
        '''

        # validate weights
        if weights is None:
            raise ValueError("No weight matrix specified ")

        if len(weights.shape) != 2 and len(weights.shape) != 3:
            raise ValueError('weights should be at least 2D and at most 3D [first index real|imag].')

        if not torch.is_tensor(weights):
            weights = torch.tensor(np.stack([np.real(weights), np.imag(weights)], 0),
                                   dtype=torch.get_default_dtype())

        if len(weights.shape) == 2:
            weights = torch.stack([weights, torch.zeros_like(weights)], 0)

        # weights should now be 3D: shape = [2=(real|imag), m, n].
        _, m, n = weights.shape
        self.num_ports = m + n

        # after num_ports is defined, we can initialize the component
        Component.__init__(self, name=name)


        parameter = Parameter if trainable else Buffer
        self.weights = parameter(weights)

    def get_S(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''
        weights = self.weights[:,None,:,:] # same weight for each wavelength
        _, _, m, n = weights.shape
        S = torch.zeros((2,self.env.num_wl, m+n, m+n))
        S[:,:, :m, m:] = weights
        S[:,:, m:, :m] = torch.transpose(weights, -1, -2)
        return S

