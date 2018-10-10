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
from ..constants import c
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
        Mmi initialization

        Args:
            weights (array). interconnection weights between input and output if MMI
            trainable (bool). if the weights are trainable or not
            name (str). name of this specific directional coupler
        '''

        # validate weights
        if weights is None:
            weights = np.sqrt(0.5*np.ones((1,2)))

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


class PhaseArray(Component):
    r''' A Phase Array adds a phase to n inputs

    Terms:
              n+1
         0___/ n+2
         1____/
          :  :
         n____
              \
               2*n

    '''

    num_ports = None

    def __init__(self, phases=None, length=1e-5, ng=2.86, trainable=True, name=None):
        if phases is None:
            phases = np.zeros(2)

        if len(phases.shape) != 1:
            raise ValueError('phases should be a 1D array or tensor')

        self.num_ports = 2*phases.shape[0]

        Component.__init__(self, name=name)

        self.ng = float(ng)
        self.length = float(length)

        parameter = Parameter if trainable else Buffer
        phases = torch.tensor(phases, dtype=torch.float64, device=self.device)
        self.phases = parameter(phases)

    def get_delays(self):
        delay = self.ng*self.length/c
        return delay*torch.ones(self.num_ports, device=self.device)

    def get_S(self):
        cos_phase = torch.diag(torch.cos(self.phases).to(torch.get_default_dtype()))
        sin_phase = torch.diag(torch.sin(self.phases).to(torch.get_default_dtype()))
        phase = torch.stack([cos_phase, sin_phase], 0)[:, None, :, :]
        phase = torch.ones((1, self.env.num_wl, 1, 1), device=self.device)*phase
        S1 = torch.cat([torch.zeros_like(phase), phase], -1)
        S2 = torch.cat([phase, torch.zeros_like(phase)], -1)
        S = torch.cat([S1, S2], -2)
        return S
