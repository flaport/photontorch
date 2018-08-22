
'''
# SOAs

'''

#############
## Imports ##
#############

## Torch
import torch

## Relative
from .component import Component
from ..torch_ext.nn import Parameter, Buffer


################
## Simple SOA ##
################

class LinearSoa(Component):
    ''' A Linear SOA is a memory-less component with one input and one output.

    It amplifies a signal instantaneously and linearly with the specified amplification factor

    A simple SOA has one trainable parameter: the amplification.

    Terms:

        0 ---- 1

    '''

    num_ports = 2

    def __init__(self, amplification=0.5, trainable=True, name=None):
        ''' SOA

        Args:
            amplification (float). Reflectivity of the mirror (between 0 and 1)
            trainable (bool): Whether the amplification is trainable
            name (str): name of this specific mirror
        '''
        Component.__init__(self, name=name)

        parameter = Parameter if trainable else Buffer
        self.amplification = parameter(torch.tensor(float(amplification), device=self.device))

    def get_S(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''
        a = torch.cat([(1.0*self.amplification).view(1,1,1)]*self.env.num_wl, dim=0)
        S = a*torch.tensor([[[0.0, 1.0],
                             [1.0, 0.0]]], device=self.device)
        return torch.stack([S, torch.zeros_like(S)])
