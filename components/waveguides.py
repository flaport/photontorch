'''
# Waveguides

Waveguides are a special kind of Connection with delay.

'''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .connection import Connection
from ..constants import pi, c

###############
## Waveguide ##
###############

class Waveguide(Connection):
    '''
    A waveguide is a Component where each of the two nodes
    introduces a delay corresponding to the length of the waveguide.

    For a waveguide, only its phase change can be trained

    Terms:

        0 ---- 1

    '''

    def __init__(self,
                 length=1e-6,
                 neff=2.86,
                 loss=0,
                 phase=None,
                 trainable=True,
                 name=None):
        ''' Waveguide

        Args:
            length (float): Length of the waveguide in meter.
            neff = 4.0 (float). Effective index of the waveguide
            loss = 0 (float): Loss in the waveguide [dB/m]
            phase (float): if a phase is given, the phase introduced by the wavelength
                becomes decoupled from the length. This can be useful for training purposes.
            trainable (bool): whether the phase of the waveguide is trainable
            name (str): Name of the specific waveguide
        '''
        Connection.__init__(self, name=name)
        # Handle inputs
        self.neff = float(neff)
        self.loss = loss
        # as the phase is very sensitive on the length, we need double precision to
        # store the length of the waveguide:
        self.length = self.buffer( # Waveguide length is not trainable (for now)
            data=length,
            dtype='double',
            requires_grad=False,
        )

        if phase is not None:
            self.phase = self.parameter(data=phase%(2*np.pi), requires_grad=trainable)
        else:
            self.phase = None

    def get_delays(self):
        ''' The delay per node is given by the propagation time in the waveguide '''
        delay = self.neff*self.length/c
        return torch.cat([delay, delay]).float()

    def get_rS(self):
        ''' Real part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        if self.phase is not None:
            cos_phase = self.phase.cos().double()
        else:
            wls = self.tensor(self.env.wls, dtype='double')
            cos_phase = torch.cos(2*pi*self.neff*self.length/wls)
        re = 10**(-self.loss*self.length/10)*cos_phase
        # we can safely convert back to single precision now:
        re = re.float()
        S = self.tensor([[[0, 1],
                                [1, 0]]])
        return re.view(-1,1,1)*S

    def get_iS(self):
        ''' Imag part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        if self.phase is not None:
            sin_phase = self.phase.sin().double()
        else:
            wls = self.tensor(self.env.wls, dtype='double')
            sin_phase = torch.sin(2*pi*self.neff*self.length/wls)
        ie = 10**(-self.loss*self.length/10)*sin_phase
        # we can safely convert back to single precision now:
        ie = ie.float()
        S = self.tensor([[[0, 1],
                                [1, 0]]])
        return ie.view(-1,1,1)*S
