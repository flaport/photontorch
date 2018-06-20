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

    For a waveguide, only its phase change can be trained. If phase==None then
    the phase will be derived from the length of the waveguide and will thus be
    untrainable (phase==None overrules the trainable flag).

    Terms:

        0 ---- 1

    '''

    def __init__(self,
                 length=1e-6,
                 loss=0,
                 neff=2.86,
                 ng=None,
                 phase=None,
                 trainable=True,
                 name=None):
        ''' Waveguide

        Args:
            length (float): Length of the waveguide in meter.
            loss = 0 (float): Loss in the waveguide [dB/m]
            neff = 2.86 (float): Effective index of the waveguide
            ng = None (float): Group index of the waveguide (equals neff if None)
            phase (float): if a phase is given, the phase introduced by the wavelength
                becomes decoupled from the length. This can be useful for training purposes.
                if phase is None, the waveguide is untrainable by definition. This
                overrules the trainable flag.
            trainable (bool): whether the phase of the waveguide is trainable
            name (str): Name of the specific waveguide
        '''
        Connection.__init__(self, name=name)
        # Handle inputs
        self.loss = float(loss)
        self.neff = float(neff)
        self.ng = self.neff if ng is None else float(ng)
        self.length = float(length)
        self.trainable=trainable

        if phase is not None:
            self.phase = self.parameter(data=phase%(2*np.pi), requires_grad=trainable)
        else:
            self.phase = None

    def get_delays(self):
        ''' The delay per node is given by the propagation time in the waveguide '''
        delay = self.ng*self.length/c
        return self.tensor([delay, delay], 'float')

    def get_S(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''
        if self.phase is not None:
            cos_phase = self.phase.cos().double()
            sin_phase = self.phase.sin().double()
        else:
            wls = self.tensor(self.env.wls, dtype='double')
            cos_phase = torch.cos(2*pi*self.neff*self.length/wls)
            sin_phase = torch.sin(2*pi*self.neff*self.length/wls)
        re = 10**(-self.loss*self.length/10)*cos_phase
        ie = 10**(-self.loss*self.length/10)*sin_phase
        # we can safely convert back to single precision now:
        re = re.float()
        ie = ie.float()
        # calculate real part and imag part
        rS = re.view(-1,1,1)*self.tensor([[[0, 1],
                                           [1, 0]]])
        iS = ie.view(-1,1,1)*self.tensor([[[0, 1],
                                           [1, 0]]])
        return torch.stack([rS,iS])