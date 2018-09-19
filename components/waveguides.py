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
from ..torch_ext.nn import Parameter, Buffer

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
                 length=1e-5,
                 loss=0,
                 neff=2.86,
                 wl0=1.55e-6,
                 ng=None,
                 phase=0,
                 trainable=False,
                 name=None):
        ''' Waveguide

        Args:
            length (float): Length of the waveguide in meter.
            loss = 0 (float): Loss in the waveguide [dB/m]
            neff = 2.86 (float): Effective index of the waveguide
            wl0 = 1.55e-6 (float): the center wavelength for which neff is defined.
            ng = None (float): Group index of the waveguide (equals neff if None)
            phase (float): additional phase correction added to the phase introduced
                by the length of the waveguide. Adding this can be useful for training
                purposes.
            trainable (bool): whether the phase of the waveguide is trainable
            name (str): Name of the specific waveguide
        '''
        Connection.__init__(self, name=name)
        # Handle inputs
        self.loss = float(loss)
        self.neff = float(neff)
        self.wl0 = float(wl0)
        self.ng = self.neff if ng is None else float(ng)
        self.length = float(length)
        self.trainable=trainable

        parameter = Parameter if trainable else Buffer
        self.phase = parameter(torch.tensor(data=float(phase%(2*np.pi)), dtype=torch.float64))

    def get_delays(self):
        ''' The delay per node is given by the propagation time in the waveguide '''
        delay = self.ng*self.length/c
        return delay*torch.ones(2, device=self.device)

    def get_S(self):
        ''' Scattering matrix with shape: (2, # wavelengths, # ports, # ports) '''

        wls = torch.tensor(self.env.wls, dtype=torch.float64, device=self.device)

        # neff depends on the wavelength:
        neff = self.neff - (wls-self.wl0)*(self.ng-self.neff)/self.wl0
        phase = 2*pi*neff*self.length/wls + self.phase
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # calculate loss
        loss = 10**(-self.loss*self.length/20) # 20 because loss works on power
        re = loss*cos_phase
        ie = loss*sin_phase

        # we can safely convert back to single precision now:
        re = re.float()
        ie = ie.float()

        # calculate real part and imag part
        rS = re.view(-1,1,1)*torch.tensor([[[0.0, 1.0],
                                            [1.0, 0.0]]], device=self.device)
        iS = ie.view(-1,1,1)*torch.tensor([[[0.0, 1.0],
                                            [1.0, 0.0]]], device=self.device)

        return torch.stack([rS,iS])
