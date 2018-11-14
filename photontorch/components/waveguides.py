"""
# Waveguides

Waveguides are a special kind of Connection with delay.

"""

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .connection import Connection
from ..torch_ext.nn import Parameter, Buffer

###############
## Waveguide ##
###############


class Waveguide(Connection):
    """
    A waveguide is a Component where each of the two nodes
    introduces a delay corresponding to the length of the waveguide.

    For a waveguide, only its phase change can be trained. If phase==None then
    the phase will be derived from the length of the waveguide and will thus be
    untrainable (phase==None overrules the trainable flag).

    Terms:

        0 ---- 1

    """

    def __init__(
        self,
        length=1e-5,
        loss=0,
        neff=2.34,
        wl0=1.55e-6,
        ng=3.40,
        phase=0,
        trainable=False,
        name=None,
    ):
        """
        Args:
            length: float = 1e-5: length of the waveguide in meter.
            loss: float = 0: loss in the waveguide [dB/m]
            neff: float = 2.34: effective index of the waveguide
            ng: float = 3.40: group index of the waveguide
            wl0: float = 1.55e-6: the center wavelength for which neff is defined.
            phase: float = 0: additional phase correction added to the phase introduced
                by the length of the waveguide. Adding this can be useful for training
                purposes.
            trainable: bool = True: makes the phase trainable
            name: str = None: the name of the component (default: lowercase classname)
        """
        Connection.__init__(self, name=name)
        # Handle inputs
        self.loss = float(loss)
        self.neff = float(neff)
        self.wl0 = float(wl0)
        self.ng = float(ng)
        self.length = float(length)
        self.trainable = trainable

        parameter = Parameter if trainable else Buffer
        self.phase = parameter(
            torch.tensor(data=float(phase % (2 * np.pi)), dtype=torch.float64)
        )

    def set_delays(self, delays):
        delays[:] = self.ng * self.length / self.env.c

    def set_S(self, S):
        wls = torch.tensor(self.env.wavelength, dtype=torch.float64, device=self.device)

        # neff depends on the wavelength:
        neff = self.neff - (wls - self.wl0) * (self.ng - self.neff) / self.wl0
        phase = (2 * np.pi * neff * self.length / wls) % (2 * np.pi) + self.phase
        cos_phase = torch.cos(phase).to(torch.get_default_dtype())
        sin_phase = torch.sin(phase).to(torch.get_default_dtype())

        # calculate loss
        loss = 10 ** (-self.loss * self.length / 20)  # 20 because loss works on power
        re = loss * cos_phase
        ie = loss * sin_phase

        # calculate real part and imag part
        S[0, :, 0, 1] = S[0, :, 1, 0] = re
        S[1, :, 0, 1] = S[1, :, 1, 0] = ie
