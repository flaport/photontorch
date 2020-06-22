""" MZIs are 4-port components coupling two waveguides together. """

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .component import Component
from ..nn.nn import Parameter, Buffer


#########################
## Directional Coupler ##
#########################


class Mzi(Component):
    r""" An MZI is a component with 4 ports.

    An MZI has two trainable parameters: the input phase phi and the phase difference
    between the arms theta. .

    Terms::

                    _[2*theta]_
        3  ______  /           \  ___2
                 \/             \/
        0__[phi]_/\_____________/\___1

    Note:
        This MZI implementation assumes the armlength difference is too small to have
        a noticable delay difference between the arms, i.e. only the phase difference matters

    """

    num_ports = 4

    def __init__(
        self,
        phi=0,
        theta=np.pi / 4,
        neff=2.34,
        ng=3.40,
        wl0=1.55e-6,
        length=1e-5,
        loss=0,
        trainable=True,
        name=None,
    ):
        """
        Args:
            phi (float): input phase
            theta (float): phase difference between the arms
            neff (float): effective index of the waveguide
            ng (float): group index of the waveguide
            wl0 (float): the center wavelength for which neff is defined.
            length (float): length of the waveguide in meter.
            loss (float): loss in the waveguide [dB/m]
            trainable (bool): whether phi and theta are trainable
            name (optional, str): name of this specific MZI
        """
        super(Mzi, self).__init__(name=name)

        parameter = Parameter if trainable else Buffer

        self.ng = float(ng)
        self.neff = float(neff)
        self.length = float(length)
        self.loss = float(loss)
        self.wl0 = float(wl0)
        self.phi = parameter(torch.tensor(phi, dtype=torch.float64, device=self.device))
        self.theta = parameter(
            torch.tensor(theta, dtype=torch.float64, device=self.device)
        )

    def set_delays(self, delays):
        delays[:] = self.ng * self.length / self.env.c

    def set_S(self, S):
        wls = torch.tensor(self.env.wl, dtype=torch.float64, device=self.device)

        # neff depends on the wavelength:
        neff = self.neff - (wls - self.wl0) * (self.ng - self.neff) / self.wl0
        phi0 = (2 * np.pi * neff * self.length / wls) % (2 * np.pi)
        phi1 = phi0 + self.phi

        # cos / sin of phases
        cos_phi0 = torch.cos(phi0).to(torch.get_default_dtype())
        sin_phi0 = torch.sin(phi0).to(torch.get_default_dtype())
        cos_phi1 = torch.cos(phi1).to(torch.get_default_dtype())
        sin_phi1 = torch.sin(phi1).to(torch.get_default_dtype())
        cos_theta = torch.cos(self.theta).to(torch.get_default_dtype())
        sin_theta = torch.sin(self.theta).to(torch.get_default_dtype())
        # scattering matrix
        S[0, :, 0, 1] = S[0, :, 1, 0] = cos_phi1 * cos_theta
        S[1, :, 0, 1] = S[1, :, 1, 0] = sin_phi1 * cos_theta
        S[0, :, 0, 2] = S[0, :, 2, 0] = cos_phi1 * sin_theta
        S[1, :, 0, 2] = S[1, :, 2, 0] = sin_phi1 * sin_theta
        S[0, :, 1, 3] = S[0, :, 3, 1] = -cos_phi0 * sin_theta
        S[1, :, 1, 3] = S[1, :, 3, 1] = -sin_phi0 * sin_theta
        S[0, :, 2, 3] = S[0, :, 3, 2] = cos_phi0 * cos_theta
        S[1, :, 2, 3] = S[1, :, 3, 2] = sin_phi0 * cos_theta
        # return scattering matrix

        # add loss
        loss = self.loss * self.length
        return S * 10 ** (-loss / 20)  # 20 bc loss is defined on power.
