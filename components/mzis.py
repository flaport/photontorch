"""
# MZIs

MZIs are 4-port components coupling two waveguides together.

"""

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .component import Component
from ..torch_ext.nn import Parameter, Buffer
from ..constants import c


#########################
## Directional Coupler ##
#########################


class Mzi(Component):
    r""" A MZI is a component with 4 ports.

    An MZI has two trainable parameter: the input phase phi and the phase difference
    between the arms theta. .

    Terms:          _[2*theta]_
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
        length=1e-5,
        phi=0,
        theta=np.pi / 4,
        loss=0,
        neff=2.86,
        ng=None,
        wl0 = 1.55e-6,
        trainable=True,
        name=None,
    ):
        """
        Directional Coupler initialization

        Args:
            length (float): length of the MZI (approximately equal for both arms!)
            phi (float): input phase
            theta (float): phase difference between the arms
            loss (float): loss of the MZI [dB]
            trainable (bool): whether phi and theta are trainable
            name (str). name of this specific MZI
        """
        Component.__init__(self, name=name)

        parameter = Parameter if trainable else Buffer

        self.ng = float(neff) if ng is None else float(ng)
        self.neff = float(neff)
        self.length = float(length)
        self.loss = float(loss)
        self.wl0 = float(wl0)
        self.phi = parameter(torch.tensor(phi, dtype=torch.float64, device=self.device))
        self.theta = parameter(
            torch.tensor(theta, dtype=torch.float64, device=self.device)
        )

    def get_delays(self):
        delay = self.ng * self.length / c
        return delay * torch.ones(self.num_ports, device=self.device)

    def get_S(self):
        """ Scattering matrix with shape: (2, # wavelengths, # ports, # ports) """
        wls = torch.tensor(self.env.wls, dtype=torch.float64, device=self.device)

        # neff depends on the wavelength:
        neff = self.neff - (wls-self.wl0)*(self.ng-self.neff)/self.wl0
        phi0 = 2*np.pi*neff*self.length/wls
        phi1 = phi0 + self.phi

        # cos / sin of phases
        cos_phi0 = torch.cos(phi0).to(torch.get_default_dtype())
        sin_phi0 = torch.sin(phi0).to(torch.get_default_dtype())
        cos_phi1 = torch.cos(phi1).to(torch.get_default_dtype())
        sin_phi1 = torch.sin(phi1).to(torch.get_default_dtype())
        cos_theta = torch.cos(self.theta).to(torch.get_default_dtype())
        sin_theta = torch.sin(self.theta).to(torch.get_default_dtype())
        # scattering matrix
        S = torch.zeros((2, self.env.num_wl, 4, 4), device=self.device)
        S[0, :, 0, 1] = S[0, :, 1, 0] = cos_phi1 * cos_theta
        S[1, :, 0, 1] = S[1, :, 1, 0] = sin_phi1 * cos_theta
        S[0, :, 0, 2] = S[0, :, 2, 0] = cos_phi1 * sin_theta
        S[1, :, 0, 2] = S[1, :, 2, 0] = sin_phi1 * sin_theta
        S[0, :, 1, 3] = S[0, :, 3, 1] = -cos_phi0*sin_theta
        S[1, :, 1, 3] = S[1, :, 3, 1] = -sin_phi0*sin_theta
        S[0, :, 2, 3] = S[0, :, 3, 2] = cos_phi0*cos_theta
        S[1, :, 2, 3] = S[1, :, 3, 2] = sin_phi0*cos_theta
        # return scattering matrix

        # add loss
        loss = self.loss * self.length
        return S * 10 ** (-loss / 20)  # 20 bc loss is defined on power.
