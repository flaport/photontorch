""" Grating Couplers

Grating couplers are a special kind of 2-port component that simulate the behavior of
coupling light from a fiber onto the chip.

"""

#############
## Imports ##
#############

## Torch
import torch
import numpy as np

## Relative
from .component import Component


#####################
## Grating Coupler ##
#####################


class GratingCoupler(Component):
    r""" Grating couplers are partly reflecting and partly transmitting connections.

    Terms::

          0
           \
            \
        _|-|_|-|_|-|___ 1

    Note:
        A Grating Coupler is not trainable (for now).
    """

    num_ports = 2

    def __init__(
        self, R=0.0, R_in=0.0, Tmax=1.0, bandwidth=0.06e-6, wl0=1.55e-6, name=None
    ):
        """
        Args:
            R (float): reflection of the grating coupler (between 0 and 1)
            R_in (float): incoupling reflection for the grating coupler
            Tmax (float): maximum transmission at center wavelength
            bandwidth (float): 3dB Bandwidth of the grating coupler
            wl0 (float): Center wavelength of the grating coupler
            name (optional, str): the name of the component (default: lowercase classname)
        """
        super(GratingCoupler, self).__init__(name=name)
        self.R = R
        self.R_in = R_in
        self.bandwidth = bandwidth
        self.wl0 = wl0
        self.Tmax = Tmax

    def set_S(self, S):
        fwhm2sigma = 1.0 / (2 * np.sqrt(2 * np.log(2)))
        sigma = fwhm2sigma * self.bandwidth
        wls = torch.tensor(
            self.env.wl, device=self.device, dtype=torch.get_default_dtype()
        )
        loss = torch.sqrt(
            self.Tmax * torch.exp(-((self.wl0 - wls) ** 2) / (2 * sigma ** 2))
        )

        S[0, :, 0, 1] = S[0, :, 1, 0] = loss
        S[1, :, 0, 0] = self.R_in
        S[1, :, 1, 1] = self.R
