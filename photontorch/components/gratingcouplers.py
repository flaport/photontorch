"""
# Grating Couplers

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
    r""" A Grating Coupler is a memory-less component with one input and one output.

    Terms:

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
            R: float = 0.0: reflection of the grating coupler (between 0 and 1)
            R_in: float = 0.0: incoupling reflection for the grating coupler
            Tmax: float = 1.0: maximum transmission at center wavelength
            bandwidth: float = 0.06e-6: 3dB Bandwidth of the grating coupler
            wl0: float = 1.55e-6: Center wavelength of the grating coupler
            name: str = None: the name of the component (default: lowercase classname)
        """
        super(GratingCoupler, self).__init__(name=name)
        self.R = R
        self.R_in = R_in
        self.bandwidth = bandwidth
        self.wl0 = wl0
        self.Tmax = Tmax

    def get_S(self):
        fwhm2sigma = 1.0 / (2 * np.sqrt(2 * np.log(2)))
        sigma = fwhm2sigma * self.bandwidth
        wls = torch.tensor(
            self.env.wavelength[:, None, None],
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        loss = torch.sqrt(
            self.Tmax * torch.exp(-(self.wl0 - wls) ** 2 / (2 * sigma ** 2))
        )
        rS = loss * torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], device=self.device)
        iS = torch.ones_like(loss) * torch.tensor(
            [[[self.R_in, 0], [0, self.R]]], device=self.device
        )
        return torch.stack([rS, iS])
