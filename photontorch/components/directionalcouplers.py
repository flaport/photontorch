""" Directional Couplers are 4-port components coupling two waveguides together.  """

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .component import Component
from .waveguides import Waveguide
from ..torch_ext.nn import BoundedParameter, Parameter, Buffer
from ..environment import current_environment


#########################
## Directional Coupler ##
#########################


class DirectionalCoupler(Component):
    r""" A directional coupler is a memory-less component with 4 ports.

    A directional coupler has one trainable parameter: the squared coupling coupling.

    Terms::

       3        2
        \______/
        /------\
       0        1

    """

    num_ports = 4

    def __init__(self, coupling=0.5, trainable=True, name=None):
        """
        Args:
            coupling (float): power coupling of the directional coupler (between 0 and 1)
            trainable (bool): makes the coupling trainable
            name (str): the name of the component (default: lowercase classname)
        """
        super(DirectionalCoupler, self).__init__(name=name)

        if trainable:
            self.coupling = BoundedParameter(
                data=torch.tensor(coupling, device=self.device),
                bounds=(0, 1),
                requires_grad=True,
            )
        else:
            self.coupling = Buffer(
                data=torch.tensor(coupling, device=self.device), requires_grad=False
            )

    def set_S(self, S):
        t = (1 - self.coupling) ** 0.5
        k = self.coupling ** 0.5

        # real part scattering matrix (transmission):
        S[0, :, 0, 1] = S[0, :, 1, 0] = t
        S[0, :, 2, 3] = S[0, :, 3, 2] = t

        # imag part scattering matrix (coupling):
        S[1, :, 0, 2] = S[1, :, 2, 0] = k
        S[1, :, 1, 3] = S[1, :, 3, 1] = k


#####################################
## Directional Coupler with length ##
#####################################


class DirectionalCouplerWithLength(Component):
    r""" A directional coupler with length is a memory-containing component with 4 ports.

    Terms::

        3        2
         \______/
         /------\
        0        1

    Note:
        This version of a directional coupler is prefered over a wg-wg-wg-wg-dc
        network becuase it only has 4 ports in stead of 12.

    """

    num_ports = 4

    def __init__(
        self,
        length=1e-5,
        coupling=0.5,
        loss=0,
        neff=2.34,
        wl0=1.55e-6,
        ng=3.40,
        phase=0,
        trainable_phase=True,
        trainable_coupling=True,
        name=None,
    ):
        """
        Args:
            length (float): length of the waveguide in meter.
            coupling (float): power coupling of the directional coupler (between 0 and 1)
            loss (float): loss in the waveguide [dB/m]
            neff (float): effective index of the waveguide
            ng (float): group index of the waveguide
            wl0 (float): the center wavelength for which neff is defined.
            phase (float): additional phase correction added to the phase introduced
                by the length of the waveguide. Adding this can be useful for training
                purposes.
            trainable_phase (bool): makes the phase parameter trainable
            trainable_coupling (bool): makes the coupling parameter trainable
            name (str): the name of the component (default: lowercase classname)
        """
        super(DirectionalCouplerWithLength, self).__init__(name=name)
        # Handle inputs
        self.loss = float(loss)
        self.neff = float(neff)
        self.wl0 = float(wl0)
        self.ng = float(ng)
        self.length = float(length)
        self.trainable_phase = bool(trainable_phase)
        self.trainable_coupling = bool(trainable_coupling)

        parameter = Parameter if self.trainable_phase else Buffer
        self.phase = parameter(
            torch.tensor(data=float(phase % (2 * np.pi)), dtype=torch.float64)
        )

        if self.trainable_coupling:
            self.coupling = BoundedParameter(
                data=torch.tensor(coupling, device=self.device),
                bounds=(0, 1),
                requires_grad=True,
            )
        else:
            self.coupling = Buffer(
                data=torch.tensor(coupling, device=self.device), requires_grad=False
            )

    def set_delays(self, delays):
        delays[:] = self.ng * self.length / self.env.c

    def set_S(self, S):
        kappa = self.coupling ** 0.5
        tau = (1 - self.coupling) ** 0.5
        cos_phase = torch.cos(self.phase)
        sin_phase = torch.sin(self.phase)
        loss = 10 ** (
            -self.loss * self.length / 20
        )  # factor 20 bc amplitudes, not intensities.

        S[0, :, 0, 1] = S[0, :, 1, 0] = tau * loss * cos_phase
        S[1, :, 0, 1] = S[1, :, 1, 0] = tau * loss * sin_phase

        S[0, :, 0, 2] = S[0, :, 2, 0] = -kappa * loss * sin_phase
        S[1, :, 0, 2] = S[1, :, 2, 0] = kappa * loss * cos_phase

        S[0, :, 1, 3] = S[0, :, 3, 1] = -kappa * loss * sin_phase
        S[1, :, 1, 3] = S[1, :, 3, 1] = kappa * loss * cos_phase

        S[0, :, 2, 3] = S[0, :, 3, 2] = tau * loss * cos_phase
        S[1, :, 2, 3] = S[1, :, 3, 2] = tau * loss * sin_phase


class RealisticDirectionalCoupler(Component):
    r""" A directional coupler is a memory-less component with 4 ports.

    Terms::

        3        2
         \______/
         /------\
        0        1

    Notes:
     - This directional coupler introduces no delays (for now)
     - This directional coupler is not trainable (for now)

    """

    num_ports = 4

    def __init__(
        self,
        length=12.8e-6,
        k0=0.2332,
        n0=0.0208,
        de1_k0=1.2435,
        de1_n0=0.1169,
        de2_k0=5.3022,
        de2_n0=0.4821,
        wl0=1.55e-6,
        name=None,
    ):
        """
        Args:
            length (float): the length
            k0 (float): the bend coupling
            n0 (float): effective index difference between even and odd mode
            de1_k0 (float): first derivative of k0 w.r.t. wavelength
            de1_n0 (float): first derivative of n0 w.r.t. wavelength
            de2_k0 (float): second derivative of k0 w.r.t. wavelength
            de2_n0 (float): second derivative of n0 w.r.t. wavelength
            wl0 (float): the center wavelength for which the parameters are defined
            name (str): the name of the component (default: lowercase classname)

        Note:
            The default parameters are based on a directional coupler with
             - bend radius : 5um
             - adiabatic angle : 10 degrees
             - gap distance : 250 nm

        """

        super(RealisticDirectionalCoupler, self).__init__(name=name)
        self.length = length
        self.k0 = k0
        self.de1_k0 = de1_k0
        self.de2_k0 = de2_k0
        self.n0 = n0
        self.de1_n0 = de1_n0
        self.de2_n0 = de2_n0
        self.wl0 = wl0

    def set_S(self, S):
        wl = torch.tensor(self.env.wl, dtype=torch.float64, device=self.device)
        dwl = wl - self.wl0
        dn = self.n0 + self.de1_n0 * dwl + 0.5 * self.de2_n0 * dwl ** 2
        kappa0 = self.k0 + self.de1_k0 * dwl + 0.5 * self.de2_k0 * dwl ** 2
        kappa1 = np.pi * dn / wl

        dtype = torch.get_default_dtype()
        tau = torch.cos(kappa0 + kappa1 * self.length).to(dtype)
        kappa = -torch.sin(kappa0 + kappa1 * self.length).to(dtype)

        # real part scattering matrix (transmission):
        S[0, :, 0, 1] = S[0, :, 1, 0] = tau
        S[0, :, 2, 3] = S[0, :, 3, 2] = tau

        # imag part scattering matrix (coupling):
        S[1, :, 0, 2] = S[1, :, 2, 0] = kappa
        S[1, :, 1, 3] = S[1, :, 3, 1] = kappa
