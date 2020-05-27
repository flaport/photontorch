""" Photodetector

The photodetector in this module transforms a raw optical power [W] to a
(possibly noisy) detection current [A]. It takes thermal noise and shot noise
into account.

"""

#############
## Imports ##
#############

# Torch
import torch

# Relative
from .lowpassdetector import LowpassDetector

###############
## Constants ##
###############

q = 1.602176563e-19  # [C] elementary charge
k = 1.3806488e-23  # [m2kg/Ks2] boltzmann constant
T = 300  # [K] room temperature


###################
## PhotoDetector ##
###################


class Photodetector(LowpassDetector):
    """ Realistic Photodector Model.

    The photodetector transforms a raw optical power [W] to a (possibly noisy)
    detection current [A].

    This photodetector takes thermal noise and shot noise into account.

    """

    def __init__(
        self,
        bitrate=40e9,
        samplerate=160e9,
        cutoff_frequency=20e9,
        responsivity=1.0,
        dark_current=1e-10,
        load_resistance=166,
        filter_order=4,
        seed=None,
    ):
        """
        Args:
            bitrate (float): [1/s] data rate of the signal to filter
            samplerate (float): [1/s] sample rate of the signal to filter
            cutoff_frequency (float): [1/s] cutoff frequency of the detector
            responsivity (float): [A/W] responsivity of the photodector
            dark_current (float): [A] dark current adding to the noise
            load_resistance (float): [Ohm] load resistance of the detector
            filter_order (int): filter order of the butter filter
            seed (int): random seed for the detector noise
        """
        super(Photodetector, self).__init__(
            bitrate=bitrate,
            samplerate=samplerate,
            cutoff_frequency=cutoff_frequency,
            responsivity=responsivity,
            filter_order=filter_order,
        )
        self.dark_current = float(dark_current)
        self.load_resistance = float(load_resistance)
        self.seed = None if seed is None else int(seed)
        self._rng = (
            torch.default_generator
            if self.seed is None
            else torch.Generator(device="cpu").manual_seed(seed)
        )

    def forward(self, signal, num_splits=1, split_padding=5):
        """ detect a bitstream by first adding physical noise and then low-pass filtering.

        Args:
            signal (Tensor): signal to detect.
            num_splits (int): number of parallel parts to split the timestream in
            split_padding (int): number of bits padding when splitting the timstream in parts.

        Note:
            Splitting the signal in parts to be processed in parallel can considerably
            speed up the detection process. However, it remains an approximation, as
            in theory each detected signal point depends (with an exponentially
            decreasing factor) on all previous detected signals.

            To partly circumvent any large discontinuities when recomposing the
            detected signal from the detected parts, a `split_padding` should be
            defined, which pads a number of bits from the previous / next
            signal part before and after the current signal part to detect.
        """
        # unit: sqrt(W) -> W (only when complex valued amplitudes are given)
        if signal.shape[0] == 2:  # complex valued signal (photontorch convention)
            signal = signal[0] ** 2 + signal[1] ** 2  # amplitude -> power

        # generate noise
        with torch.no_grad():
            # thermal noise variance
            var_thermal_noise = 4 * k * T / self.load_resistance * self.cutoff_frequency

            # shot noise variance
            var_shot_noise = self.responsivity * signal.detach().clone()
            var_shot_noise += self.dark_current
            var_shot_noise *= 2 * q * self.cutoff_frequency

            # noise standard deviation
            sigma_noise = torch.sqrt(var_thermal_noise + var_shot_noise)

            # noise
            noise = sigma_noise * torch.randn(
                *signal.shape, device=self._rng.device, generator=self._rng
            ).to(signal.device)

        # low pass filter:
        # note that the lowpass detector takes responsivity into account.
        signal = super(Photodetector, self).forward(signal + noise / self.responsivity)

        return signal
