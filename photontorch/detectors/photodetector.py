""" Photodetector

The photodetector transforms the raw output power (for example the result of a
photontorch simulation) according to a realistic filtering model.

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

    The photodetector transforms the raw output power (for example the result of a
    photontorch simulation) according to a realistic filtering model.

    """

    def __init__(
        self,
        bitrate=40e9,
        samplerate=160e9,
        cutoff_frequency=20e9,
        responsivity=1.0,
        dark_current=1e-10,
        load_resistance=1e6,
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
            filter_order=filter_order,
        )
        self.responsivity = float(responsivity)
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
        if signal.shape[0] == 2:  # complex valued signal (photontorch convention)
            signal = signal[0] ** 2 + signal[1] ** 2  # amplitude -> power

        # Add random noise
        var_shot_noise = (
            2
            * q
            * self.responsivity
            * signal.mean(-1, keepdims=True)
            * self.cutoff_frequency
        )
        var_thermal_noise = (
            4 * k * T * self.cutoff_frequency / self.load_resistance
        ) ** 2
        sigma_noise = torch.sqrt(var_shot_noise + var_thermal_noise)
        noise = sigma_noise * torch.randn(
            *signal.shape, device=self._rng.device, generator=self._rng
        ).to(signal.device)

        # low pass filter:
        signal = super(Photodetector, self).forward(signal + noise)

        return signal
