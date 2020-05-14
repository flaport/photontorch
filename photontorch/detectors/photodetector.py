"""
# Photodetector


The photodetector transforms the raw output power (for example the result of a
photontorch simulation) according to a realistic filtering model.

"""

#############
## Imports ##
#############

# standard library
import warnings

# Torch
import torch

# 3rd Party
from scipy.signal import butter

# Relative
from ..torch_ext.nn import Module

###############
## Constants ##
###############

q = 1.602176563e-19  # [C] elementary charge
k = 1.3806488e-23  # [m2kg/Ks2] boltzmann constant
T = 300  # [K] room temperature


###################
## PhotoDetector ##
###################


class LowpassDetector(torch.nn.Module):
    """ LowpassDetector: Detect by lowpass filtering the signal.

    This detector performs the (differentiable) PyTorch equivalent of the
    following numpy/scipy function:

    ```
    from scipy.signal import butter, lfilter
    def detect(x, bitrate, samplerate, cutoff_frequency, filter_order):
        normal_cutoff = cutoff_frequency / ( 0.5 * samplerate)
        b, a = butter(N=filter_order, Wn=normal_cutoff, btype='lowpass', analog=False)
        return lfilter(b, a, x, axis=0)
    ```

    """

    def __init__(
        self, bitrate=40e9, samplerate=160e9, cutoff_frequency=20e9, filter_order=4,
    ):
        """
        Args:
            bitrate: float = 50e9: [1/s] data rate of the signal to filter
            samplerate: float = 80e9: [1/s] sample rate of the signal to filter
            cutoff_frequency: float = 25e9: [1/s] cutoff frequency of the detector
            filter_order: int = 4: filter order of the butter filter
        """
        super(LowpassDetector, self).__init__()
        self.bitrate = float(bitrate)
        self.samplerate = float(samplerate)
        self.cutoff_frequency = float(cutoff_frequency)
        self.filter_order = int(filter_order + 0.5)
        normal_cutoff = 2 * self.cutoff_frequency / self.samplerate
        if normal_cutoff > 1.0:
            raise ValueError(
                "The samplerate of the photodetector is smaller than the nyquist "
                "frequency [=2 x cutoff_frequency]\n"
                "%.2e > %.2e" % (self.samplerate, 2 * self.cutoff_frequency)
            )

        # get filter parameters:
        b, a = butter(self.filter_order, normal_cutoff, btype="lowpass", analog=False)
        a[0] *= -1
        self.register_buffer(  # we're gonna use a[::-1] but PyTorch does not have negative strides.
            "a", torch.tensor(a[::-1].copy(), dtype=torch.get_default_dtype())
        )
        self.register_buffer("b", torch.tensor(b, dtype=torch.get_default_dtype()))

    def forward(self, signal):
        if signal.shape[0] == 2:  # complex valued signal (photontorch convention)
            signal = signal[0] ** 2 + signal[1] ** 2  # amplitude -> power

        # reshape to (# detected streams, # timesteps)
        original_shape = signal.shape  #
        signal = signal.view(signal.shape[0], -1).t()

        # convolve with b [first part of filtering]
        signal = torch.cat(
            [torch.zeros_like(signal[:, : (self.b.shape[-1] - 1)]), signal], -1
        )
        signal = torch.nn.functional.conv1d(signal[:, None, :], self.b[None, None])
        signal = signal[:, 0, :].t()

        # filter with a [second part of filtering]
        # TODO: find a way to fix bottleneck here.
        N = self.a.shape[0]
        for n in range(N, signal.shape[0], 1):
            signal[n] = -torch.sum(self.a[:, None] * signal[n - N + 1 : n + 1], 0)

        # reshape to original form
        signal = signal.view(*original_shape)

        return signal


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
            bitrate: float = 50e9: [1/s] data rate of the signal to filter
            samplerate: float = 80e9: [1/s] sample rate of the signal to filter
            cutoff_frequency: float = 25e9: [1/s] cutoff frequency of the detector
            responsivity: float = 1.0: [A/W] responsivity of the photodector
            dark_current: float = 1e-10: [A] dark current adding to the noise
            load_resistance: float = 1e6: [Ohm] load resistance of the detector
            filter_order: int = 4: filter order of the butter filter
            seed: int = 0: random seed for the detector noise
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

    def forward(self, signal):
        if signal.shape[0] == 2:  # complex valued signal (photontorch convention)
            signal = signal[0] ** 2 + signal[1] ** 2  # amplitude -> power

        # reshape to (# detected streams, # timesteps)
        original_shape = signal.shape  #
        signal = signal.view(signal.shape[0], -1)

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
        signal = (
            super(Photodetector, self).forward(signal + noise).view(*original_shape)
        )

        return signal
