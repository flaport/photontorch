""" A simple detector based on low-pass filtering the signal.

The ``LospassDetector`` performs the (differentiable) PyTorch equivalent of the
following numpy/scipy function: ::

    from scipy.signal import butter, lfilter
    def detect(x, bitrate, samplerate, cutoff_frequency, responsivity, filter_order):
        normal_cutoff = cutoff_frequency / ( 0.5 * samplerate)
        b, a = butter(N=filter_order, Wn=normal_cutoff, btype='lowpass', analog=False)
        return lfilter(b, a, responsivity * x, axis=0)

"""

#############
## Imports ##
#############

# Torch
import torch

# 3rd Party
from scipy.signal import butter

# Relative
from ..nn.nn import Module
from ..environment.environment import current_environment

###############
## Constants ##
###############

q = 1.602176563e-19  # [C] elementary charge
k = 1.3806488e-23  # [m2kg/Ks2] boltzmann constant
T = 300  # [K] room temperature

#############
## lfilter ##
#############

try:
    from torch_lfilter import lfilter  # C++ compiled version of lfilter below
except ImportError:
    from warnings import warn
    warn("Please install `torch_lfilter` for better detector performance")
    def lfilter(b, a, x):
        """PyTorch lfilter

        Args:
            b (torch.Tensor): The numerator coefficient vector in a 1-D sequence.
            a (torch.Tensor): The denominator coefficient vector in a 1-D sequence.
                if ``a[0]`` is not 1, then both ``a`` and ``b`` are normalized by ``a[0]``.
            x (torch.Tensor): An N-dimensional input tensor.

        Note:
            The filtering happens along dimension (axis) 0.
        """
        if not (b.ndim == a.ndim == 1):
            raise ValueError("filter vectors b and a should be 1D.")
        b = torch.tensor(
            [float(bb) / float(a[0]) for bb in b], dtype=x.dtype, device=x.device
        )[:, None]
        a = torch.tensor(
            [float(aa) / float(a[0]) for aa in reversed(a[1:])],
            dtype=x.dtype,
            device=x.device,
        )[:, None]

        order = b.shape[0]
        num_timesteps = x.shape[0]
        original_shape = x.shape

        x = x.reshape(num_timesteps, -1)
        y = torch.zeros_like(x)

        y[0] = y[0] + b[-1] * x[0]
        for n in range(1, order, 1):
            y[n] = y[n] + (b[-1 - n :] * x[: n + 1]).sum(0)
            y[n] = y[n] - (a[-n:] * y[:n]).sum(0)

        for n in range(order, num_timesteps, 1):
            y[n] = y[n] + (b * x[n - order + 1 : n + 1]).sum(0)
            y[n] = y[n] - (a * y[n - order + 1 : n]).sum(0)

        return y.reshape(*original_shape)


#####################
## LospassDetector ##
#####################


class LowpassDetector(Module):
    """ Detect by lowpass filtering the signal.

    The LowpassDetector transforms a raw optical power [W] to a detection current [A].

    """

    def __init__(
        self,
        bitrate=40e9,
        samplerate=160e9,
        cutoff_frequency=20e9,
        filter_order=4,
        responsivity=1.0,
    ):
        """
        Args:
            bitrate (float): [1/s] data rate of the signal to filter
            samplerate (float): [1/s] sample rate of the signal to filter
            cutoff_frequency (float): [1/s] cutoff frequency of the detector
            filter_order (int): filter order of the butter filter
            responsivity (float): [A/W] resonsivity of the detector
        """
        super(LowpassDetector, self).__init__()
        self.bitrate = float(bitrate)
        self.samplerate = float(samplerate)
        self.cutoff_frequency = float(cutoff_frequency)
        self.filter_order = int(filter_order + 0.5)
        self.responsivity = float(responsivity)

        normal_cutoff = 2 * self.cutoff_frequency / self.samplerate
        if normal_cutoff > 1.0:
            raise ValueError(
                "The samplerate of the signal is smaller than the nyquist "
                "frequency [=2 x cutoff_frequency]\n"
                "%.2e < %.2e" % (self.samplerate, 2 * self.cutoff_frequency)
            )

    def forward(
        self,
        signal,
        bitrate=None,
        samplerate=None,
        cutoff_frequency=None,
        filter_order=None,
        responsivity=None,
    ):
        """ detect a bitstream by low-pass filtering

        Args:
            signal (Tensor): [W] optical power to detect.
            bitrate (optional, float): [1/s] override data rate of the signal to filter
            samplerate (optional, float): [1/s] override sample rate of the signal to filter
            cutoff_frequency (optional, float): [1/s] override cutoff frequency of the detector
            filter_order (optional, int): override filter order of the butter filter
            responsivity (optional, float): [A/W] override resonsivity of the detector

        Note:
            If a bitrate and/or samplerate can be found in the current
            environment, those values will be regarded as keyword arguments and
            hence get precedence over the values given during the detector
            initialization.

        Note:
            The detector is quite efficient on CPU-tensors, but not on CUDA
            (GPU) tensors. Consider converting the signal to CPU before
            detecting.
        """

        # handle arguments
        try:
            env = current_environment()
            bitrate = env.bitrate if bitrate is None else bitrate
            samplerate = env.samplerate if samplerate is None else samplerate
        except RuntimeError:
            pass
        cutoff_frequency = (
            self.cutoff_frequency
            if cutoff_frequency is None
            else float(cutoff_frequency)
        )
        filter_order = (
            self.filter_order if filter_order is None else int(filter_order + 0.5)
        )
        responsivity = (
            self.responsivity if responsivity is None else float(responsivity)
        )
        bitrate = self.bitrate if bitrate is None else float(bitrate)
        samplerate = self.samplerate if samplerate is None else float(samplerate)

        # unit: sqrt(W) -> W (only when complex valued amplitudes are given)
        if signal.shape[0] == 2:  # complex valued signal (photontorch convention)
            signal = signal[0] ** 2 + signal[1] ** 2  # amplitude -> power

        # unit: W -> A:
        signal = responsivity * signal

        # calculate normal cutoff frequency
        normal_cutoff = 2 * cutoff_frequency / samplerate

        # do no filtering when nyquist frequency equals samplerate
        if normal_cutoff == 1.0:
            return signal

        # error when samplerate is too low for chosen cutoff_frequency
        if normal_cutoff > 1.0:
            raise ValueError(
                "The samplerate of the signal is smaller than the nyquist "
                "frequency [=2 x cutoff_frequency]\n"
                "%.2e < %.2e" % (samplerate, 2 * cutoff_frequency)
            )

        # set filter parameters:
        b, a = butter(filter_order, normal_cutoff, btype="lowpass", analog=False)

        return lfilter(b, a, signal)
