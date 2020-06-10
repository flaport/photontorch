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
from ..torch_ext.nn import Module
from ..environment.environment import current_environment

###############
## Constants ##
###############

q = 1.602176563e-19  # [C] elementary charge
k = 1.3806488e-23  # [m2kg/Ks2] boltzmann constant
T = 300  # [K] room temperature


#####################
## LospassDetector ##
#####################


class LowpassDetector(torch.nn.Module):
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
        num_splits=1,
        split_padding=5,
        bitrate=None,
        samplerate=None,
        cutoff_frequency=None,
        filter_order=None,
        responsivity=None,
    ):
        """ detect a bitstream by low-pass filtering

        Args:
            signal (Tensor): [W] optical power to detect.
            num_splits (int): number of parallel parts to split the timestream in
            split_padding (int): number of bits padding when splitting the timstream in parts.
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
            Splitting the signal in parts to be processed in parallel can considerably
            speed up the detection process. However, it remains an approximation, as
            in theory each detected signal point depends (with an exponentially
            decreasing factor) on all previous detected signals.

            To partly circumvent any large discontinuities when recomposing the
            detected signal from the detected parts, a `split_padding` should be
            defined, which pads a number of bits from the previous / next
            signal part before and after the current signal part to detect.
        """

        # handle arguments
        num_splits = int(num_splits + 0.5)
        split_padding = int(split_padding + 0.5)
        if num_splits > 1 and split_padding < 1:
            raise ValueError(
                "split padding should be larger than 0 when the number of splits is bigger than 1"
            )
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
        a = torch.tensor(
            a[::-1, None].copy()[:-1], dtype=signal.dtype, device=signal.device
        )
        b = torch.tensor(b[::-1, None].copy(), dtype=signal.dtype, device=signal.device)

        # reshape to (# detected streams, # timesteps)
        original_shape = signal.shape  #
        signal = signal.reshape(signal.shape[0], -1)
        num_signals = signal.shape[-1]

        if num_splits > 1:
            if signal.shape[0] % num_splits:
                raise ValueError(
                    "num_splits=%i does not cleanly devide the number of timesteps in the signal (%i)"
                    % (num_splits, signal.shape[0])
                )
            short_length = signal.shape[0] // num_splits
            f = min(split_padding * int(samplerate / bitrate), short_length)
            signal = signal.reshape(num_splits, short_length, num_signals)
            prepend = torch.cat(
                [torch.zeros_like(signal[:1, -f:]), signal[:-1, -f:]], 0
            )
            append = torch.cat([signal[1:, :f], torch.zeros_like(signal[-1:, :f])], 0)
            signal = torch.cat([prepend, signal, append], 1)
            long_length = signal.shape[1]
            signal = signal.permute(1, 2, 0).reshape(long_length, -1)

        a = a.to(dtype=signal.dtype, device=signal.device)
        b = b.to(dtype=signal.dtype, device=signal.device)

        # custom lfilter implementation
        N = a.shape[0] + 1
        filtered = signal.clone()
        filtered[0] = (b[-1:] * signal[:1]).sum(0)
        for n in range(1, N, 1):
            filtered[n] = (b[-1 - n :] * signal[: n + 1]).sum(0)
            filtered[n] = filtered[n] - (a[-n:] * filtered[:n]).sum(0)
        for n in range(N, signal.shape[0], 1):
            filtered[n] = (b * signal[n - N + 1 : n + 1]).sum(0)
            filtered[n] = filtered[n] - (a * filtered[n - N + 1 : n]).sum(0)

        if num_splits > 1:
            filtered = (
                filtered.reshape(long_length, num_signals, num_splits)[f:-f]
                .permute(2, 0, 1)
                .reshape(*original_shape)
            )
        else:
            filtered = filtered.reshape(*original_shape)

        return filtered
