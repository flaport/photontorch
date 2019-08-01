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


class Photodetector(Module):
    """ Realistic Photodector Model.

    The photodetector transforms the raw output power (for example the result of a
    photontorch simulation) according to a realistic filtering model.

    """

    def __init__(
        self,
        bitrate=50e9,
        frequency=80e9,
        bandwidth=25e9,
        responsivity=1.0,
        dark_current=1e-10,
        load_resistance=1e6,
        filter_order=4,
        seed=0,
    ):
        """
        Args:
            bitrate: float = 50e9: data rate of the signal to filter
            frequency: float = 80e9: highest frequency of the signal to filter
            bandwidth: float = 25e9: bandwidth of the photodector
            responsivity: float = 1.0: responsivity of the photodector
            dark_current: float = 1e-10: dark current adding to the noise
            load_resistance: float = 1e6: load resistance of the detector
            filter_order: int = 4: filter order of the butter filter
            seed: int = 0: random seed of the noise
        """
        super(Photodetector, self).__init__()
        self.bitrate = bitrate  # bitrate of the input signal
        self.bandwidth = bandwidth  # Bandwidth
        self.responsivity = responsivity  # A/W
        self.dark_current = dark_current  # dark current
        self.frequency = frequency
        self.load_resistance = load_resistance  # load resistance
        self.filter_order = filter_order  # filter order of the butter filter
        self.seed = seed

        self.normal_cutoff = self.bandwidth / self.frequency
        if self.normal_cutoff > 1.0:
            warnings.warn(
                "bandwidth of the detector is bigger than its highest filter "
                "frequency. The detector will be disabled."
            )
            return
        b, a = butter(
            self.filter_order, self.normal_cutoff, btype="lowpass", analog=False
        )

        # we reverse the order of a for efficiency.
        # reversing it later is not possible, as pytorch does not allow negative step sizes.
        self.register_buffer("a", torch.tensor(a[::-1].copy(), dtype=torch.float64))
        b = torch.tensor(b, dtype=torch.float64)[None, None, :]
        self.conv = torch.nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=b.shape[-1], bias=False
        )

        # we hack the convolution layer to have no trainable weights
        # by replacing its parameters by buffers.
        del self.conv._parameters["bias"]
        del self.conv._parameters["weight"]
        self.conv.bias = None
        self.conv.register_buffer("weight", b)

    def forward(self, signal):
        # before we can act on the signal, we need to transform it to float64.
        # otherwise the filtering does not work as expected
        signal = signal.to(torch.float64)

        # we will perform a convolution, however, the convolution layer needs the signal
        # in a specific shape: (# batches, # in channels, # time)
        # our convention in photontorch is different, so we reshape the signal:
        signal_shape = signal.shape
        signal = signal.view(signal.shape[0], -1).t()
        signal = signal[:, None, :]

        # we prepend zeros to make sure the filtered signal and the original signal
        # will have the same shape [pytorch convolutions default to kind='valid']
        if self.normal_cutoff <= 1:
            zero = torch.zeros_like(signal[:, :, : (self.conv.weight.shape[-1] - 1)])
            signal = torch.cat([zero, signal], -1)

        # Add random noise according to specified seed
        initial_random_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        noise_sd = torch.sqrt(
            2
            * q
            * self.bitrate
            * (torch.mean(self.responsivity * signal, 0) + self.dark_current)
            + 4 * k * T * self.bitrate / self.load_resistance
        )
        signal = (
            noise_sd
            * torch.randn(*signal.shape, device=signal.device, dtype=torch.float64)
            + self.responsivity * signal
        )
        torch.random.set_rng_state(initial_random_state)

        if self.normal_cutoff > 1:
            return signal.view(*signal_shape)

        # convolve with b [first part of filtering]
        signal = self.conv(signal)
        signal = signal[:, 0, :].t()

        # filter with a [second part of filtering]
        N = len(self.a)
        filtered_signal = signal[:N].clone()
        for n in range(N, len(signal), 1):
            x = torch.sum(self.a[:-1, None] * filtered_signal[n - N + 1 : n], 0)
            filtered_signal = torch.cat([filtered_signal, (signal[n] - x)[None]], 0)

        # reshape to original form
        filtered_signal = filtered_signal.view(*signal_shape)

        return filtered_signal.to(torch.get_default_dtype())
