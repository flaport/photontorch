''' Photodetector module '''

#############
## Imports ##
#############

# Torch
import torch

# 3rd Party
from scipy.signal import butter

# Relative
from ..torch_ext.nn import Module

###############
## Constants ##
###############

q = 1.602176563e-19 #[C] elementary charge
k = 1.3806488e-23 #[m2kg/Ks2] boltzmann constant
T = 300 #[K] room temperature


###################
## PhotoDetector ##
###################

class Photodetector(torch.nn.Module):
    ''' Realistic Photodector Model. '''
    def __init__(self, bitrate=50e9, dt=1e-12, bandwidth=25e9, responsivity=1, dark_current=0.1e-9,
                 load_resistance=1e6, filter_order=4, seed=0):
        ''' Photodector __init__
        Args:
            bitrate (float): data rate of incoming signal
            bandwidth (float): Bandwidth of the photodector
            responsivity (float) : Responsivity of the photodector
            dt (float): Sampling duration of the input signal
            dark_current (float): Dark current
            load_resistance (float) : Load resistance of the detector
            filter_order (int) : filter order of the butter filte
            seed (int) : random seed of the noise
        '''
        super(Photodetector, self).__init__()
        self.bitrate = bitrate # bitrate of the input signal
        self.bandwidth = bandwidth # Bandwidth
        self.tau = 1./bandwidth # RC time constant
        self.responsivity = responsivity #A/W
        self.dark_current = dark_current # dark current
        self.dt = dt # sampling timestep
        self.sampling_rate = 1./dt # sampling rate
        self.load_resistance = load_resistance # load resistance
        self.filter_order = filter_order # filter order of the butter filter
        self.seed = seed

        nyq = 0.5 * self.sampling_rate
        normal_cutoff = self.bandwidth / nyq
        if normal_cutoff >= 1:
            normal_cutoff = 0.99
        b, a = butter(self.filter_order, normal_cutoff, btype='lowpass', analog=False)

        # we reverse the order of a for efficiency.
        # reversing it later is not possible, as pytorch does not allow negative step sizes.
        self.register_buffer('a', torch.tensor(a[::-1].copy(), dtype=torch.float64))
        b = torch.tensor(b, dtype=torch.float64)[None, None, :]
        self.conv = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=b.shape[-1],
            bias=False,
        )

        # we hack the convolution layer to have no trainable weights
        # by replacing its parameters by buffers.
        del self.conv._parameters['bias']
        del self.conv._parameters['weight']
        self.conv.bias = None
        self.conv.register_buffer('weight', b)

    # for easy access of the b parameter
    @property
    def b(self):
        ''' get b-parameter for lowpass filtering '''
        return self.conv.weight
    @b.setter
    def b(self, value):
        ''' set b-parameter for lowpass filtering '''
        del self.conv._buffers['weight']
        self.conv.register_buffer('weight', value)

    def forward(self, signal):
        # we will perform a convolution, however, the convolution layer needs the signal
        # in a specific shape: (# batches, # in channels, # time)
        # our convention in photontorch is different, so we reshape the signal:
        signal = signal.double()
        signal_shape = signal.shape
        signal = signal.view(signal.shape[0], -1).t()
        signal = signal[:, None, :]

        # we prepend zeros to make sure the filtered signal and the original signal
        # will have the same shape [pytorch convolutions default to kind='valid']
        zero = torch.zeros_like(signal[:,:,:(self.conv.weight.shape[-1]-1)])
        signal = torch.cat([zero, signal], -1)

        # Add random noise
        initial_random_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        noise_sd = torch.sqrt(2*q*self.bitrate*(torch.mean(self.responsivity*signal, 0)
                                                + self.dark_current)
                              + 4*k*T*self.bitrate/self.load_resistance)
        signal = (noise_sd*torch.randn(*signal.shape, device=signal.device, dtype=torch.float64)
                  + self.responsivity*signal)
        torch.random.set_rng_state(initial_random_state)

        # convolve with b [first part of linear filtering]
        signal = self.conv(signal)
        signal = signal[:,0,:].t()

        # filter with a [second part of linear filtering]
        N = len(self.a)
        filtered_signal = signal[:N].clone()
        for n in range(N, len(signal), 1):
            x = torch.sum(self.a[:-1, None]*filtered_signal[n-N+1:n],0)
            filtered_signal = torch.cat([filtered_signal, (signal[n]-x)[None]], 0)

        # reshape to original form
        filtered_signal = filtered_signal.view(*signal_shape)

        return filtered_signal.float()
