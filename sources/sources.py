''' Input Sources '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from ..torch_ext import is_variable


#################
## Base Source ##
#################

class BaseSource(object):
    ''' Source BaseClass.

    A source represents a torch.Tensor with shape
    (2=(real|imag), # timesteps, # MC nodes, # batches)
    Yet, internally, it only stores the bare minimum of values to not be too
    demanding on the RAM.

    A Source should contain a reference to the network that created it and it should
    contain a single __getitem__ method, such that the source value at the time index
    specified can be generated on the fly.

    Attributes:
        nw (Network): The Network that created the source

    Note:
        Sources should be create from an initialized Network:

    ```
            source = nw.BaseSource()
    ```
        This way, the source will get the reference to the network so that it can calculate
        the correct source values at the correct ports.
    '''

    nw = None # Network that created the source. This will be set by the network itself.

    def __init__(self):
        ''' BaseSource

        Create a new source representing a virtual tensor with shape
        (2=(real|imag), # timesteps, # MC nodes, # batches)
        '''
        # shape of the virtual array represented by this source:
        self.shape = (2, self.nw.env.num_timesteps, self.nw.nmc, self.nw.env.num_batches)

    def size(self, idx=None):
        ''' Get the size (or shape) of the virtual source signal '''
        if idx is None:
            return self.shape
        return self.shape[idx]

    def __getitem__(self, key):
        ''' Get the source signal at a certain timestep.

        Args:
            key (tuple): When indexing the source the first the first index should
            be 0 or 1 (real or imag part), while the second index should be the time index.
        '''
        raise NotImplementedError()



#####################
## Constant Source ##
#####################

class ConstantSource(BaseSource):
    ''' A source with constant amplitude '''

    def __init__(self, amplitude=1):
        ''' ConstantSource

        Create a source with constant amplitude

        Args:
            amplitude=1 (float): the amplitude of the network. One can also specify an
                array of amplitudes, corresponding to the different batches of the simulation

        Note:
            Different amplitudes for different source nodes is not yet supported.

        '''
        # initialize the base version
        # NOTE: You cannot use super() here because of the class copying to the network
        BaseSource.__init__(self)

        # shape of the virtual array returned by this source:
        self.real_source = self.nw.new_variable(self.nw.zeros((
            self.nw.env.num_wl, # number of wavelengths in the simulation
            self.nw.nmc, # number of memory containing nodes in the network
            self.nw.env.num_batches, # number of parallel simulations
        ), cuda=self.nw.is_cuda))
        self.imag_source = torch.zeros_like(self.real_source)


        # set amplitude of the source nodes to one
        # remember that the source nodes are the first nodes listed in the network
        self.real_source[:, :self.nw.num_sources] = 1
        self.imag_source[:, :self.nw.num_sources] = 1

        # make amplitude a tensor:
        if torch.is_tensor(amplitude):
            amplitude = self.nw.new_variable(amplitude)
        if is_variable(amplitude):
            real_amplitude = amplitude
            imag_amplitude = torch.zeros_like(amplitude)
        else:
            real_amplitude = np.asarray(np.real(amplitude))
            imag_amplitude = np.asarray(np.imag(amplitude))
            if real_amplitude.ndim == 0: # Torch cannot handle 0D tensors
                real_amplitude = real_amplitude[None] # make 1D
                imag_amplitude = imag_amplitude[None] # make 1D
            # make a tensor from the amplitude:
            real_amplitude = self.nw.new_variable(torch.from_numpy(real_amplitude))
            imag_amplitude = self.nw.new_variable(torch.from_numpy(imag_amplitude))

        # make sure the amplitude is a 3D tensor (so it can cast to the shape of the source)
        if len(real_amplitude.size()) == 1:
            real_amplitude.unsqueeze_(0)
            imag_amplitude.unsqueeze_(0)
        if len(real_amplitude.size()) == 2:
            real_amplitude.unsqueeze_(0)
            imag_amplitude.unsqueeze_(0)

        # multiply the source (right now still with amplitude 1) with the amplitude:
        self.real_source *= real_amplitude
        self.imag_source *= imag_amplitude

    def __getitem__(self, key):
        ''' Get the source signal at a certain timestep.

        Args:
            key (tuple): When indexing the source the first the first index should
            be 0 or 1 (real or imag part), while the second index should be the time index.
        '''
        if key[0] == 0:
            return self.real_source
        elif key[0] == 1:
            return self.imag_source



#################
## Time Source ##
#################
class TimeSource(ConstantSource):
    ''' A source defined by a time signal '''
    def __init__(self, signal):
        ''' TimeSource

        Args:
            signal (np.ndarray): the time signal of the source.
        '''
        ConstantSource.__init__(self, 1)
        self.store_signal(signal)

    def store_signal(self, signal):
        ''' Handle and store the time signal of the source into a torch FloatTensor

        This method sets the attributes real_source and imag_source.

        Args:
            signal (np.ndarray): the time signal of the source. If a 2D signal is given,
                then he second dimension signifies the different signal for different batches.

        Note:
            different source signals for different input nodes is not yet supported.
        '''
        if isinstance(signal, np.ndarray):
            real_signal = self.nw.new_variable(np.real(signal))
            imag_signal = self.nw.new_variable(np.imag(signal))
        elif torch.is_tensor(signal):
            real_signal = self.nw.new_variable(signal)
            imag_signal = torch.zeros_like(real_signal)
        else:
            real_signal = signal
            imag_signal = torch.zeros_like(real_signal)
        self.real_signal = real_signal
        self.imag_signal = imag_signal
        self.is_real = (self.imag_signal == 0).all()
        # Store numpy version of the signal for fast acces:
        self.signal = self.real_signal.data.cpu().numpy()
        if not self.is_real:
            self.signal = np.asarray(self.signal, dtype=complex)
            self.signal += self.imag_signal.data.cpu().numpy()

        if len(self.real_signal.size()) == 1:
            self.real_signal.unsqueeze_(1)
            self.imag_signal.unsqueeze_(1)

    def __getitem__(self, key):
        ''' Get the source signal at a certain timestep.

        Args:
            key (tuple): When indexing the source the first the first index should
            be 0 or 1 (real or imag part), while the second index should be the time index.
        '''
        time_idx = key[1]
        if time_idx >= self.real_signal.shape[0]:
            return 0*self.real_source # no signal anymore
        elif key[0] == 0: # Real Part
            return self.real_signal[time_idx]*self.real_source
        elif key[0] == 1: # Imag Part
            return self.imag_signal[time_idx]*self.real_source # self.real_source, because we
                                                               # initialized with amplitude 1.


################
## Bit Source ##
################

class BitSource(TimeSource):
    ''' A source with a bit stream as signal '''
    def __init__(self, bits, bitrate, rising_fraction=0.1):
        ''' BitSource

        Args:
            bits (np.ndarray(bool)): The bits to create the signal for. If a 2D bit array
                is given, then the second dimension results in different bits for different batches.
            bitrate: The bitrate of the signal.
            rising_fraction: The fraction of the bit period that the signal rises or falls

        Note:
            Different bit signals for different input nodes (or different amplitudes for
            different input nodes) is not yet supported.

        '''

        # set variables
        self.bits = np.asarray(bits)
        if self.bits.ndim == 1: # bits should be 2D
            self.bits = self.bits[:,None]
        self.samplerate = 1/self.nw.env.dt
        self.rising_fraction = rising_fraction

        # set properties:
        self.bitrate = bitrate

        # finish initialization
        TimeSource.__init__(self, self.signal)

    @property
    def bitrate(self):
        '''@property

        The bitrate of the signal.

        Note:
            Although a bitrate was given during initialization, the exact bitrate will
            depend on the timestep of the simulation. If your timestep is sufficiently
            small, this difference will be negligeable.

        Returns:
            float: the bitrate
        '''
        return self.samplerate/float(self.bitlength)
    @bitrate.setter
    def bitrate(self, value):
        '''@property

        Setting the bitrate will recalculate the complete time signal.

        Note:
            Although a bitrate was given during initialization, the exact bitrate will
            depend on the timestep of the simulation. If your timestep is sufficiently
            small, this difference will be negligeable.

        '''
        self.bitlength = int(self.samplerate/value + 0.5)
        self.store_signal(self._signal(self.bits))

    @property
    def riselength(self):
        '''@property

        The length in timesteps of the simulation that a bit is rising.
            int: the rise length of a bit
        '''
        return int(self.rising_fraction*self.bitlength+0.5)

    def _signal(self, bits):
        ''' Calculate a bit signal according to the specified bits

        Args:
            bits: (np.ndarray(bool)): The bits to calculate the signal for.

        Returns:
            np.ndarray: the bit stream signal.
        '''
        crop_start = self.bitlength // 2
        crop_end = self.bitlength // 2 + self.bitlength % 2

        signals = [None]*bits.shape[1]

        for i, batch_bits in enumerate(bits.T):
            _bits = np.hstack(([batch_bits[0]], batch_bits))
            bits_ = np.hstack((batch_bits, [batch_bits[-1]]))
            batch_signal = np.hstack([self._edge(b1, b2) for b1, b2 in zip(_bits,bits_)])
            signals[i] = batch_signal[crop_start:-crop_end]

        signal = np.stack(signals, -1)
        return signal

    def _edge(self, b1, b2):
        ''' Transition between two consecutive bits
        Args:
            b1 (bool): first bit
            b2 (bool): second bit

        Returns:
            np.ndarray: The transition between the bits for the bitrate and rising_fraction
                stored in the source.
        '''
        flatlength = self.bitlength - self.riselength
        x = 2.5*np.linspace(-1,1,self.riselength) # Factor 2.5 hardcoded for now.
        start = -np.ones(flatlength//2)
        middle = np.tanh(x)
        end = np.ones(flatlength//2)
        rest = [1]*(flatlength%2)
        edge = np.hstack([start, middle, end, rest])
        if b1 and b2: # stay up
            edge = np.ones(self.bitlength)
        elif b1 and not b2: # go down:
            edge = -edge
        elif not b1 and b2: # go up:
            edge = edge
        elif not b1 and not b2: # stay down
            edge = -np.ones(self.bitlength)
        return 0.5*(edge + 1.0) # scale between 0 and 1
