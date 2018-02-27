''' Input Sources '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np


#################
## Base Source ##
#################

class BaseSource(object):
    '''
    A Source MetaClass.
    A Source should contain a reference to the network that created it and it should
    contain a single __getitem__ method, such that the source value at a certain time
    can be generated.

    Note
    ----
    It is not recommended to create the sources from scratch here. The sources will
    be available in the network you created. You should create them from there:

    source = nw.BaseSource()

    This way, the source will get the reference to the network so that it can calculate
    is the correct source values at the correct ports.
    '''

    nw = None

    def __getitem__(self, key):
        '''
        Get the source signal at a certain timestep.
        When indexing the source, the first index should be 0 or 1 (real or imag part),
        while the second index should be the time index
        '''
        raise NotImplementedError()

#####################
## Constant Source ##
#####################


class ConstantSource(BaseSource):
    ''' A source with constant amplitude '''

    nw = None # Network for which this source is defined.

    def __init__(self, amplitude=1):
        ''' ConstantSource __init__

        Create a source with constant amplitude

        Arguments
        ---------
        amplitude=1: the amplitude of the network. One can also specify an array of
                    amplitudes, corresponding to the different batches of the simulation
        '''

        # shape of the virtual array represented by this source:
        self.shape = (2, self.nw.env.num_timesteps, self.nw.nmc, self.nw.env.num_batches)

        # shape of the virtual array returned by this source:
        self.source = self.nw.new_variable(self.nw.zeros((
                self.nw.env.num_wl, # number of wavelengths in the simulation
                self.nw.nmc, # number of memory containing nodes in the network
                self.nw.env.num_batches, # number of parallel simulations
        ), cuda=self.nw.is_cuda))

        self.source[:, :self.nw.num_sources] = 1

        if isinstance(amplitude, np.ndarray):
            amplitude = self.nw.new_variable(torch.from_numpy(amplitude))
            if len(amplitude.size()) == 1:
                amplitude.unsqueeze_(0)
            if len(amplitude.size()) == 2:
                amplitude.unsqueeze_(0)

        self.source *= amplitude

    def size(self, idx=None):
        ''' Get the size (or shape) of the complete source signal '''
        if idx is None:
            return self.shape
        return self.shape[idx]

    def __getitem__(self, key):
        '''
        Get the source signal at a certain timestep.
        When indexing the source, the first index should be 0 or 1 (real or imag part),
        while the second index should be the time index
        '''
        if key[0] == 0: # source has only a real part...
            return self.source
        elif key[0] == 1:
            return 0*self.source


#################
## Time Source ##
#################

class TimeSource(ConstantSource):
    ''' A source defined by a time signal '''
    def __init__(self, signal):
        ''' TimeSource __init__

        Arguments
        ---------
        signal : the time signal of the source
        '''
        ConstantSource.__init__(self, 1)

        self.store_signal(signal)

    def store_signal(self, signal):
        ''' Handle and store the time signal of the source '''
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
        '''
        Get the source signal at a certain timestep.
        When indexing the source, the first index should be 0 or 1 (real or imag part),
        while the second index should be the time index
        '''
        time_idx = key[1]
        if time_idx >= self.real_signal.shape[0]:
            return 0*self.source # no signal anymore
        elif key[0] == 0: # Real Part
            return self.real_signal[time_idx]*self.source
        elif key[0] == 1: # Imag Part
            return self.imag_signal[time_idx]*self.source


################
## Bit Source ##
################

class BitSource(TimeSource):
    ''' A source with a bit signal '''
    def __init__(self, bits, bitrate, rising_fraction=0.1):
        ''' BitSource __init__

        Arguments
        ---------
        bits : the bits of the source
        bitrate : the bitrate of the source signal
        rising_fraction : the fraction of the bit period that the signal rises (or falls)
                        from 0 to 1 (or from 1 to 0).
        '''

        # set variables
        self.bits = np.asarray(bits)
        if self.bits.ndim == 1:
            self.bits = self.bits[:,None]
        self.samplerate = 1/self.nw.env.dt
        self.rising_fraction = rising_fraction

        # set properties:
        self.bitrate = bitrate

        # finish initialization
        TimeSource.__init__(self, self.signal)

    @property
    def bitrate(self):
        ''' The exact bitrate is determined by the timestep of the simulation '''
        return self.samplerate/float(self.bitlength)
    @bitrate.setter
    def bitrate(self, value):
        ''' The exact bitrate is determined by the timestep of the simulation '''
        self.bitlength = int(self.samplerate/value + 0.5)
        self.store_signal(self._signal(self.bits))
    @property
    def riselength(self):
        ''' The exact riselength is determined by the timestep of the simulation '''
        return int(self.rising_fraction*self.bitlength+0.5)

    def _signal(self, bits):
        ''' Calculation of the complete bitsignal '''
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
        ''' Get transition between two consecutive bits '''
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
