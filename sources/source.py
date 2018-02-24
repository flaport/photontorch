''' Input Sources '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np


#####################
## Constant Source ##
#####################

class ConstantSource(object):
    ''' A source with constant amplitude '''
    def __init__(self, nw, amplitude=1):
        ''' ConstantSource __init__

        Create a source with constant amplitude

        Arguments
        ---------
        nw : the network for which to create the source
        amplitude=1: the amplitude of the network. One can also specify an array of
                     amplitudes, corresponding to the different batches of the simulation
        '''

        # shape of the virtual array represented by this source:
        self.shape = (2, nw.env.num_timesteps, nw.nmc, nw.env.num_batches)

        # shape of the virtual array returned by this source:
        self.source = nw.new_variable(nw.zeros((
                nw.env.num_wl, # number of wavelengths in the simulation
                nw.nmc, # number of memory containing nodes in the network
                nw.env.num_batches, # number of parallel simulations
        ), cuda=nw.is_cuda))

        self.source[:, :nw.num_sources] = 1

        if isinstance(amplitude, np.ndarray):
            amplitude = nw.new_variable(torch.from_numpy(amplitude))
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
        return key[0]*self.source


#################
## Time Source ##
#################

class TimeSource(ConstantSource):
    ''' A source defined by a time signal '''
    def __init__(self, nw, signal):
        ''' TimeSource __init__

        Arguments
        ---------
        nw : the network for which to create the source
        signal : the time signal of the source
        '''
        ConstantSource.__init__(self, nw, 1)
        self.signal = signal

    def __getitem__(self, key):
        '''
        Get the source signal at a certain timestep.
        When indexing the source, the first index should be 0 or 1 (real or imag part),
        while the second index should be the time index
        '''
        return key[0]*float(self.signal[key[1]])*self.source


################
## Bit Source ##
################

class BitSource(TimeSource):
    ''' A source with a bit signal '''
    def __init__(self, nw, bits, bitrate, rising_fraction=0.1):
        ''' BitSource __init__

        Arguments
        ---------
        nw : the network for which to create the source
        bits : the bits of the source
        bitrate : the bitrate of the source signal
        rising_fraction : the fraction of the bit period that the signal rises (or falls)
                          from 0 to 1 (or from 1 to 0).
        '''

        # set variables
        self.bits = np.asarray(bits)
        self.samplerate = 1/nw.env.dt
        self.rising_fraction = rising_fraction

        # set properties:
        self.bitrate = bitrate

    @property
    def bitrate(self):
        ''' The exact bitrate is determined by the timestep of the simulation '''
        return self.samplerate/float(self.bitlength)
    @bitrate.setter
    def bitrate(self, value):
        ''' The exact bitrate is determined by the timestep of the simulation '''
        self.bitlength = int(self.samplerate/value + 0.5)
        self.signal = self._signal(self.bits)
    @property
    def riselength(self):
        ''' The exact riselength is determined by the timestep of the simulation '''
        return int(self.rising_fraction*self.bitlength+0.5)

    def _signal(self, bits):
        ''' Calculation of the complete bitstream '''
        _bits = np.hstack(([bits[0]], bits))
        bits_ = np.hstack((bits, [bits[-1]]))
        stream = np.hstack([self._edge(b1, b2) for b1, b2 in zip(_bits,bits_)])
        crop_start = self.bitlength // 2
        crop_end = self.bitlength // 2 + self.bitlength % 2
        stream = stream[crop_start:-crop_end]
        return stream

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
