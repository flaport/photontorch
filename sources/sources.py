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
            amplitude=1 (float | dict):
                * float: the amplitude of the network. One can also specify an
                    array of amplitudes, corresponding to the different batches of the simulation
                    If a float is given, all Source terms will get the same amplitude
                * dict: One can also specify a dictionary with amplitudes. The keys
                    specify the name of the Source Term, while the values specify the
                    amplitude.

        '''
        # initialize the base version
        # NOTE: You cannot use super() here because of the class copying to the network
        BaseSource.__init__(self)

        # shape of the virtual array returned by this source:
        real_source = self.nw.zeros((
            self.nw.env.num_wl, # number of wavelengths in the simulation
            self.nw.nmc, # number of memory containing nodes in the network
            self.nw.env.num_batches, # number of parallel simulations
        ))
        imag_source = torch.zeros_like(real_source)

        # set amplitude of the source nodes to one
        # remember that the source nodes are the first nodes listed in the network
        real_source[:, :self.nw.num_sources] = 1
        imag_source[:, :self.nw.num_sources] = 1

        # change amplitudes from 1 to specified amplitude.
        sources = (comp for comp in self.nw.components if isinstance(comp, Source))
        if isinstance(amplitude, dict):
            for i, source in enumerate(sources):
                if source.name in amplitude:
                    amp = amplitude[source.name]
                    real_amplitude, imag_amplitude = self._handle_amplitude(amp)
                    real_source[:,i:i+1,:] *= real_amplitude
                    imag_source[:,i:i+1,:] *= imag_amplitude
        else:
            real_amplitude, imag_amplitude = self._handle_amplitude(amplitude)
            real_source *= real_amplitude
            imag_source *= imag_amplitude

        # save source values as variables
        self.real_source = self.nw.tensor(real_source)
        self.imag_source = self.nw.tensor(imag_source)

    def _handle_amplitude(self, amplitude):
        ''' Handle the amplitude of the source to form a torch FloatTensor

        Args:
            amplitude (float | np.ndarray | torch.Tensor): the amplitude of the source.
                if an array is given, the different amplitudes correspond to different
                batches.

        Returns:
            real_amplitude (torch.FloatTensor), imag_amplitude (torch.FloatTensor)
        '''
        # make amplitude a tensor:
        if torch.is_tensor(amplitude):
            real_amplitude = amplitude
            imag_amplitude = torch.zeros_like(amplitude)
        else:
            real_amplitude = np.asarray(np.real(amplitude))
            imag_amplitude = np.asarray(np.imag(amplitude))
            if real_amplitude.ndim == 0: # Torch cannot _handle 0D tensors
                real_amplitude = real_amplitude[None] # make 1D (will cast to # batches)
                imag_amplitude = imag_amplitude[None] # make 1D (will cast to # batches)
            # make a tensor from the amplitude:
            real_amplitude = self.nw.tensor(real_amplitude)
            imag_amplitude = self.nw.tensor(imag_amplitude)

        # make sure the amplitude is a 3D tensor (so it can cast to the shape of the source)
        if len(real_amplitude.shape) == 1: # will cast to # nodes
            real_amplitude.unsqueeze_(0)
            imag_amplitude.unsqueeze_(0)
        if len(real_amplitude.shape) == 2: # will cast to # timesteps
            real_amplitude.unsqueeze_(0)
            imag_amplitude.unsqueeze_(0)

        return real_amplitude, imag_amplitude

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
            signal (np.ndarray | dict): the time signal of the source.
        '''
        # Initialize with amplitude 1 at the active source nodes
        sources = (comp for comp in self.nw.components if isinstance(comp, Source))
        if isinstance(signal, dict):
            ConstantSource.__init__(self, amplitude={k:1 for k in signal})
            self.loc = [i for i, src in enumerate(sources) if src.name in signal]
            real_signals, imag_signals = [], []
            for sgl in signal.values():
                r_sgl, i_sgl = self._handle_signal(sgl)
                real_signals += [r_sgl]
                imag_signals += [i_sgl]
            real_signal = torch.cat(real_signals, dim=1)
            imag_signal = torch.cat(imag_signals, dim=1)
        else:
            ConstantSource.__init__(self, amplitude=1)
            self.loc = slice(None, None, None)
            real_signal, imag_signal = self._handle_signal(signal)

        self.real_signal = self.nw.tensor(real_signal)
        self.imag_signal = self.nw.tensor(imag_signal)

    @property
    def signal(self):
        ''' Numpy representation of the source signal '''
        signal = self.real_signal.data.cpu().numpy()
        isignal = self.imag_signal.data.cpu().numpy()
        if not (isignal == 0).all():
            signal = np.asarray(signal, dtype=complex)
            signal += 1j*isignal
        return signal

    def _handle_signal(self, signal):
        ''' Handle the time signal of the source to form a torch FloatTensor

        Args:
            signal (np.ndarray | torch.Tensor): the time signal of the source. if a 2D
                array is given, the second dimension corresponds to the different
                batches.

        Returns:
            real_signal (torch.FloatTensor), imag_signal (torch.FloatTensor)
        '''
        # make amplitude a tensor:
        if torch.is_tensor(signal):
            real_signal = signal
            imag_signal = torch.zeros_like(signal)
        else:
            real_signal = self.nw.tensor(np.asarray(np.real(signal)))
            imag_signal = self.nw.tensor(np.asarray(np.imag(signal)))

        # make sure the signal is a 3D tensor (so it can cast to the shape of the source)
        if len(real_signal.shape) == 1: # will cast to # batches
            real_signal.unsqueeze_(-1)
            imag_signal.unsqueeze_(-1)
        if len(real_signal.shape) == 2: # will cast to # nodes
            real_signal.unsqueeze_(1)
            imag_signal.unsqueeze_(1)

        return real_signal, imag_signal

    def expand(self, clear_memory=False):
        ''' Create a full torch variable from the current source

        Using an expanded source will speed up simulations in exchange for more RAM usage.

        Args:
            clear_memory=False (bool): clear the memory (this will make the current source
            unusable.)

        Returns
            A torch variable containing all the source data for all the MC nodes of the
            network.

        '''
        ret = self.nw.tensor(torch.zeros(self.shape))
        ret[:,:,self.loc,:] = torch.cat((
            self.real_signal*self.real_source[:,self.loc],
            self.imag_signal*self.imag_source[:,self.loc],
        ), dim=0)
        # free up memory
        if clear_memory:
            for k in self.__dict__.keys():
                self.__delattr__(k)
            del self
        return ret

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
            ret = torch.zeros_like(self.real_source)
            ret[:,self.loc] = self.real_signal[time_idx:time_idx+1]*self.real_source[:,self.loc]
            return ret
        elif key[0] == 1: # Imag Part
            ret = torch.zeros_like(self.imag_source)
            ret[:,self.loc] = self.imag_signal[time_idx:time_idx+1]*self.imag_source[:,self.loc]
            return ret


################
## Bit Source ##
################

class BitSource(TimeSource):
    ''' A source with a bit stream as signal '''
    def __init__(self, bits, bitrate, rising_fraction=0.1, amplitude=1):
        ''' BitSource

        Args:
            bits (np.ndarray(bool) | dict): The bits to create the signal for.
                If a 2D bit array is given, then the second dimension results in
                different bits for different batches. A dictionary of bits is also
                possible: in that case, the keys correspond to the names of the different
                Source Terms.
            bitrate: The bitrate of the signal.
            amplitude = (float | dict): The amplitude of the bitstream. If a dictionary
                is specified, the keys indicate the name of the Source Term and the
                values their respective amplitude.
            rising_fraction: The fraction of the bit period that the signal rises or falls

        '''
        self.amplitude = amplitude
        self.samplerate = 1/self.nw.env.dt
        self.rising_fraction = rising_fraction
        self.bitlength = int(self.samplerate/bitrate + 0.5)
        self.bits = self._handle_bits(bits)
        TimeSource.__init__(self, self._signal(self.bits, self.amplitude))

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
        self.__init__(self.bits, value, self.rising_fraction, self.weights)

    @property
    def riselength(self):
        '''@property

        The length in timesteps of the simulation that a bit is rising.
            int: the rise length of a bit
        '''
        return int(self.rising_fraction*self.bitlength+0.5)

    def _handle_bits(self, bits):
        ''' Create numpy array of bits in the expected form '''
        if isinstance(bits, dict):
            return {k:self._handle_bits(b) for k, b in bits.items()}
        bits = np.asarray(bits)
        if bits.ndim == 1: # bits should be 2D
            bits = bits[:,None]
        return bits

    def _signal(self, bits, amplitude=1):
        ''' Calculate a bit signal according to the specified bits

        Args:
            bits: (np.ndarray(bool) | dict): The bits to calculate the signal for. If
                a dictionary is specified, then the keys signify the Source Term name
                and the value specify the bits for that source term.
            amplitude=1 (float | dict): The amplitude / phase for the different input
                Source Terms. In case of a dictionary, the keys specifiy the source
                term name and the values specify the source term amplitudes/phases.

        Returns:
            np.ndarray: the bit stream signal.
        '''
        if isinstance(bits, dict) and isinstance(amplitude, dict):
            return {k:self._signal(b,amplitude=amplitude.get(k,1)) for k, b in bits.items()}
        elif isinstance(amplitude, dict):
            return {k:self._signal(bits, a) for k, a in amplitude.items()}
        elif isinstance(bits, dict):
            return {k:self._signal(b) for k, b in bits.items()}

        crop_start = self.bitlength // 2
        crop_end = self.bitlength // 2 + self.bitlength % 2

        signals = [None]*bits.shape[1]

        for i, batch_bits in enumerate(bits.T):
            _bits = np.hstack(([batch_bits[0]], batch_bits))
            bits_ = np.hstack((batch_bits, [batch_bits[-1]]))
            batch_signal = np.hstack([self._edge(b1, b2) for b1, b2 in zip(_bits,bits_)])
            signals[i] = batch_signal[crop_start:-crop_end]

        signal = float(amplitude)*np.stack(signals, -1)
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
