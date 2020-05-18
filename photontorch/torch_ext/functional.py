""" Photontorch Functional

Mostly for error metrics not found in PyTorch itself.

"""

#############
## Imports ##
#############

## Torch
import torch

## 3rd party
import numpy as np
from scipy.signal import butter, lfilter


###############
## Functions ##
###############


class BitStreamGenerator:
    """BitStreamGenerator

    Generate a bitstream from a sequence of bits (or from a random seed)

    """

    def __init__(
        self,
        bitrate=40e9,
        samplerate=160e9,
        cutoff_frequency=None,
        filter_order=4,
        seed=None,
        dtype=None,
        device=None,
    ):
        """
        Args:
            bitrate (float): [1/s] data rate of the bitstream
            samplerate (float): [1/s] sample rate of the bitstream
            cutoff_frequency (float): [1/s] cutoff frequency of the bitstream. If None: no lowpass filtering.
            filter_order (int): filter order to enforce cutoff frequency
            seed (int): seed used to generate bits (if needed)
            dtype (torch.dtype):: dtype to generate the bits for. None -> "torch.get_default_dtype()"
            device (torch.device):: device to generate the bits on. None -> "cpu"

        """
        self.bitrate = float(bitrate)
        self.samplerate = float(samplerate)
        self.cutoff_frequency = (
            None if cutoff_frequency is None else float(cutoff_frequency)
        )
        self.filter_order = None if cutoff_frequency is None else int(filter_order)
        self.seed = None if seed is None else int(seed)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self._rng = np.random.RandomState(seed=self.seed)

    def __call__(self, bits=100):
        """generate a bitstream from a sequence of bits (or from a random seed)

        Args:
            bits (int|sequence): - if int: generate that number of bits, then create stream.
                - if sequence: interpret the sequence as bits, then create stream.
        """
        try:
            if len(bits.shape) == 0:
                bits = int(bits.item())
        except:
            pass

        if torch.is_tensor(bits):
            bits = bits.detach().cpu().numpy()

        if isinstance(bits, int):
            bits = self._rng.rand(bits) > 0.5

        bits = np.array(bits, dtype=np.float64)
        num_samples_per_n_bits = int(self.samplerate) // np.gcd(
            int(self.samplerate), int(self.bitrate)
        )
        n_bits = int(self.bitrate) // np.gcd(int(self.samplerate), int(self.bitrate))
        stream = (
            np.stack([bits] * num_samples_per_n_bits, 1)
            .reshape(-1, *bits.shape[1:])[::n_bits]
            .copy()
        )

        if self.cutoff_frequency is not None:
            normal_cutoff = self.cutoff_frequency / (0.5 * self.samplerate)
            if normal_cutoff > 1.0:
                raise ValueError(
                    "The samplerate of the signal is smaller than the nyquist "
                    "frequency [=2 x cutoff_frequency]\n"
                    "%.2e > %.2e" % (self.samplerate, 2 * self.cutoff_frequency)
                )
            if normal_cutoff < 1.0:
                b, a = butter(
                    N=self.filter_order, Wn=normal_cutoff, btype="lowpass", analog=False
                )
                stream = lfilter(b, a, stream, axis=0)

        stream = torch.tensor(stream, dtype=self.dtype, device=self.device)

        return stream


class MSELoss(torch.nn.Module):
    """ Mean Squared Error for bitstreams """

    def __init__(self, latency=0.0, warmup=0, bitrate=40e9, samplerate=160e9):
        """
        Args:
            latency (float): fractional latency [in bit lengths]. This value can be a floating point number bigger than 1.
            warmup (int): integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate (float): the bit rate of the signal [in Hz]
            samplerate (float): the sample rate of the signal [in Hz]
        """
        super(MSELoss, self).__init__()
        self.bitrate = float(bitrate)
        self.samplerate = float(samplerate)
        self.latency = float(latency)
        self.warmup = int(warmup + 0.5)

    def forward(
        self,
        prediction,
        target,
        latency=None,
        warmup=None,
        bitrate=None,
        samplerate=None,
    ):
        """ Mean Squared Error for bitstreams
        Args:
            prediction (Tensor): 4D output power tensor with shape (# timesteps, # wavelengths, # readouts, # batches)
            target (Tensor): target power tensor. Should be broadcastable to the same shape as prediction.
            **kwargs: all keyword arguments can be used to temporary override values given during the MSEloss initialization.
        """
        bitrate = self.bitrate if bitrate is None else float(bitrate)
        samplerate = self.samplerate if samplerate is None else float(samplerate)
        latency = self.latency if latency is None else float(latency)
        warmup = self.warmup if warmup is None else int(warmup + 0.5)

        if not torch.is_tensor(prediction):
            prediction = torch.tensor(prediction)
        if not torch.is_tensor(target):
            target = torch.tensor(
                target, dtype=prediction.dtype, device=prediction.device
            )
        target = target.to(dtype=prediction.dtype, device=prediction.device)

        while len(prediction.shape) < 4:
            prediction = prediction[:, None]
        while len(target.shape) < 4:
            target = target[:, None]
        try:
            prediction, target = torch.broadcast_tensors(
                prediction.clone(), target.clone()
            )
        except RuntimeError:
            raise RuntimeError(
                "failed to broadcast target in the same shape as prediction"
            )

        nt, _, _, _ = prediction.shape

        # delay sequences with warmup and latency
        l = int(latency * samplerate / bitrate + 0.5)  # latency sample points
        w = int(int(warmup + 0.5) * samplerate / bitrate + 0.5)  # warmup sample points
        target = target[w::]
        prediction = prediction[w + l : :]

        # make sure both sequences have the same length:
        m = min(target.shape[0], prediction.shape[0])
        target = target[:m]
        prediction = prediction[:m]

        # calculate mse:
        error = ((target - prediction) ** 2).mean()

        return error


class BERLoss(torch.nn.Module):
    """ Bit Error Rate (non-differentiable)"""

    def __init__(
        self, threshold=0.5, latency=0.0, warmup=0, bitrate=40e9, samplerate=160e9
    ):
        """
        Args:
            threshold (float): where to place the 0 / 1 threshold on the output power.
            latency (float): fractional latency [in bit lengths]. This value can be a floating point number bigger than 1.
            warmup (int): integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate (float): the bit rate of the signal [in Hz]
            samplerate (float): the sample rate of the signal [in Hz]
        """
        super(BERLoss, self).__init__()
        self.threshold = float(threshold)
        self.bitrate = float(bitrate)
        self.samplerate = float(samplerate)
        self.latency = float(latency)
        self.warmup = int(warmup + 0.5)

    def forward(
        self,
        prediction,
        target,
        threshold=None,
        latency=None,
        warmup=None,
        bitrate=None,
        samplerate=None,
    ):
        """ Bit Error Rate (non-differentiable)
        Args:
            prediction (Tensor): 4D output power tensor with shape (# timesteps, # wavelengths, # readouts, # batches)
            target (Tensor): target power tensor. Should be broadcastable to the same shape as prediction.
            **kwargs: all keyword arguments can be used to temporary override values given during the BERLoss initialization.
        """
        threshold = self.threshold if threshold is None else float(threshold)
        bitrate = self.bitrate if bitrate is None else float(bitrate)
        samplerate = self.samplerate if samplerate is None else float(samplerate)
        latency = self.latency if latency is None else float(latency)
        warmup = self.warmup if warmup is None else int(warmup + 0.5)

        if not torch.is_tensor(prediction):
            prediction = torch.tensor(prediction)
        if not torch.is_tensor(target):
            target = torch.tensor(
                target, dtype=prediction.dtype, device=prediction.device
            )
        target = target.to(dtype=prediction.dtype, device=prediction.device)

        with torch.no_grad():
            while len(prediction.shape) < 4:
                prediction = prediction[:, None]
            while len(target.shape) < 4:
                target = target[:, None]
            try:
                prediction, target = torch.broadcast_tensors(
                    prediction.clone(), target.clone()
                )
            except RuntimeError:
                raise RuntimeError(
                    "failed to broadcast target in the same shape as prediction"
                )

            nt, nw, nd, nb = prediction.shape

            # always normalize target around 0 (this way we don't need to care about how the target bits are defined [0,1] or [-1, 1] or smth else)
            target -= target.mean(0)
            target /= target.std(0)

            # handle fractional sampling:
            rates_gcd = np.gcd(int(samplerate + 0.5), int(bitrate + 0.5))
            rs, rb = int(samplerate + 0.5) // rates_gcd, int(bitrate + 0.5) // rates_gcd
            samplerate = rb * samplerate
            prediction = torch.stack([prediction] * rb, 1).view(-1, nw, nd, nb)
            target = torch.stack([target] * rb, 1).view(-1, nw, nd, nb)

            # delay and sample sequences
            s = int(samplerate / bitrate + 0.5)  # samples per bit
            l = int(latency * samplerate / bitrate + 0.5)  # latency sample points
            w = int(int(warmup + 0.5) * samplerate / bitrate + 0.5)  # warmup samples
            target = target[w + s // 2 :: s]
            prediction = prediction[w + s // 2 + l :: s]

            # make sure both sequences have the same length:
            m = min(target.shape[0], prediction.shape[0])
            target = target[:m]
            prediction = prediction[:m]

            # find wrong bits
            wrong_bits = (prediction > threshold) != (target > 0.0)

            # calculate error
            error = wrong_bits.to(dtype=torch.float64).mean().item()

        return error
