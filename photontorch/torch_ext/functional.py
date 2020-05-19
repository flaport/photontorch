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
        filter_order=1,
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
            dtype (torch.dtype): dtype to generate the bits for. None -> "torch.get_default_dtype()"
            device (torch.device): device to generate the bits on. None -> "cpu"
        """
        self.bitrate = float(bitrate)
        self.samplerate = float(samplerate)
        self.cutoff_frequency = (
            None if cutoff_frequency is None else float(cutoff_frequency)
        )
        self.filter_order = (
            None if cutoff_frequency is None else int(filter_order + 0.5)
        )
        self.seed = None if seed is None else int(seed + 0.5)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self._rng = np.random.RandomState(seed=self.seed)

    def __call__(
        self,
        bits=100,
        bitrate=None,
        samplerate=None,
        cutoff_frequency=None,
        filter_order=None,
        seed=None,
        dtype=None,
        device=None,
    ):
        """generate a bitstream from a sequence of bits (or from a random seed)

        Args:
            bits (int|sequence): - if int: generate that number of bits, then create stream.
                - if sequence: interpret the sequence as bits, then create stream.
            bitrate (optional, float): [1/s] override data rate of the bitstream
            samplerate (optional, float): [1/s] override sample rate of the bitstream
            cutoff_frequency (optional, float): [1/s] override cutoff frequency of the bitstream. If None: no lowpass filtering.
            filter_order (optional, int): override filter order to enforce cutoff frequency
            seed (optional, int): override seed used to generate bits (if needed)
            dtype (optional, torch.dtype): override dtype to generate the bits for. None -> "torch.get_default_dtype()"
            device (optional, torch.device): override device to generate the bits on. None -> "cpu"
        """
        bitrate = self.bitrate if bitrate is None else float(bitrate)
        samplerate = self.samplerate if samplerate is None else float(samplerate)
        cutoff_frequency = (
            self.cutoff_frequency
            if cutoff_frequency is None
            else float(cutoff_frequency)
        )
        filter_order = (
            self.filter_order if filter_order is None else int(filter_order + 0.5)
        )
        seed = self.seed if seed is None else int(seed + 0.5)
        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype
        rng = self._rng if seed == self.seed else np.random.RandomState(seed=seed)
        try:
            if len(bits.shape) == 0:
                bits = int(bits.item())
        except:
            pass

        if torch.is_tensor(bits):
            bits = bits.detach().cpu().numpy()

        if isinstance(bits, int):
            bits = rng.rand(bits) > 0.5

        # handle fractional sampling:
        temp_samplerate = max(
            int(8 * cutoff_frequency + 0.5) // int(samplerate + 0.5) * samplerate,
            samplerate,
        )
        rc = int(temp_samplerate + 0.5) // int(samplerate + 0.5)
        rates_gcd = np.gcd(int(temp_samplerate + 0.5), int(bitrate + 0.5))
        rs = int(temp_samplerate + 0.5) // rates_gcd
        rb = int(bitrate + 0.5) // rates_gcd
        stream = np.stack([bits] * rs, 1).reshape(-1, *bits.shape[1:]).copy()

        if cutoff_frequency is not None:
            normal_cutoff = cutoff_frequency / (0.5 * temp_samplerate * rb)
            b, a = butter(
                N=filter_order, Wn=normal_cutoff, btype="lowpass", analog=False
            )
            stream = lfilter(b, a, stream, axis=0)

        stream = torch.tensor(stream[:: rb * rc].copy(), dtype=dtype, device=device)

        return stream


def _broadcast_prediction_target(prediction, target):
    """ broadcast prediction and target in identical shapes

    Args:
        prediction (Tensor): prediction
        target (Tensor): target

    Returns:
        prediction and target in the same shape

    """
    if not torch.is_tensor(prediction):
        prediction = torch.tensor(prediction)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=prediction.dtype, device=prediction.device)
    target = target.to(dtype=prediction.dtype, device=prediction.device)

    while len(prediction.shape) < 4:
        prediction = prediction[:, None]
    while len(target.shape) < 4:
        target = target[:, None]
    try:
        prediction, target = torch.broadcast_tensors(prediction.clone(), target.clone())
    except RuntimeError:
        raise RuntimeError("failed to broadcast target in the same shape as prediction")
    return prediction, target


class _Loss(torch.nn.Module):
    """ Base class for loss function extensions. """

    def __init__(self, latency=0.0, warmup=0, bitrate=40e9, samplerate=160e9):
        """
        Args:
            latency (float): [bits] fractional latency in bit lengths. This value can be a floating point number bigger than 1.
            warmup (int): [bits] integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate (float): [1/s] data rate of the bitstream
            samplerate (float): [1/s] sample rate of the bitstream
        """
        super(_Loss, self).__init__()
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
        """ Calculate loss
        Args:
            prediction (Tensor): prediction power tensor. Should be broadcastable to tensor with shape (# timesteps, # wavelengths, # readouts, # batches)
            target (Tensor): target power tensor. Should be broadcastable to the same shape as prediction.
            latency (optional, float): [bits] override fractional latency in bit lengths. This value can be a floating point number bigger than 1.
            warmup (optional, int): [bits] override integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate (optional, float): [1/s] override data rate of the bitstream
            samplerate (optional, float): [1/s] override sample rate of the bitstream
        """
        raise NotImplementedError(
            "Implement forward function for your loss by subclassing."
        )

    def plot(
        self,
        x,
        latency=None,
        warmup=None,
        bitrate=None,
        samplerate=None,
        unit="ns",
        show=False,
        **kwargs
    ):
        """ Plot prediction and target
        Args:
            x (Tensor): Should be broadcastable to tensor with shape (# timesteps, # wavelengths, # readouts, # batches)
            latency (optional, float): [bits] override fractional latency in bit lengths. This value can be a floating point number bigger than 1.
            warmup (optional, int): [bits] override integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate (optional, float): [1/s] override data rate of the bitstream
            samplerate (optional, float): [1/s] override sample rate of the bitstream
            unit (str): unit to use for time array (time values will be multiplied accordingly)
            show (bool): run plt.show at the end of this method.
            **kwargs: keyword arguments given to plt.plot.
        """
        import matplotlib.pyplot as plt

        bitrate = self.bitrate if bitrate is None else float(bitrate)
        samplerate = self.samplerate if samplerate is None else float(samplerate)
        latency = self.latency if latency is None else float(latency)
        warmup = self.warmup if warmup is None else int(warmup + 0.5)
        x, _ = _broadcast_prediction_target(x, x)
        x = x.view(x.shape[0], -1)

        # delay sequences with warmup and latency
        l = int(latency * samplerate / bitrate + 0.5)  # latency sample points
        w = int(int(warmup + 0.5) * samplerate / bitrate + 0.5)  # warmup sample points
        x = x[w + l : :]

        time = np.arange(x.shape[0]) / samplerate
        units = {"s": 0, "ms": 3, "μs": 6, "ns": 9, "ps": 12, "fs": 15}
        unit = unit.replace("u", "μ")
        if unit not in units:
            unit = "ns"
        factor = 10 ** units[unit]
        p = plt.plot(time * factor, x.data.cpu().numpy(), **kwargs)
        plt.xlabel("t [%s]" % unit)
        return p


class MSELoss(_Loss):
    """ Mean Squared Error for bitstreams """

    def forward(
        self,
        prediction,
        target,
        latency=None,
        warmup=None,
        bitrate=None,
        samplerate=None,
    ):
        bitrate = self.bitrate if bitrate is None else float(bitrate)
        samplerate = self.samplerate if samplerate is None else float(samplerate)
        latency = self.latency if latency is None else float(latency)
        warmup = self.warmup if warmup is None else int(warmup + 0.5)
        prediction, target = _broadcast_prediction_target(prediction, target)

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


class BERLoss(_Loss):
    """ Bit Error Rate (non-differentiable)"""

    def __init__(
        self, threshold=0.5, latency=0.0, warmup=0, bitrate=40e9, samplerate=160e9
    ):
        """
        Args:
            threshold (float): threshold value (where to place the 0/1 threshold)
            latency (float): fractional latency [in bit lengths]. This value can be a floating point number bigger than 1.
            warmup (int): integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate (float): the bit rate of the signal [in Hz]
            samplerate (float): the sample rate of the signal [in Hz]
        """
        super(BERLoss, self).__init__(
            latency=latency, warmup=warmup, bitrate=bitrate, samplerate=samplerate
        )
        self.threshold = float(threshold)

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
        """ Calculate loss
        Args:
            prediction (Tensor): prediction power tensor. Should be broadcastable to tensor with shape (# timesteps, # wavelengths, # readouts, # batches)
            target (Tensor): target power tensor. Should be broadcastable to the same shape as prediction.
            threshold (optional, float): override threshold value (where to place the 0/1 threshold)
            latency (optional, float): [bits] override fractional latency in bit lengths. This value can be a floating point number bigger than 1.
            warmup (optional, int): [bits] override integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate (optional, float): [1/s] override data rate of the bitstream
            samplerate (optional, float): override the sample rate of the signal [in Hz]
        """
        with torch.no_grad():
            threshold = self.threshold if threshold is None else float(threshold)
            bitrate = self.bitrate if bitrate is None else float(bitrate)
            samplerate = self.samplerate if samplerate is None else float(samplerate)
            latency = self.latency if latency is None else float(latency)
            warmup = self.warmup if warmup is None else int(warmup + 0.5)
            prediction, target = _broadcast_prediction_target(prediction, target)

            _, nw, nd, nb = prediction.shape

            # always normalize target around 0 (this way we don't need to care about how the target bits are defined [0,1] or [-1, 1] or smth else)
            target = (target - target.mean(0)) / target.std()

            # handle fractional sampling:
            rb = int(bitrate + 0.5) // np.gcd(int(samplerate + 0.5), int(bitrate + 0.5))
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
