""" neural network (nn) extensions """


#############
## Imports ##
#############

## Standard Library
from copy import deepcopy

## Torch
import torch
from torch.nn import Parameter
from torch.nn import Module as _Module_

## 3rd party
import numpy as np
from scipy.signal import butter, lfilter

## Relative
from ..environment.environment import current_environment


############
## Buffer ##
############


class Buffer(torch.Tensor):
    """ A Buffer is a Module Variable which is automatically registered in _buffers

    Each Module has an OrderedDict named _buffers. In this Dictionary, all
    model related parameters that do not require optimization are stored.

    The Buffer class makes it easier to register a buffer in the Module. If an
    attribute of a module is set to a Buffer, it will automatically be added to
    the _buffers attribute.

    Note:
        For the automatic registration of the Buffer to work, you need to use
        the `photontorch.nn.Module`, which is a subclass of `torch.nn.Module`.

    """

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __repr__(self):
        return "Buffer containing:\n" + super(Buffer, self).__repr__()


#######################
## Bounded Parameter ##
#######################


class BoundedParameter(torch.nn.Parameter):
    """ A BoundedParameter is special Parameter that is bounded between a range.

    Under the hood it registers an unbounded weight in our
    photontorch.nn.Module and a class property calculating the desired
    parameter value on the fly by performing a scaled sigmoid

    Note:
        For the registration of the BoundedParameter to work, you need to use
        the `photontorch.nn.Module`, which is a subclass of `torch.nn.Module`.

    """

    def __new__(cls, data=None, bounds=(0, 1), requires_grad=True):
        if data is None:
            data = torch.Tensor()

        if not torch.is_tensor(data):
            raise TypeError(
                "argument 'data' must be Tensor, not %s" % type(data).__name__
            )

        # check bounds
        try:
            a, b = bounds
        except ValueError:
            raise ValueError("bounds should be a tuple with length 2")
        if b <= a:
            raise ValueError(
                "bounds should be a tuple with length 2 and with bounds[0] < bounds[1]"
            )
        if (data < a).any() or (data > b).any():
            raise ValueError(
                "some of your data is outside the specified bounds: [%i, %i]" % (a, b)
            )
        new = torch.Tensor._make_subclass(
            cls, cls._inverse_sigmoid(data.data, (a, b)), requires_grad
        )
        new.bounds = (float(a), float(b))
        return new

    @staticmethod
    def _sigmoid(weights, bounds):
        a, b = bounds
        scaled_data = torch.sigmoid(weights)
        data = (b - a) * scaled_data + a
        return data

    @staticmethod
    def _inverse_sigmoid(data, bounds):
        a, b = bounds
        scaled_data = (data - a) / (b - a)
        weights = -torch.log(1 / scaled_data - 1)
        return weights

    def __repr__(self):
        tensor_repr = torch.Tensor.__repr__(self._sigmoid(self.data, self.bounds))
        return (
            "BoundedParameter in [%.2f, %.2f] representing:\n" % self.bounds
            + tensor_repr
        )

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = torch.Tensor._make_subclass(cls, self.data.clone(), self.requires_grad)
        new.bounds = deepcopy(self.bounds, memo)
        return new


######################
## Module Extension ##
######################


class Module(_Module_):
    """ Torch.nn Module extension with some extra features. """

    def __init__(self):
        super(Module, self).__init__()
        self.device = torch.device(type="cpu")

    def __setattr__(self, attr, value):
        # Check for buffers and if the value is a buffer, register it.
        if isinstance(value, Buffer):
            self.register_buffer(attr, value)
        if isinstance(value, BoundedParameter):
            # register the unbounded weight of the bounded parameter:
            _attr = "_" + attr
            self.register_parameter(_attr, value)

            # register the class property returning the bounded value
            value_property = property(
                lambda self: self._parameters[_attr]._sigmoid(
                    self._parameters[_attr], self._parameters[_attr].bounds
                )
            )
            # register the property in a subclass of the original class
            # and give the subclass the same name as the original module:
            self.__class__ = type(  # <-- special way of making a subclass on the fly
                self.__class__.__name__, (self.__class__,), {attr: value_property}
            )
        else:
            super(Module, self).__setattr__(attr, value)

    @property
    def is_cuda(self):
        """ check if the model parameters live on the GPU """
        return False if self.device.type == "cpu" else True

    def to(self, *args, **kwargs):
        """ move the module to a device (cpu or cuda) """
        new = super(Module, self).to(*args, **kwargs)
        for k, v in self._modules.items():
            self._modules[k] = v.to(*args, **kwargs)
        try:  # torch < 1.5.1
            new.device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        except ValueError:  # torch >= 1.5.1
            new.device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        return new

    def cpu(self):
        """ Transform the Module to live on the CPU """
        return self.to(device="cpu")

    def cuda(self, device=None):
        """ Transform the Module to live on the GPU

        Args:
            device (int): index of the GPU device.
        """
        if device is None:
            device = "cuda:0"
        elif isinstance(device, int):
            device = "cuda:%i" % device
        return self.to(device=device)


###############
## Functions ##
###############


class BitStreamGenerator(Module):
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

        Note:
            Although the causality of using negative latencies is questionable,
            they *are* allowed. However, each (fractional) negative latency
            should be compensated with an (integer) number of warmup bits
            (rounded up) to make it work.

        """
        super(BitStreamGenerator, self).__init__()
        self.bitrate = float(bitrate)
        self.samplerate = float(samplerate)
        self.cutoff_frequency = (
            None if cutoff_frequency is None else float(cutoff_frequency)
        )
        self.filter_order = None if filter_order is None else int(filter_order + 0.5)
        self.seed = None if seed is None else int(seed + 0.5)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        self._rng = np.random.RandomState(seed=self.seed)

    def forward(
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
            bitrate (optional, float): [1/s] override data rate of the bitstream (defaults to bitrate found in environment)
            samplerate (optional, float): [1/s] override the sample rate of the signal (defaults to samplerate found in environment)
            cutoff_frequency (optional, float): [1/s] override cutoff frequency of the bitstream. If None: no lowpass filtering.
            filter_order (optional, int): override filter order to enforce cutoff frequency
            seed (optional, int): override seed used to generate bits (if needed)
            dtype (optional, torch.dtype): override dtype to generate the bits for. None -> "torch.get_default_dtype()"
            device (optional, torch.device): override device to generate the bits on. None -> "cpu"

        Note:
            If a bitrate and/or samplerate can be found in the current
            environment, those values will be regarded as keyword arguments and
            hence get precedence over the values given during the
            BitStreamGenerator initialization.

        Note:
            Although the causality of using negative latencies is questionable,
            they *are* allowed. However, each (fractional) negative latency
            should be compensated with an (integer) number of warmup bits
            (rounded up) to make it work.
        """

        try:
            env = current_environment()
            bitrate = env.bitrate if bitrate is None else bitrate
            samplerate = env.samplerate if samplerate is None else samplerate
        except RuntimeError:
            pass

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

        with torch.no_grad():
            rc = 1
            temp_samplerate = samplerate
            if cutoff_frequency is not None:
                # handle fractional sampling:
                temp_samplerate = max(
                    int(8 * cutoff_frequency + 0.5)
                    // int(samplerate + 0.5)
                    * samplerate,
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


class _Loss(Module):
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
            bitrate (optional, float): [1/s] override data rate of the bitstream (defaults to bitrate found in environment)
            samplerate (optional, float): [1/s] override the sample rate of the signal (defaults to samplerate found in environment)

        Note:
            If a bitrate and/or samplerate can be found in the current environment,
            those values will be regarded as keyword arguments and hence get precedence over
            the values given during the loss initialization.

        Note:
            Although the causality of using negative latencies is questionable, they *are* allowed. However, each (fractional) negative latency should be compensated with an (integer) number of warmup bits (rounded up) to make it work.
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
            bitrate (optional, float): [1/s] override data rate of the bitstream (defaults to bitrate found in environment)
            samplerate (optional, float): [1/s] override sample rate of the bitstream
            unit (str): unit to use for time array (time values will be multiplied accordingly)
            show (bool): run plt.show at the end of this method.
            **kwargs: keyword arguments given to plt.plot.

        Note:
            If a bitrate and/or samplerate can be found in the current environment,
            those values will be regarded as keyword arguments and hence get precedence over
            the values given during the loss initialization.

        Note:
            Although the causality of using negative latencies is questionable, they *are* allowed. However, each (fractional) negative latency should be compensated with an (integer) number of warmup bits (rounded up) to make it work.
        """

        try:
            env = current_environment()
            bitrate = env.bitrate if bitrate is None else bitrate
            samplerate = env.samplerate if samplerate is None else samplerate
        except RuntimeError:
            pass

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
        if w < 0:
            raise ValueError(
                "warmup should be a positive integer (number of warmup bits)"
            )
        if w + l < 0:
            raise ValueError(
                "please add more warmup bits for negative latency %.2f" % latency
            )
        x = x[w + l : :]

        time = np.arange(x.shape[0]) / samplerate
        units = {"s": 0, "ms": 3, "us": 6, "ns": 9, "ps": 12, "fs": 15}
        if unit not in units:
            raise ValueError(
                "Invalid unit '%s'. Valid units are: %s"
                % (unit, str([unit for unit in units]))
            )
        unit = unit.replace("us", r"$\mu$s")
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

        try:
            env = current_environment()
            bitrate = env.bitrate if bitrate is None else bitrate
            samplerate = env.samplerate if samplerate is None else samplerate
        except RuntimeError:
            pass

        bitrate = self.bitrate if bitrate is None else float(bitrate)
        samplerate = self.samplerate if samplerate is None else float(samplerate)
        latency = self.latency if latency is None else float(latency)
        warmup = self.warmup if warmup is None else int(warmup + 0.5)
        prediction, target = _broadcast_prediction_target(prediction, target)

        # delay sequences with warmup and latency
        l = int(latency * samplerate / bitrate + 0.5)  # latency sample points
        w = int(int(warmup + 0.5) * samplerate / bitrate + 0.5)  # warmup sample points
        if w < 0:
            raise ValueError(
                "warmup should be a positive integer (number of warmup bits)"
            )
        if w + l < 0:
            raise ValueError(
                "please add more warmup bits for negative latency %.2f" % latency
            )
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
            bitrate (optional, float): [1/s] override data rate of the bitstream (defaults to bitrate found in environment)
            samplerate (optional, float): [1/s] override the sample rate of the signal (defaults to samplerate found in environment)

        Note:
            If a bitrate and/or samplerate can be found in the current environment,
            those values will be regarded as keyword arguments and hence get precedence over
            the values given during the loss initialization.

        Note:
            Although the causality of using negative latencies is questionable, they *are* allowed. However, each (fractional) negative latency should be compensated with an (integer) number of warmup bits (rounded up) to make it work.
        """

        try:
            env = current_environment()
            bitrate = env.bitrate if bitrate is None else bitrate
            samplerate = env.samplerate if samplerate is None else samplerate
        except RuntimeError:
            pass

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
            if w < 0:
                raise ValueError(
                    "warmup should be a positive integer (number of warmup bits)"
                )
            if w + s // 2 + l < 0:
                raise ValueError(
                    "please add more warmup bits for negative latency %.2f" % latency
                )
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
