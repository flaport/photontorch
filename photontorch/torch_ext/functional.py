"""
# Photontorch Functional

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


class MSELoss(torch.nn.Module):
    """ Mean Squared Error for bitstreams """

    def __init__(self, latency=0.0, warmup=0, bitrate=40e9, samplerate=160e9):
        """
        Args:
            latency = 0.5: fractional latency [in bit lengths]. This value can be a floating point number bigger than 1.
            warmup = 0: integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate = 40e9: the bit rate of the signal [in Hz]
            samplerate = 160e9: the sample rate of the signal [in Hz]
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
            prediction: torch.Tensor: 4D output power tensor with shape (# timesteps, # wavelengths, # readouts, # batches)
            target: torch.Tensor: target power tensor. Should be broadcastable to the same shape as prediction.
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
            threshold = 0.5: where to place the 0 / 1 threshold on the output power.
            latency = 0.5: fractional latency [in bit lengths]. This value can be a floating point number bigger than 1.
            warmup = 0: integer number of warmup bits. warmup bits are disregarded during the loss calculation.
            bitrate = 40e9: the bit rate of the signal [in Hz]
            samplerate = 160e9: the sample rate of the signal [in Hz]
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
            prediction: torch.Tensor: 4D output power tensor with shape (# timesteps, # wavelengths, # readouts, # batches)
            target: torch.Tensor: target power tensor. Should be broadcastable to the same shape as prediction.
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
