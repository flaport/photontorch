""" component tests """

#############
## Imports ##
#############

import torch
import pytest
import numpy as np
from scipy.signal import lfilter, butter

import photontorch as pt

from fixtures import lpdet


#############
## Helpers ##
#############


def scipy_detect(x, bitrate, samplerate, cutoff_frequency=20e9, filter_order=4):
    normal_cutoff = cutoff_frequency / (0.5 * samplerate)
    b, a = butter(N=filter_order, Wn=normal_cutoff, btype="lowpass", analog=False)
    return lfilter(b, a, x, axis=0)


###########
## Tests ##
###########


def test_lowpass_detector_creation(lpdet):
    pt.LowpassDetector(
        bitrate=40e9, samplerate=160e9, cutoff_frequency=20e9, filter_order=4,
    )


def test_photodetector_creation(lpdet):
    pt.Photodetector(
        bitrate=40e9,
        samplerate=160e9,
        cutoff_frequency=20e9,
        responsivity=1.0,
        dark_current=1e-10,
        load_resistance=1e6,
        filter_order=4,
        seed=9,
    )


def test_lowpass_detector_a_parameter(lpdet):
    a = lpdet.a
    assert a.shape[0] == lpdet.filter_order


def test_lowpass_detector_b_parameter(lpdet):
    b = lpdet.b
    assert b.shape[0] == lpdet.filter_order + 1


def test_lowpass_detector_forward(lpdet):
    num_bits = 24
    num_samples_per_bit = int(lpdet.samplerate / lpdet.bitrate + 0.5)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(23)
        stream = (
            torch.stack(
                [torch.rand(num_bits, generator=gen) > 0.5] * num_samples_per_bit, 1
            )
            .view(-1)
            .to(torch.get_default_dtype())
        )
        detected = lpdet(stream).detach().cpu().numpy()
        detected_scipy = scipy_detect(
            stream.detach().cpu().numpy(),
            bitrate=lpdet.bitrate,
            samplerate=lpdet.samplerate,
            filter_order=lpdet.filter_order,
        )
        assert np.allclose(detected, detected_scipy, atol=1e-6)
        detected2 = lpdet(stream, num_splits=3).detach().cpu().numpy()
        assert np.allclose(detected2, detected_scipy, atol=1e-2)


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
