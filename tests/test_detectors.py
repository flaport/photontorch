""" component tests """

#############
## Imports ##
#############

import torch
import pytest
import photontorch as pt

from fixtures import det


###########
## Tests ##
###########


def test_photodetector_creation(det):
    pass


def test_photodetector_a_parameter(det):
    a = det.a
    assert a in det._buffers.values()
    assert a.shape[0] == det.filter_order + 1


def test_photodetector_forward(det):
    with torch.no_grad():
        num_bits = 10
        bits = torch.rand(num_bits) > 0.5
        det = pt.Photodetector(bitrate=50e9, dt=1.0 / (10 * 50e9))
        timesteps_per_bit = int(1.0 / (det.bitrate * det.dt) + 0.5)
        bitstream = torch.stack([bits] * timesteps_per_bit, -1).flatten().float()
        detected_bitstream = det(bitstream)


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
