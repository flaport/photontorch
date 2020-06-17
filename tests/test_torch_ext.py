""" torch_ext tests """

#############
## Imports ##
#############

from copy import deepcopy

import torch
import pytest
import numpy as np
import photontorch as pt

from fixtures import gen, nw as mod

###########
## Tests ##
###########

## Autograd
def test_block_diag(gen):
    t1 = torch.rand((2, 2), requires_grad=True, generator=gen)
    t2 = torch.rand((3, 3), requires_grad=True, generator=gen)
    t = pt.block_diag(t1, t2)
    loss = ((t - torch.ones_like(t)) ** 2).sum()
    loss.backward()


## Neural Networks


def test_module_creation(mod):
    pass


def test_module_to(mod):
    mod.to(torch.float64)


def test_module_cuda(mod):
    if not torch.cuda.is_available():  # pragma: no cover
        pytest.skip("cannot convert to cuda on a pc without cuda")
    mod.cuda()
    mod.cuda(0)
    assert mod.is_cuda


def test_bounded_parameter_with_no_data():
    bp = pt.BoundedParameter()
    with pytest.raises(RuntimeError):
        bool(bp)


def test_bounded_parameter_with_data_out_of_bounds():
    with pytest.raises(ValueError):
        bp = pt.BoundedParameter(data=torch.tensor([0.5, 1.5]), bounds=(0, 1))


def test_bounded_parameter_with_wrong_bounds():
    with pytest.raises(ValueError):
        bp = pt.BoundedParameter(data=torch.tensor([0.5]), bounds=(1, 0))


def test_bounded_parameter_with_no_tensor_data():
    with pytest.raises(TypeError):
        bp = pt.BoundedParameter(data=[0.5])


def test_bounded_parameter_with_too_many_bounds():
    with pytest.raises(ValueError):
        bp = pt.BoundedParameter(data=torch.tensor(0.6), bounds=np.array([0, 1, 2]))


def test_bounded_parameter_repr():
    s = repr(pt.BoundedParameter(data=torch.tensor([0.5])))
    assert s.startswith("BoundedParameter")


def test_bounded_parameter_deepcopy():
    bp = pt.BoundedParameter(data=torch.tensor([0.5]))
    bp2 = deepcopy(bp)
    assert bp2 is not bp
    assert bp.__class__ is bp2.__class__


def test_buffer_with_no_data():
    bp = pt.Buffer()
    with pytest.raises(RuntimeError):
        bool(bp)


def test_buffer_repr():
    s = repr(pt.Buffer(data=torch.tensor([0.5])))
    assert s.startswith("Buffer")


def test_ber():
    berfunc = pt.BERLoss(bitrate=50e9, samplerate=160e9)  # uneven sample rate
    streamgenerator = pt.BitStreamGenerator(bitrate=50e9, samplerate=160e9)
    output_bits = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1])
    target_bits = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1])  # one bit difference
    output_stream = streamgenerator(output_bits)
    target_stream = streamgenerator(target_bits)
    ber = berfunc(output_stream, target_stream)
    assert ber == 0.1


def test_mse():
    msefunc = pt.MSELoss(bitrate=50e9, samplerate=160e9)  # uneven sample rate
    streamgenerator = pt.BitStreamGenerator(bitrate=50e9, samplerate=160e9)
    output_bits = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1])
    target_bits = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1])  # one bit difference
    output_stream = streamgenerator(output_bits).requires_grad_()
    target_stream = streamgenerator(target_bits)
    mse = msefunc(output_stream, target_stream)
    assert np.allclose(mse.item(), 0.09375)
    assert mse.requires_grad


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
