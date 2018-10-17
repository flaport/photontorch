""" torch_ext tests """

#############
## Imports ##
#############

from copy import deepcopy

import torch
import pytest
import numpy as np
import photontorch as pt

from fixtures import nw as mod

###########
## Tests ##
###########

## Autograd
def test_block_diag():
    t1 = torch.rand((2, 2), requires_grad=True)
    t2 = torch.rand((3, 3), requires_grad=True)
    t = pt.torch_ext.autograd.block_diag(t1, t2)
    loss = ((t - torch.ones_like(t)) ** 2).sum()
    loss.backward()


## Neural Networks


def test_module(mod):
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
    bp = pt.nn.BoundedParameter()
    with pytest.raises(RuntimeError):
        bool(bp)


def test_bounded_parameter_with_data_out_of_bounds():
    with pytest.raises(ValueError):
        bp = pt.nn.BoundedParameter(data=torch.tensor([0.5, 1.5]), bounds=(0, 1))


def test_bounded_parameter_with_wrong_bounds():
    with pytest.raises(ValueError):
        bp = pt.nn.BoundedParameter(data=torch.tensor([0.5]), bounds=(1, 0))


def test_bounded_parameter_with_no_tensor_data():
    with pytest.raises(TypeError):
        bp = pt.nn.BoundedParameter(data=[0.5])


def test_bounded_parameter_with_too_many_bounds():
    with pytest.raises(ValueError):
        bp = pt.nn.BoundedParameter(data=torch.tensor(0.6), bounds=np.array([0, 1, 2]))


def test_bounded_parameter_repr():
    s = repr(pt.nn.BoundedParameter(data=torch.tensor([0.5])))
    assert s.startswith("BoundedParameter")


def test_bounded_parameter_deepcopy():
    bp = pt.nn.BoundedParameter(data=torch.tensor([0.5]))
    bp2 = deepcopy(bp)
    assert bp2 is not bp
    assert bp.__class__ is bp2.__class__


def test_buffer_with_no_data():
    bp = pt.nn.Buffer()
    with pytest.raises(RuntimeError):
        bool(bp)


def test_buffer_repr():
    s = repr(pt.nn.Buffer(data=torch.tensor([0.5])))
    assert s.startswith("Buffer")


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
