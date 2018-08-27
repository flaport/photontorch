''' comp tests '''

#############
## Imports ##
#############

import torch
import pytest
import numpy as np
import photontorch as pt

from fixtures import tenv, fenv, nw, tnw, fnw

###########
## Tests ##
###########

def test_source(tnw):
    src = tnw.Source(np.random.random(tnw.env.t.shape))

def test_source_with_batch_axis(tnw):
    src = tnw.Source(np.random.random(5), axes=['b'])

def test_source_with_wl_axis_mismatch(fnw):
    with pytest.raises(ValueError):
        src = fnw.Source(np.random.random(5), axes=['w'])

def test_source_with_num_sources_known(tnw):
    src = tnw.Source(np.random.random(tnw.num_sources), axes=['s'])

def test_source_with_cuda_nw(tnw):
    if torch.cuda.is_available(): # pragma: no cover
        src = tnw.cuda().Source(np.random.rand(2), axes=['b'])

def test_source_shape(tnw):
    src = tnw.Source(np.random.random((tnw.env.num_timesteps, 10)), axes=['t','b'])
    desired_shape = (2, tnw.env.num_timesteps, tnw.env.num_wl, tnw.nmc, 10)
    actual_shape = src.shape
    assert desired_shape == actual_shape

def test_source_getitem(tnw):
    src = tnw.ConstantSource(1)
    src[0,1]

def test_constantsource_wrong_input_shape(tnw):
    with pytest.raises(ValueError):
        src = tnw.ConstantSource(np.random.random((3,3,3)))

def test_constantsource_with_input_shape_equal_to_num_sources(tnw):
    src = tnw.ConstantSource(np.random.random(tnw.num_sources))


###############
## Run Tests ##
###############

if __name__ == '__main__': # pragma: no cover
    pytest.main([__file__])
