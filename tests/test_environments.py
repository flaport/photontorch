''' comp tests '''

#############
## Imports ##
#############

import torch
import pytest
from pytest import approx
import numpy as np

import photontorch as pt

from fixtures import tenv, fenv


###########
## Tests ##
###########

def test_tenv(tenv):
    pass

def test_fenv(fenv):
    assert fenv.use_delays == False
    assert fenv.num_timesteps == 1
    assert fenv.t == approx(np.array([fenv.dt]))
    assert fenv.dt == approx(fenv.t_end)

def test_env_with_multiple_wavelengths():
    env = pt.Environment(num_wl=3)

def test_env_with_dwl_specified():
    env = pt.Environment(dwl=1e-7)

def test_env_with_wl_specified():
    env = pt.Environment(wl=1.55e-6)

def test_env_with_wl_specified_but_should_be_wls():
    wls = np.linspace(1.5,1.6,10)
    env = pt.Environment(wl=wls)
    assert wls == approx(env.wls)

def test_env_with_wls_specified_as_wl():
    env = pt.Environment(wls=1.55e-6)

def test_env_with_no_delays():
    env = pt.Environment(use_delays=False)

def test_env_with_cuda():
    if not torch.cuda.is_available(): # pragma: no cover
        pytest.skip('cuda not available. test skipped')
    env = pt.Environment(cuda=True)
    assert env.cuda == True

def test_env_with_cuda_and_no_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda : False)
    with pytest.warns(RuntimeWarning):
        env = pt.Environment(cuda=True)
    assert env.cuda == False

def test_env_with_extra_arguments():
    env = pt.Environment(test_attribute='hello')
    assert env.test_attribute == 'hello'

def test_env_copy():
    env1 = pt.Environment(dt=1e-14)
    env2 = env1.copy(dt=1e-16)
    assert env1 != env2
    for k in env1.__dict__:
        if not isinstance(getattr(env1, k), (str, bool, type(None))):
            assert (getattr(env1, k) is not getattr(env2, k))
    assert env1.dt == approx(1e-14)
    assert env2.dt == approx(1e-16)

def test_env_c(tenv):
    assert isinstance(tenv.c, float)
    assert int(round(tenv.c)) == 299792458

def test_env_wl():
    wls = np.linspace(1.5,1.6,11, endpoint=True)
    env = pt.Environment(wls=wls)
    assert env.wl == approx(np.mean(wls))

def test_env_wl_setter(tenv):
    tenv.wl = 3.0
    assert tenv.wl == approx(3.0)
    assert tenv.wls == approx(np.array([3.0]))

def test_env_dwl_with_single_wavelength(tenv):
    assert tenv.dwl is None

def test_env_dwl_with_multiple_wavelengths(fenv):
    assert fenv.dwl == approx((fenv.wls[1] - fenv.wls[0]))

def test_env_dwl_setter(fenv):
    original_dwl = fenv.dwl
    fenv.dwl = 0.5*original_dwl
    assert fenv.dwl == approx(0.5*original_dwl)

def test_env_wl_start(fenv):
    new_wl_start = np.mean(fenv.wls)
    fenv.wl_start = new_wl_start
    assert fenv.wl_start == approx(new_wl_start)
    assert fenv.wls[0] == approx(new_wl_start)

def test_env_wl_end(fenv):
    new_wl_end = np.mean(fenv.wls)
    fenv.wl_end = new_wl_end
    assert fenv.wl_end == approx(new_wl_end)
    assert fenv.wls[-1] == approx(new_wl_end)

def test_num_wl(fenv):
    fenv.num_wl = 10
    assert fenv.num_wl == 10

def test_set_num_wl_equal_to_one(fenv):
    wl = np.mean(fenv.wls)
    fenv.num_wl = 1
    assert fenv.wl == approx(wl)

def test_t_start(tenv):
    new_t_start = np.mean(tenv.t)
    tenv.t_start = new_t_start
    assert tenv.t_start == approx(new_t_start)
    assert tenv.t[0] == approx(new_t_start)

def test_t_end(tenv):
    new_t_end = np.mean(tenv.t)
    tenv.t_end = new_t_end
    t_end =  (new_t_end//tenv.dt + 1)*tenv.dt
    assert tenv.t_end == approx(t_end)
    assert  tenv.t[-1] == approx(t_end - tenv.dt)

def test_num_timesteps(tenv):
    tenv.num_timesteps = 2*tenv.num_timesteps

def test_no_delays(tenv):
    assert tenv.no_delays is False
    tenv.no_delays = True
    assert tenv.no_delays is True

def test_repr(tenv, fenv):
    assert isinstance(repr(tenv), str)
    assert isinstance(repr(fenv), str)

def test_str(tenv, fenv):
    assert isinstance(str(tenv), str)
    assert isinstance(str(fenv), str)

###############
## Run Tests ##
###############

if __name__ == '__main__': # pragma: no cover
    pytest.main([__file__])
