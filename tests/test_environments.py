""" comp tests """

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


def test_tenv_creation(tenv):
    pass


def test_fenv_creation(fenv):
    assert fenv.freqdomain == True
    assert fenv.num_t == 1


def test_env_with_multiple_wavelengths_creation():
    env = pt.Environment(num_wl=3)


def test_env_with_wl_specified_creation():
    env = pt.Environment(wl=1.55e-6)


def test_env_with_no_delays_creation():
    env = pt.Environment(freqdomain=True)


def test_env_with_extra_arguments_creation():
    env = pt.Environment(test_attribute="hello")
    assert env.test_attribute == "hello"


def test_env_copy():
    env1 = pt.Environment(dt=1e-14)
    env2 = env1.copy(dt=1e-16)
    assert env1 is not env2
    assert env1.dt == approx(1e-14)
    assert env2.dt == approx(1e-16)


def test_env_c(tenv):
    assert isinstance(tenv.c, float)
    assert int(round(tenv.c)) == 299792458


def test_repr(tenv, fenv):
    assert isinstance(repr(tenv), str)
    assert isinstance(repr(fenv), str)


def test_str(tenv, fenv):
    assert isinstance(str(tenv), str)
    assert isinstance(str(fenv), str)


def test_environment_with_many_wavelengths():
    env = pt.Environment(wl0=1500e-9, wl1=1600e-9, num_wl=10000)
    assert env.num_wl == 10000
    assert env.wl0 == pytest.approx(1500e-9)
    assert env.wl1 == pytest.approx(1600e-9)


def test_environment_with_many_frequencies():
    env = pt.Environment(f0=200e12, f1=198e12, num_wl=10000)
    assert env.num_f == 10000
    assert env.f0 == pytest.approx(200e12)
    assert env.f1 == pytest.approx(198e12)


def test_environment_with_many_timesteps():
    env = pt.Environment(t0=0, t1=1e-9, dt=1e-13, f=198e12)
    assert env.num_t == 10000
    assert env.t0 == pytest.approx(0)
    assert env.t1 == pytest.approx(1e-9)


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
