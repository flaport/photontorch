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


def test_tenv(tenv):
    pass


def test_fenv(fenv):
    assert fenv.frequency_domain == True
    assert fenv.num_timesteps == 1


def test_env_with_multiple_wavelengths():
    env = pt.Environment(num_wavlengths=3)


def test_env_with_wl_specified():
    env = pt.Environment(wavelength=1.55e-6)


def test_env_with_no_delays():
    env = pt.Environment(use_delays=False)


def test_env_with_extra_arguments():
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


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
