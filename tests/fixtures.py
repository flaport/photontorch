""" test fixtures """

# pragma: no cover

#############
## Imports ##
#############

import torch
import pytest
import numpy as np

import photontorch as pt


######################
## Useful Functions ##
######################


def default_components():
    excluded = ["Component"]
    for name, cls in pt.components.__dict__.items():
        if name[0] != "_" and name[0] == name[0].upper() and name not in excluded:
            yield cls()


##############
## Fixtures ##
##############


## Environments


@pytest.fixture
def tenv():
    """ default time domain environment """
    return pt.Environment(num_timesteps=7, num_wavelengths=2)


@pytest.fixture
def fenv():
    """ default frequency domain environment """
    return pt.Environment(wavelength=np.linspace(1.5, 1.6, 100), frequency_domain=True)


## Components


@pytest.fixture
def comp():
    """ default base component """
    return pt.Component()


@pytest.fixture
def wg():
    """ default base waveguide """
    return pt.Waveguide()


@pytest.fixture
def s():
    """ default source """
    return pt.Source()


@pytest.fixture
def d():
    """ default detector """
    return pt.Detector()


## Networks


@pytest.fixture
def unw():
    """ default unterminated network """
    with pt.Network() as nw:
        nw.wg1 = nw.wg2 = pt.Waveguide(length=5e-6)
        nw.link(1, '0:wg1:1','0:wg2:1', 0)
    return nw


@pytest.fixture
def nw():
    """ default network (source-waveguide-detector) """
    with pt.Network() as nw:
        nw.wg = pt.Waveguide(length=1e-5)
        nw.s = pt.Source()
        nw.d = pt.Detector()
        nw.link('s:0','0:wg:1','0:d')
    return nw


@pytest.fixture
def twoportnw():
    """ default two-port network with random connection matrix """
    C, _, _ = np.linalg.svd(
        np.random.rand(5, 5) + 1j * np.random.rand(5, 5)
    )  # random unitary matrix
    C[range(5), range(5)] = 0  # no self connections
    nw = pt.TwoPortNetwork(
        twoportcomponents=[pt.Waveguide(length=1e-5, loss=1000) for i in range(5)],
        conn_matrix=C,
        sources_at=[1, 0, 0, 0, 0],
        detectors_at=[0, 0, 0, 0, 1],
    )
    return nw


@pytest.fixture
def rnw():
    """ default ring network """
    return pt.RingNetwork(2, 6).terminate()


@pytest.fixture
def reck():
    """ default reck network """
    return pt.ReckMxN(3, 4).terminate()


@pytest.fixture
def clements():
    """ default reck network """
    return pt.ClementsNxN(4).terminate()


## Detectors


@pytest.fixture
def det():
    """ default detector """
    return pt.Photodetector(bitrate=50e9)


## Connectors


@pytest.fixture
def conn():
    """ default connector """
    wg = pt.Waveguide()
    s = pt.Source()
    d = pt.Detector()
    conn = wg["ab"] * s["a"] * d["b"]
    return conn
