''' test fixtures '''

#pragma: no cover

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
    excluded = ['Component']
    for name, cls in pt.components.__dict__.items():
        if name[0] != '_' and name[0] == name[0].upper() and name not in excluded:
            yield cls()


##############
## Fixtures ##
##############


## Environments

@pytest.fixture
def tenv():
    ''' default time domain environment '''
    return pt.Environment(num_timesteps=7)

@pytest.fixture
def fenv():
    ''' default frequency domain environment '''
    return pt.Environment(wls=np.linspace(1.5,1.6,100), use_delays=False)


## Components

@pytest.fixture
def comp():
    ''' default base component '''
    return pt.Component()

@pytest.fixture
def wg():
    ''' default base waveguide '''
    return pt.Waveguide()

@pytest.fixture
def s():
    ''' default source '''
    return pt.Source()

@pytest.fixture
def d():
    ''' default detector '''
    return pt.Detector()


## Networks

@pytest.fixture
def unw():
    ''' default unterminated network '''
    wg = pt.Waveguide(length=5e-6)
    return pt.Network(wg['ab']*wg['bc'])

@pytest.fixture
def nw():
    ''' default network (source-waveguide-detector) '''
    wg = pt.Waveguide(length=1e-5)
    s = pt.Source()
    d = pt.Detector()
    nw = pt.Network(wg['ab']*s['a']*d['b'])
    return nw

@pytest.fixture
def tnw():
    ''' default time initialized network (source-waveguide-detector) '''
    wg = pt.Waveguide(length=1e-5)
    s = pt.Source()
    d = pt.Detector()
    nw = pt.Network(wg['ab']*s['a']*d['b'])
    return nw.initialize(pt.Environment(num_timesteps=7))

@pytest.fixture
def fnw():
    ''' default frequency initialized network (source-waveguide-detector) '''
    wg = pt.Waveguide(length=1e-5)
    s = pt.Source()
    d = pt.Detector()
    nw = pt.Network(wg['ab']*s['a']*d['b'])
    return nw.initialize(pt.Environment(wls=np.linspace(1.5,1.6,100), use_delays=False))

@pytest.fixture
def twoportnw():
    ''' default two-port network with random connection matrix '''
    C,_,_ = np.linalg.svd(np.random.rand(5,5) + 1j*np.random.rand(5,5)) # random unitary matrix
    C[range(5),range(5)] = 0 # no self connections
    nw = pt.TwoPortNetwork(
        twoportcomponents=[pt.Waveguide(length=1e-5, loss=1000) for i in range(5)],
        conn_matrix=C,
        sources_at = [1,0,0,0,0],
        detectors_at = [0,0,0,0,1],
    )
    return nw

@pytest.fixture
def um():
    ''' default unitary matrix network '''
    um = pt.UnitaryMatrixNetwork(
        shape=(2,2),
        dc=pt.DirectionalCoupler(coupling=0.5),
        wg=pt.Waveguide(length=1e-5),
    )
    return um


## Detectors

@pytest.fixture
def det():
    ''' default detector '''
    return pt.Photodetector(bitrate=50e9)


## Connectors

@pytest.fixture
def conn():
    ''' default connector '''
    wg = pt.Waveguide()
    s = pt.Source()
    d = pt.Detector()
    conn = wg['ab']*s['a']*d['b']
    return conn
