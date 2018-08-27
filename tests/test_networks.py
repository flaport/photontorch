''' networks tests '''

#############
## Imports ##
#############

import torch
import pytest
import numpy as np
import photontorch as pt

from fixtures import unw, nw, tenv, fenv, tnw, fnw, det, wg, twoportnw, um

###########
## Tests ##
###########

def test_network(nw):
    pass

def test_network_with_component_list():
    nw = pt.Network(
        components=[
            pt.Waveguide(name='wg1'),
            pt.Waveguide(name='wg2'),
        ],
        connections=[
            'wg1:1:wg2:1',
        ]
    )

def test_network_defined_in_class():
    class NewNetwork(pt.Network):
        components=[
            pt.Waveguide(name='wg1'),
            pt.Waveguide(name='wg2'),
        ]
        connections=[
            'wg1:1:wg2:1',
        ]
    nw = NewNetwork()

def test_network_with_component_not_defined():
    with pytest.raises(KeyError):
        nw = pt.Network(
            components=[
                pt.Waveguide(name='wg1'),
                pt.Waveguide(name='wg2'),
            ],
            connections=[
                'wg3:1:wg2:1',
            ]
        )

def test_terminate(unw):
    unw.terminate()
    unw.terminate([pt.Source('src'), pt.Detector('det')])
    if torch.cuda.is_available(): # pragma: no cover
        unw.cuda()
        unw.terminate()

def test_cuda(tnw):
    if torch.cuda.is_available(): # pragma: no cover
        tnw.cuda()
        assert tnw.is_cuda

def test_cpu(tnw):
    if torch.cuda.is_available(): # pragma: no cover
        tnw.cuda().cpu()
        assert not tnw.is_cuda

def test_unterminate(unw):
    nw2 = unw.terminate().unterminate()

def test_termination_on_terminated_network(nw):
    with pytest.raises(IndexError):
        nw.terminate()

def test_reinitialize(tnw):
    tnw.initialized = False
    tnw.initialize()
    assert tnw.initialized

def test_initialize_on_unterminated_network(unw, tenv):
    unw.initialize(tenv)
    assert not unw.initialized

def test_initializion_with_too_big_simulation_timestep(nw, tenv):
    with pytest.warns(RuntimeWarning):
        nw.initialize(tenv.copy(dt=1))

def test_forward_with_uninitialized_network(nw):
    with pytest.raises(RuntimeError):
        nw(source=1)

def test_forward_with_constant_source(tnw):
    tnw(source=1)

def test_forward_with_timesource(tnw):
    tnw(np.random.rand(tnw.env.num_timesteps))

def test_forward_with_different_value_for_each_source(tnw):
    tnw(np.random.rand(tnw.num_sources))

def test_forward_with_batch_weights(tnw):
    tnw(np.random.rand(tnw.env.num_wl, tnw.env.num_wl, tnw.num_sources, 3))

def test_forward_with_power_false(tnw):
    tnw(1, power=False)

def test_forwar_with_detector(tnw, det):
    tnw(1, detector=det)

def test_network_connection_with_equal_ports(wg):
    with pytest.raises(IndexError):
        nw = pt.Network(
            components={
                'wg1':wg,
                'wg2':wg,
            },
            connections=[
                'wg1:1:wg1:1',
            ]
        )

def test_network_connection_with_too_high_port_index(wg):
    with pytest.raises(ValueError):
        nw = pt.Network(
            components={
                'wg1':wg,
                'wg2':wg,
            },
            connections=[
                'wg1:1:wg2:2',
            ]
        )

def test_twoportnetwork(twoportnw):
    pass

def test_twoportnetwork_with_delays():
    C,_,_ = np.linalg.svd(np.random.rand(5,5) + 1j*np.random.rand(5,5)) # random unitary matrix
    C[range(5),range(5)] = 0 # no self connections
    nw = pt.TwoPortNetwork(
        twoportcomponents=[pt.Waveguide(loss=1000) for i in range(5)],
        conn_matrix=C,
        sources_at = [1,0,0,0,0],
        detectors_at = [0,0,0,0,1],
        delays=np.random.rand(5),
    )

def test_twoportnetwork_termination(twoportnw):
    with pytest.raises(RuntimeError):
        twoportnw.terminate()
    with pytest.raises(RuntimeError):
        twoportnw.unterminate()

def test_twoportnetwork_initialiation(twoportnw, tenv):
    twoportnw.initialize(tenv)

def test_unitary_matrix_network(um):
    pass

def test_unitary_matrix_network_with_m_bigger_than_n():
    with pytest.raises(ValueError):
        um = pt.UnitaryMatrixNetwork(
            shape=(4,2),
            dc=pt.DirectionalCoupler(coupling=0.5),
            wg=pt.Waveguide(length=1e-5),
        )

def test_unitary_matrix_network_with_wrong_dc():
    with pytest.raises(TypeError):
        um = pt.UnitaryMatrixNetwork(
            shape=(2,4),
            dc=pt.Waveguide(),
            wg=pt.Waveguide(length=1e-5),
        )

def test_unitary_matrix_network_with_wrong_wg():
    with pytest.raises(TypeError):
        um = pt.UnitaryMatrixNetwork(
            shape=(2,4),
            dc=pt.DirectionalCoupler(coupling=0.5),
            wg=pt.DirectionalCoupler(coupling=0.5),
        )

def test_unitary_matrix_network_termination(um):
    um.unterminate() # should do nothing
    tum = um.terminate()
    tum.terminate() # should do nothing
    tum.unterminate()

def test_network_plot(tenv, fenv):
    tnw = pt.AddDrop(
        term_in=pt.Source(name='in'),
        term_pass=pt.Detector(name='pass'),
        term_add=pt.Detector(name='add'),
        term_drop=pt.Detector(name='drop')
    ).initialize(tenv.copy(wls=np.array([1.5,1.55,1.6])*1e-6))

    # test time mode 0
    with pytest.raises(ValueError):
        detected = torch.tensor(np.random.rand(5), dtype=torch.float32)
        tnw.plot(detected)

    # test time mode 1
    detected = torch.tensor(np.random.rand(tnw.env.num_timesteps), dtype=torch.float32)
    tnw.plot(detected)

    # test time mode 2
    detected = np.random.rand(tnw.env.num_timesteps, tnw.num_detectors)
    tnw.plot(detected)

    # test time mode 3
    detected = np.random.rand(tnw.env.num_timesteps, tnw.env.num_wl)
    tnw.plot(detected)

    # test time mode 4
    detected = np.random.rand(tnw.env.num_timesteps, tnw.env.num_wl, tnw.num_detectors)
    tnw.plot(detected)

    # test time mode 5
    detected = np.random.rand(tnw.env.num_timesteps, tnw.env.num_wl, 11)
    tnw.plot(detected)

    # test time mode 6
    detected = np.random.rand(tnw.env.num_timesteps, tnw.num_detectors, 11) # this one is not covered?
    tnw.plot(detected)

    # test time mode 6
    detected = np.random.rand(tnw.env.num_timesteps, tnw.env.num_wl, tnw.num_detectors, 11)
    tnw.plot(detected)

    # test time mode 7
    with pytest.raises(RuntimeError):
        detected = np.random.rand(tnw.env.num_timesteps, tnw.env.num_wl, tnw.num_detectors, 11, 2)
        tnw.plot(detected)

    # test wl mode 1
    detected = np.random.rand(tnw.env.num_wl)
    tnw.plot(detected)

    # test wl mode 2
    detected = np.random.rand(tnw.env.num_wl, tnw.num_detectors)
    tnw.plot(detected)

    # test wl mode 3
    detected = np.random.rand(tnw.env.num_wl, 11)
    tnw.plot(detected)

    # test wl mode 4
    detected = np.random.rand(tnw.env.num_wl, tnw.num_detectors, 11)
    tnw.plot(detected)

    # test wl mode 5
    with pytest.raises(RuntimeError):
        detected = np.random.rand(tnw.env.num_wl, tnw.num_detectors, 11, 2)
        tnw.plot(detected)




###############
## Run Tests ##
###############

if __name__ == '__main__': # pragma: no cover
    pytest.main([__file__])
