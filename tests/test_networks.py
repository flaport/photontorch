""" networks tests """

#############
## Imports ##
#############

import torch
import pytest
import numpy as np
import photontorch as pt

from fixtures import gen, unw, nw, tenv, fenv, lpdet, wg, rnw, reck, clements

###########
## Tests ##
###########


def test_network_creation(nw):
    pass


def test_frequency_initialization(nw, fenv):
    with fenv:
        nw.initialize()


def test_network_defined_in_class_creation():
    class NewNetwork(pt.Network):
        def __init__(self):
            super(NewNetwork, self).__init__()
            self.wg1 = pt.Waveguide()
            self.wg2 = pt.Waveguide()
            self.link("wg1:1", "0:wg2")

    nw = NewNetwork()


def test_network_with_component_not_defined_creation():
    class NewNetwork(pt.Network):
        def __init__(self):
            super(NewNetwork, self).__init__()
            self.wg1 = pt.Waveguide()
            self.wg2 = pt.Waveguide()
            self.link("wg3:1", "0:wg2")

    with pytest.raises(KeyError):
        nw = NewNetwork()


def test_network_with_component_with_different_name_as_attribute():
    with pytest.raises(ValueError):
        with pt.Network() as nw:
            nw.wg = pt.Waveguide(name="wg0")


def test_ringnetwork_creation(rnw):
    pass


def test_recknetwork_creation(reck):
    pass


def test_clementsnetwork_creation(clements):
    pass


def test_termination(unw):
    unw.terminate()
    unw.terminate([pt.Source("src"), pt.Detector("det")])
    if torch.cuda.is_available():  # pragma: no cover
        unw.cuda()
        unw.terminate()


def test_untermination(unw):
    nw2 = unw.terminate().unterminate()


def test_termination_on_terminated_network(nw):
    with pytest.raises(IndexError):
        nw.terminate()


def test_cuda(nw):
    if torch.cuda.is_available():  # pragma: no cover
        nw.cuda()
        assert nw.is_cuda


def test_cpu(nw):
    if torch.cuda.is_available():  # pragma: no cover
        nw.cuda().cpu()
        assert not nw.is_cuda


def test_reinitialize(nw, tenv):
    with tenv:
        nw.initialized = False  # fake the fact that the network is uninitialized
        nw.initialize()
        assert nw.initialized


def test_initialize_on_unterminated_network(unw, tenv):
    with tenv:
        unw.initialize()
    assert not unw.initialized


def test_initializion_with_too_big_simulation_timestep(nw, tenv):
    with pytest.warns(RuntimeWarning):
        with tenv.copy(dt=1):
            nw.initialize()


def test_forward_with_uninitialized_network(nw):
    with pytest.raises(RuntimeError):
        nw(source=1)


def test_forward_with_constant_source(nw, tenv):
    with tenv:
        nw(source=1)


def test_forward_with_timesource(gen, nw, tenv):
    with tenv:
        nw(torch.rand(tenv.num_timesteps, generator=gen).rename("t"))


def test_forward_with_different_value_for_each_source(gen, nw, tenv):
    with tenv:
        nw.initialize()
        nw(torch.rand(nw.num_sources, generator=gen).rename("s"))


def test_forward_with_batch_weights(gen, nw, tenv):
    with tenv:
        nw.initialize()
        nw(
            torch.rand(
                tenv.num_timesteps, tenv.num_wavelengths, nw.num_sources, 3
            )
        )


def test_forward_with_power_false(nw, tenv):
    with tenv:
        nw(1, power=False)


def test_forward_with_detector(nw, tenv, lpdet):
    with tenv:
        nw(1, detector=lpdet)


def test_network_connection_with_equal_ports(wg):
    with pytest.raises(IndexError):
        nw = pt.Network(components={"wg1": wg, "wg2": wg}, connections=["wg1:1:wg1:1"])


def test_network_connection_with_too_high_port_index(wg):
    with pytest.raises(ValueError):
        nw = pt.Network(components={"wg1": wg, "wg2": wg}, connections=["wg1:1:wg2:2"])


def test_network_plot(gen, tenv, fenv):
    class AddDrop(pt.Network):
        def __init__(self):
            super(AddDrop, self).__init__()
            self.term_in = pt.Source()
            self.term_pass = pt.Detector()
            self.term_add = pt.Detector()
            self.term_drop = pt.Detector()
            self.dc1 = pt.DirectionalCoupler()
            self.dc2 = pt.DirectionalCoupler()
            self.wg1 = pt.Waveguide()
            self.wg2 = pt.Waveguide()
            self.link("term_in:0", "0:dc1:2", "0:wg1:1", "1:dc2:3", "0:term_drop")
            self.link("term_add:0", "2:dc2:0", "0:wg2:1", "3:dc1:1", "0:term_pass")

    tnw = AddDrop()

    with tenv.copy(wls=np.array([1.5, 1.55, 1.6]) * 1e-6) as env:
        tnw.initialize()

        # test time mode 1
        detected = torch.rand(env.num_timesteps, generator=gen)
        tnw.plot(detected)

        # test time mode 0
        with pytest.raises(ValueError):
            detected = torch.rand(5, dtype=torch.float32, generator=gen)
            tnw.plot(detected)

        # test time mode 2
        detected = torch.rand(env.num_timesteps, tnw.num_detectors, generator=gen)
        tnw.plot(detected)

        # test time mode 3
        detected = torch.rand(env.num_timesteps, env.num_wavelengths, generator=gen)
        tnw.plot(detected)

        # test time mode 4
        detected = torch.rand(
            env.num_timesteps, env.num_wavelengths, tnw.num_detectors, generator=gen
        )
        tnw.plot(detected)

        # test time mode 5
        detected = torch.rand(env.num_timesteps, env.num_wavelengths, 11, generator=gen)
        tnw.plot(detected)

        # test time mode 6
        detected = torch.rand(
            env.num_timesteps, tnw.num_detectors, 11, generator=gen,
        )  # this one is not covered?
        tnw.plot(detected)

        # test time mode 6
        detected = torch.rand(
            env.num_timesteps, env.num_wavelengths, tnw.num_detectors, 11, generator=gen,
        )
        tnw.plot(detected)

        # test time mode 7
        with pytest.raises(RuntimeError):
            detected = torch.rand(
                env.num_timesteps, env.num_wavelengths, tnw.num_detectors, 11, 2, generator=gen
            )
            tnw.plot(detected)

        # test wl mode 1
        detected = torch.rand(env.num_wavelengths, generator=gen)
        tnw.plot(detected)

        # test wl mode 2
        detected = torch.rand(env.num_wavelengths, tnw.num_detectors, generator=gen)
        tnw.plot(detected)

        # test wl mode 3
        detected = torch.rand(env.num_wavelengths, 11, generator=gen)
        tnw.plot(detected)

        # test wl mode 4
        detected = torch.rand(env.num_wavelengths, tnw.num_detectors, 11, generator=gen)
        tnw.plot(detected)

        # test wl mode 5
        with pytest.raises(RuntimeError):
            detected = torch.rand(env.num_wavelengths, tnw.num_detectors, 11, 2, generator=gen)
            tnw.plot(detected)


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
