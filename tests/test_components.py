""" comp tests """

#############
## Imports ##
#############

import torch
import pytest
import photontorch as pt

from fixtures import default_components, tenv, comp, wg

###########
## Tests ##
###########


def test_component_name(comp):
    assert comp.name == "component"


def test_component_repr(comp):
    assert comp.__repr__() == "component"


def test_component_str(comp):
    assert comp.__str__() == "component"


@pytest.mark.parametrize("comp", default_components())
def test_component_initialization(comp, tenv):
    x = comp.initialize(tenv)
    comp.delays
    comp.S
    assert x is not None


def test_initialization_with_sources_detectors_at_same_port(tenv):
    class WrongTerm(pt.Term):
        def get_sources_at(self):
            return torch.ones(1, dtype=torch.uint8, device=self.device)

        def get_detectors_at(self):
            return torch.ones(1, dtype=torch.uint8, device=self.device)

    with pytest.raises(ValueError):
        wt = WrongTerm()
        wt.initialize(tenv)


def test_initialization_with_device_cuda(wg, tenv):
    if not torch.cuda.is_available():  # pragma: no cover
        pytest.skip("cannot perform cuda-required tests on a pc without cuda.")
    wg = wg.initialize(tenv.copy(device="cuda"))
    assert wg.is_cuda


def test_initialization_with_device_cpu(wg, tenv):
    wg.device = type("device", (object,), {"type": "dummy"})
    wg = wg.initialize(tenv.copy(device="cpu"))
    assert not wg.is_cuda


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
