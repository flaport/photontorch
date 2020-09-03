""" comp tests """

#############
## Imports ##
#############

import torch
import pytest
import numpy as np
import photontorch as pt

from fixtures import default_components, tenv, comp, wg

###########
## Tests ##
###########


def test_initialization_with_sources_detectors_at_same_port(tenv):
    class WrongTerm(pt.Term):
        def set_sources_at(self, sources_at):
            sources_at[0] = True

        def set_detectors_at(self, detectors_at):
            detectors_at[0] = True

    with pytest.raises(ValueError):
        wt = WrongTerm()
        assert wt.sources_at.any()
        with tenv:
            wt.initialize()


def test_agrawal_soa():
    # fmt: off
    # single delay too many in buffermask:
    # target = np.array([0.00000, 0.00814, 0.10276, 0.04285, 0.02142, 0.12841, 0.03331, 0.04192, 0.14501, 0.02096])
    # delay compensated in buffermask:
    target = np.array([0.00000, 0.01252, 0.10493, 0.03503, 0.02863, 0.12833, 0.02530, 0.05175, 0.14189, 0.01401])
    # fmt: on
    with pt.Network() as nw:
        nw.src = pt.Source()
        nw.soa = pt.AgrawalSoa()
        nw.det = pt.Detector()
        nw.link("src:0", "0:soa:1", "0:det")
    env = pt.Environment(dt=6e-13, num_t=150)
    src = torch.tensor(
        0.3 * np.sin(0.0001 * 2 * np.pi * env.t * env.c / env.wl),
        dtype=torch.get_default_dtype(),
    )[:, None, None, None]
    with env:
        det = nw(src)[:, 0, 0, 0].detach().cpu().numpy()
    np.testing.assert_almost_equal(target, det[::15], decimal=5)


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
