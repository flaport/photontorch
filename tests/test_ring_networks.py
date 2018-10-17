""" ring networks tests """

#############
## Imports ##
#############

import torch
import pytest
import numpy as np

import photontorch as pt
from photontorch.networks.rings import RingAtom

from fixtures import unw, nw, tenv, fenv, tnw, fnw, det

###########
## Tests ##
###########


def test_allpass():
    ap = pt.AllPass()


def test_adddrop():
    ad = pt.AddDrop()


def test_ringatom():
    ra = RingAtom(pt.Waveguide(), 4)


def test_ringmolecule():
    rm = pt.RingMolecule(
        "@oo.\n.oo@", rings={"o": pt.Waveguide()}, coupling={"oo": 0.3, "standard": 0.5}
    )


def test_ringmolecule_with_copy_rings():
    rm = pt.RingMolecule(
        "@oo@", rings={"o": pt.Waveguide()}, coupling=0.2, copy_rings=True
    )


def test_ringmolecule_with_wrong_ring_symbols2():
    with pytest.raises(ValueError):
        rm = pt.RingMolecule(".@oo@.", rings={".": pt.Waveguide()}, coupling=0.2)


def test_ringmolecule_with_wrong_ring_symbols1():
    with pytest.raises(ValueError):
        rm = pt.RingMolecule(".@oo@.", rings={"@": pt.Waveguide()}, coupling=0.2)


def test_ringmolecule_initialize(tenv):
    rm = pt.RingMolecule(
        "@oo@", rings={"o": pt.Waveguide()}, coupling={"oo": 0.3, "standard": 0.5}
    )
    rm.initialize(tenv)


def test_ringmolecule_wrong_type():
    with pytest.raises(ValueError):
        rm = pt.RingMolecule(
            ".@oo@.", rings={".": pt.Waveguide()}, coupling=0.2, type="wrongtype"
        )


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
