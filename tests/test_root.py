""" root tests """

#############
## Imports ##
#############

import torch
import pytest
import photontorch

try:
    reload
except NameError:
    from importlib import reload


###########
## Tests ##
###########


def test_wrong_pytorch_version(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "0.3.1")
    with pytest.raises(ImportError):
        reload(photontorch)


def test_import():
    reload(photontorch)
    assert True


###############
## Run Tests ##
###############

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
