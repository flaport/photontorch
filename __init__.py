'''
# Photontorch


## Introduction

PhotonTorch is a photonic simulation framework based on the deep learning framework PyTorch.


## Features

PhotonTorch features CUDA enabled optimization of photonic circuits. It leverages the
deep learning framework PyTorch to view the photonic circuit as essentially a recurrent
neural network. This enables the use of native PyTorch optimizers to optimize the
(physical) parameters of your circuit.


## Dependencies

#### Required
* [`numpy`](http://www.numpy.org/)
* [`pytorch`](http://pytorch.org/)

#### Optional
* [`tqdm`](https://pypi.python.org/pypi/tqdm) (for progress bars)
* [`matplotlib`](https://matplotlib.org/) (for visualization)


## Copyright

(c) Floris Laporte

'''

# Useful while in development...
import sys
sys.dont_write_bytecode = True


## Submodules
from . import components
from . import constants
from . import environment
from . import networks
from . import sources
from . import torch_ext
from . import tests


## Components

# Terms
from .components.terms import Term
from .components.terms import Source
from .components.terms import Detector

# Mirrors
from .components.mirrors import Mirror

# Waveguides
from .components.waveguides import Waveguide
from .components.waveguides import Connection

# Grating Couplers
from .components.gratingcouplers import GratingCoupler

# Directional couplers
from .components.directionalcouplers import DirectionalCoupler
from .components.directionalcouplers import RealisticDirectionalCoupler
from .components.directionalcouplers import DirectionalCouplerWithLength


## Networks

# Base Network
from .networks.network import Network

# Ring Networks
from .networks.rings import AllPass
from .networks.rings import AddDrop
from .networks.rings import RingNetwork

# Directional Coupler Networks
from .networks.directionalcouplers import DirectionalCouplerNetwork


## Environment

from .environment.environment import Environment


## PyTorch extensions

# autograd
from .torch_ext.autograd import block_diag
from .torch_ext.autograd import batch_block_diag

# neural networks
from .torch_ext.nn import Module
from .torch_ext.nn import Parameter
from .torch_ext.nn import BoundedParameter

# tensor
from .torch_ext.tensor import zeros
from .torch_ext.tensor import where
from .torch_ext.tensor import is_variable
