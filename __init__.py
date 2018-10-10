'''
# PhotonTorch

PhotonTorch is a photonic simulation framework based on the deep learning framework PyTorch.

PhotonTorch features CUDA enabled optimization of photonic circuits. It leverages the
deep learning framework PyTorch to view the photonic circuit as essentially a recurrent
neural network. This enables the use of native PyTorch optimizers to optimize the
(physical) parameters of your circuit.
'''

# Test pytorch version
import torch
if torch.__version__.split('.')[0] == '0' and int(torch.__version__.split('.')[1]) < 4:
    raise ImportError('Torch version [%s] is not compatible with'
                      'minimum required version >= 0.4.x'%torch.__version__)

## Submodules
from . import components
from . import constants
from . import environment
from . import networks
from . import torch_ext


## Components

# Component
from .components.component import Component

# Terms
from .components.terms import Term
from .components.terms import Source
from .components.terms import Detector

# Mirrors
from .components.mirrors import Mirror

# SOAs
from .components.soas import Soa
from .components.soas import BaseSoa
from .components.soas import LinearSoa
from .components.soas import AgrawalSoa

# MMIs
from .components.mmis import Mmi
from .components.mmis import PhaseArray

# MZIs
from .components.mzis import Mzi

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
from .networks.rings import RingMolecule
from .networks.rings import RingNetwork

# Two Port Networks
from .networks.twoport import TwoPortNetwork

# Reck Network
from .networks.reck import ReckMxN
from .networks.reck import ReckNxN
from .networks.reck import ReckMmi
from .networks.clements import ClementsNxN


## Environment

from .environment.environment import Environment


## Detectors

from .detectors.photodetector import Photodetector


## PyTorch extensions

# autograd
from .torch_ext.autograd import block_diag

# neural networks
from .torch_ext import nn

# tensor
from .torch_ext.tensor import where
