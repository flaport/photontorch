''' A package using the machine-learning module PyTorch to easily optimize Photonic Circuits '''

# Useful while in development...
import sys
sys.dont_write_bytecode = True


## Components

# Terms
from .components.terms import Term
from .components.terms import Source
from .components.terms import Detector

# Mirrors
from .components.mirrors import Mirror
from .components.mirrors import SlantedMirror

# Waveguides
from .components.waveguides import Waveguide
from .components.waveguides import Connection

# Grating Couplers
from .components.gratingcouplers import GratingCoupler

# Directional couplers
from .components.directionalcouplers import DirectionalCoupler


## Networks

# Base Network
from .networks.network import Network

# Ring Networks
from .networks.rings import AllPass
from .networks.rings import AddDrop
from .networks.rings import RingNetwork

# Directional Coupler Networks
from .networks.directionalcouplers import DirectionalCouplerNetwork
from .networks.directionalcouplers import DirectionalCouplerWithLength


## Environment

from .environment.environment import Environment


## Useful utils

# autograd
from .utils.autograd import block_diag

# tensor
from .utils.tensor import zeros
from .utils.tensor import where

# non-autograd functions
from .utils.functions import inv_sigmoid