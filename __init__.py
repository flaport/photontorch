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


## Sources

from .sources.source import BitSource
from .sources.source import TimeSource
from .sources.source import ConstantSource


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


## Tests
from . import tests
