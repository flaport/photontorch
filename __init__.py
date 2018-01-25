''' A package using the machine-learning module PyTorch to easily optimize Photonic Circuits '''

# Useful while in development...
import sys
sys.dont_write_bytecode = True


## Components

from .components import Term
from .components import Source
from .components import Mirror
from .components import Detector
from .components import Waveguide
from .components import SlantedMirror
from .components import GratingCoupler
from .components import DirectionalCoupler


## Network

from .network import Network


## Environment

from .environment import Environment


## Useful utils

from .utils import block_diag
