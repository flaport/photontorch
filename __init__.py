''' A package using the machine-learning module PyTorch to easily optimize Photonic Circuits '''


## Components

from .components import Term
from .components import Source
from .components import Mirror
from .components import Network
from .components import Detector
from .components import Waveguide
from .components import SlantedMirror


## Useful utils

from .utils import block_diag
