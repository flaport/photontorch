""" Photontorch network

The Network is the core of Photontorch: it is where everything comes together.

"""

## Networks

from . import network

# Base Network
from .network import Network
from .network import current_network
from .network import link

# Ring Networks
from .rings import RingNetwork

# Reck Network
from .reck import ReckNxN

# Clements Network
from .clements import ClementsNxN
