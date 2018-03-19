'''
# PhotonTorch Networks

The Network is the core of Photontorch. This is where everything comes together.

## Creation of a network

There are three accepted forms to create a network:

### First option:
```
    nw = Network(string, *components)
```
s is a string specifying how the components are connected. It follows
the einstein summation convention.

#### Example:
```
        nw = Network('ij,jklm,mn', wg1, dc, wg2)
```
makes a connection between two waveguides and a directional coupler.
The connection is made where equal indices occur:
    last port of wg1 is connected to first port of dc
    last port of dc is connected to first port of wg2.

### Second option
```
    nw = Network(
        (comp_1, s_1),
        (comp_2, s_2),
        ...
        (comp_n, s_n),
    )
```
Network initialization with a list of tuples of length two, each consisting of
a component and its connection string. The connection strings also follow the
einstein summation convention.

#### Example
```
        nw = Network(
            (wg1, 'ij'),
            (dc, 'jklm'),
            (wg3, 'mn')
        )
```

### Third Option
```
    nw = Network(comp_1[s_1]*comp_2[s_2]*...*comp_n[s_n])
```
Network initialization with a product of indexed components (connectors).
#### Example
```
        nw = Network(wg1['ij']*dc['jklm']*wg2['mn'])
```

'''

## Networks

from . import network

# Base Network
from .network import Network

# Array Network
from .array import ArrayNetwork

# Ring Networks
from .rings import AllPass
from .rings import AddDrop
from .rings import RingNetwork

# Directional Coupler Networks
from .directionalcouplers import DirectionalCouplerNetwork


## Modify docstring for documentation:
__doc__ = __doc__ + network.__doc__.replace('#','##')