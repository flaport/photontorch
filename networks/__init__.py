"""
# PhotonTorch Networks

The Network is the core of Photontorch. This is where everything comes together.

## Creation of a network

There are two accepted forms to create a network:

### First option (for small networks):
```
    nw = Network(comp_1[s_1]*comp_2[s_2]*...*comp_n[s_n])
```
Network initialization with a product of indexed components (connectors), where the
string indices follow the einstein summation convention.

#### Example
```
        nw = Network(wg1['ij']*dc['jklm']*wg2['mn'])
```

#### Example:
```
        nw = Network('ij,jklm,mn', wg1, dc, wg2)
```
makes a connection between two waveguides and a directional coupler.
The connection is made where equal indices occur:
    last port of wg1 is connected to first port of dc
    last port of dc is connected to first port of wg2.

### Second option (for bigger networks):
```
    nw = Network(
        components={
            'name1':comp1,
            'name2':comp2,
            ...
            'nameN':compN,
        },
        connections=[
            'name1:0:compN:1,
            'comp2:5:comp1:3,
            ...
            'comp1:2:comp2:2,
        ]
    )
```
Where each network is defined by a dictionary containing all the components and a list
of component connections, where each connection entry hase the form
```
    'first_component_name:first_component_port_index:second_component_name:second_component_port_index'
```

#### Example
```
    allpass = Network(
        components={
            wg_in = pt.Waveguide(),
            wg_out= pt.Waveguide(),
            wg_ring=pt.Waveguide(),
            dc=pt.DirectionalCoupler(),
        },
        connections=[
            'wg_in:1:dc:0',
            'dc:1:wg_out:0',
            'dc:2:wg_ring:0',
            'wg_ring:1:dc:3',
        ]
    )
```

"""

## Networks

from . import network

# Base Network
from .network import Network

# Ring Networks
from .rings import AllPass
from .rings import AddDrop
from .rings import RingMolecule
from .rings import RingNetwork

# Two Port Networks
from .twoport import TwoPortNetwork

# Reck Network
from .reck import ReckMxN
from .reck import ReckNxN
from .reck import ReckMmi

from .clements import ClementsNxN
