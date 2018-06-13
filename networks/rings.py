''' Ring Networks '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .network import Network

from ..environment.environment import Environment
from ..components.terms import Term, Detector, Source
from ..components.waveguides import Waveguide
from ..components.directionalcouplers import DirectionalCoupler
from ..components.connection import Connection

from ..torch_ext.nn import Buffer


#####################
## All Pass Filter ##
#####################

class AllPass(Network):
    r''' All Pass Filter

    An AllPass filter is a memory-containing component with one input and one output.

    Terms:
             ___
            /   \
            \___/
        0-----------1
    '''
    components={
        'dc':DirectionalCoupler(0.5),
        'wg_ring':Waveguide(7e-6, loss=4230, neff=3.47),
        'wg_in':Waveguide(3.5e-6, neff=3.47),
        'wg_pass':Waveguide(3.5e-6, neff=3.47),
        'term_in':Source(),
        'term_pass':Detector(),
    }
    connections=[
        'term_in:0:wg_in:0',
        'wg_in:1:dc:0',
        'dc:1:wg_pass:0',
        'dc:2:wg_ring:0',
        'dc:3:wg_ring:1',
        'wg_pass:1:term_pass:0',
    ]
    def __init__(self, **kwargs):
        '''
        AllPass Filter Initialization

        Args:
            **kwargs: 'dc','wg_ring','wg_in','wg_pass','term_in','term_pass','name'
        '''
        name = kwargs.pop('name', None)
        self.components.update(kwargs)

        if 'term_in' not in self.components:
            self.connections[0] = 'wg_in:0:0'
        if 'term_pass' not in self.components:
            self.connections[-1] = 'wg_pass:1:1'

        super(AllPass, self).__init__(name=name)


#####################
## Add Drop Filter ##
#####################

class AddDrop(Network):
    r''' Add Drop Filter

    An AddDrop filter is a memory-containing component with one input and one output.

    Terms:
        3----===----2
            /   \
            \___/
        0-----------1
    '''
    components = {
        'term_in':Source(),
        'term_pass':Detector(),
        'term_add':Detector(),
        'term_drop':Detector(),
        'dc1':DirectionalCoupler(0.5),
        'dc2':DirectionalCoupler(0.5),
        'wg1':Waveguide(2.5e-5, loss=0, neff=2.86),
        'wg2':Waveguide(2.5e-5, loss=0, neff=2.86),
    }
    connections = [
        'dc1:0:term_in:0',
        'dc1:1:term_pass:0',
        'dc1:2:wg1:0',
        'dc1:3:wg2:0',
        'dc2:0:wg2:1',
        'dc2:1:wg1:1',
        'dc2:2:term_add:0',
        'dc2:3:term_drop:0',
    ]
    def __init__(self, **kwargs):
        '''
        AddDrop Filter Initialization

        Args:
            **kwargs: 'dc1', 'dc2', 'wg1','wg2','term_in','term_pass','term_add','term_drop','name'
        '''
        name = kwargs.pop('name', None)
        self.components.update(kwargs)

        if 'term_in' not in self.components:
            self.connections[0] = 'dc1:0:0'
        if 'term_pass' not in self.components:
            self.connections[-1] = 'dc1:1:1'
        if 'term_add' not in self.components:
            self.connections[0] = 'dc2:2:2'
        if 'term_drop' not in self.components:
            self.connections[-1] = 'dc2:2:3'

        super(AddDrop, self).__init__(name=name)