''' Ring Networks '''

#############
## Imports ##
#############

## Standard Library
from copy import copy
from collections import OrderedDict

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
from ..torch_ext.nn import Buffer, Parameter
from ..torch_ext import block_diag, batch_block_diag


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
    def __init__(self, **kwargs):
        '''
        AllPass Filter Initialization

        Args:
            **kwargs: 'dc','wg_ring','wg_in','wg_pass','term_in','term_pass','name'
        '''
        name = kwargs.pop('name', None)
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
        components.update(kwargs)

        if 'term_in' not in components:
            connections[0] = 'wg_in:0:0'
        if 'term_pass' not in components:
            connections[-1] = 'wg_pass:1:1'

        super(AllPass, self).__init__(components, connections, name=name)


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
    def __init__(self, **kwargs):
        '''
        AddDrop Filter Initialization

        Args:
            **kwargs: 'dc1', 'dc2', 'wg1','wg2','term_in','term_pass','term_add','term_drop','name'
        '''
        name = kwargs.pop('name', None)
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
        components.update(kwargs)

        if 'term_in' not in components:
            connections[0] = 'dc1:0:0'
        if 'term_pass' not in components:
            connections[-1] = 'dc1:1:1'
        if 'term_add' not in components:
            connections[0] = 'dc2:2:2'
        if 'term_drop' not in components:
            connections[-1] = 'dc2:2:3'

        super(AddDrop, self).__init__(components, connections, name=name)


####################
## Ring Molecules ##
####################

class RingAtom(Network):
    ''' A Ring Atom is a part of a ring Molecule and should not be used seperately '''
    def __init__(self, wg, num_segments=4, name=None):
        self.num_segments=num_segments
        self.num_ports=2*num_segments
        segment=Waveguide(
            length=wg.length/num_segments,
            loss=wg.loss,
            neff=wg.neff,
            ng=wg.ng,
            phase=None if wg.phase is None else wg.phase.detach().cpu().numpy()/num_segments,
            trainable=wg.trainable,
            name='segment',
        )
        components = OrderedDict([('segment%i'%i,segment) for i in range(num_segments)])
        connections = []
        for i in range(num_segments):
            connections += ['segment%i:1:segment%i:0'%(i, (i+1)%num_segments)]

        super(RingAtom, self).__init__(
            components=components,
            connections=connections,
            name=name,
        )

class RingMolecule(Network):
    ''' A Ring Molecule is a network of rings.'''
    def __init__(self, map, rings, coupling=None, type='square', trainable=True, copy_rings=False, name=None):
        ''' ring Molecule __init__

        Args:
            map: a string map of the rings, [e.g. '@oOx@']. This map can contain multiple
                lines. Each character of the map corresponds to a different kind of ring.
                Special characters are '.' [signifying an empty space in the map] and
                '@' [signifying an outward connection in the map].
            rings: a dictionary containing the different ring types. The keys should be
                the characters used in the map, and the values should be waveguides.
                e.g. rings = {'a': Waveguide(...), 'b': Waveguide(...)}
            coupling: The coupling between the different rings. A single value will default
                to same coupling everywhere, while a dictionary with two-character strings
                as keys signifies the coupling between those ring types.
                e.g. coupling = {'aa':0.5, 'ab':0.4, 'bb':0.2}
            type: The lattice type of the map. For now, only square lattices are supported
            trainable (bool): If the couplings in the RingMolecule are trainable
            copy_rings (bool): If the rings are copied, then the phase of each ring can be
                trained seperately. Obviously, this uses more RAM.
            name: name of the ring molecule.
        '''
        num_segments = {'square':4}[type]
        self.type = type

        rings=OrderedDict((k,RingAtom(wg, num_segments=num_segments)) for k, wg in rings.items())

        if '.' in rings:
            raise ValueError('The "." character is reserved for empty spaces in the network')
        if '@' in rings:
            raise ValueError('The "@" character is reserved for an input/output in the network')

        self.map = np.array([list(line.strip()) for line in map.strip().splitlines()], str)
        self.m, self.n = m,n = self.map.shape

        self.compmap = np.zeros((m,n), object)
        self.compmap.fill(None)

        self.idxmap = np.zeros((m,n), int)

        idx = 0
        self.components = OrderedDict()
        for i in range(m):
            for j in range(n):
                ring = rings.get(self.map[i,j], None)
                if ring is not None:
                    idx += ring.num_ports
                    if copy_rings:
                        ring = ring.copy() # deep copy
                    else:
                        ring = copy(ring) # shallow copy
                    ring.name = 'ring_%i_%i'%(i,j)
                    self.components[ring.name] = ring
                    self.compmap[i,j] = ring
                    self.idxmap[i,j] = idx - ring.num_ports

        if not isinstance(coupling, dict):
            standard_coupling=coupling if coupling is not None else 0.5
            coupling={}
        else:
            standard_coupling = coupling.pop('standard', 0.5)

        self.standard_coupling = standard_coupling

        self.trainable = trainable
        parameter = Parameter if self.trainable else Buffer
        self.couplings = {k:parameter(torch.tensor(v)) for k,v in coupling.items()}

        super(RingMolecule, self).__init__(
            components=self.components,
            connections=None,
            name=name,
        )

        for k, v in self.couplings.items():
            setattr(self, 'coupling_%s'%k, v)


    def initialize(self, env=None):
        if self.trainable:
            self.C = self.get_C()
        super(RingMolecule, self).initialize(env)

    def get_coupling(self, type1, type2):
        return self.couplings.get(type1+type2,
                                  self.couplings.get(type2+type1,
                                                     self.standard_coupling))

    def get_used_components(self):
        return self.components.keys()

    def get_order(self):
        return slice(None)

    def get_C(self):
        rC = block_diag(*(comp.C[0] for comp in self.components.values()))
        iC = block_diag(*(comp.C[1] for comp in self.components.values()))
        C = torch.stack([rC,iC])

        if self.type == 'square':
            for i in range(self.m):
                for j in range(self.n-1):
                    r0 = self.compmap[i,j]
                    r1 = self.compmap[i,j+1]
                    c0 = self.map[i,j]
                    c1 = self.map[i,j+1]
                    idx0 = self.idxmap[i,j]
                    idx1 = self.idxmap[i,j+1]
                    if c0 == '@' and r1 is not None:
                        C[0,idx1+0,idx1+7]=C[0,idx1+7,idx1+0]=0
                    elif r0 is not None and c1 == '@':
                        C[0,idx0+3,idx0+4]=C[0,idx0+4,idx0+3]=0
                    elif r0 is not None and r1 is not None:
                        coupling = self.get_coupling(self.map[i,j], self.map[i,j+1])
                        transmission = 1-coupling
                        k = coupling**0.5
                        t = transmission**0.5
                        C[0,idx0+3,idx0+4]=C[0,idx0+4,idx0+3]=t
                        C[0,idx1+0,idx1+7]=C[0,idx1+7,idx1+0]=t
                        C[1,idx0+3,idx1+7]=C[1,idx1+7,idx0+3]=k
                        C[1,idx0+4,idx1+0]=C[1,idx1+0,idx0+4]=k

            for i in range(self.m-1):
                for j in range(self.n):
                    r0 = self.compmap[i,j]
                    r1 = self.compmap[i+1,j]
                    c0 = self.map[i,j]
                    c1 = self.map[i+1,j]
                    idx0 = self.idxmap[i,j]
                    idx1 = self.idxmap[i+1,j]
                    if c0 == '@' and r1 is not None:
                        C[0,idx1+1,idx1+2]=C[0,idx1+2,idx1+1]=0
                    elif r0 is not None and c1 == '@':
                        C[0,idx0+5,idx0+6]=C[0,idx0+6,idx0+5]=0
                    if r0 is not None and r1 is not None:
                        coupling = self.get_coupling(self.map[i,j], self.map[i+1,j])
                        transmission = 1-coupling
                        k = coupling**0.5
                        t = transmission**0.5
                        C[0,idx0+5,idx0+6]=C[0,idx0+6,idx0+5]=t
                        C[0,idx1+1,idx1+2]=C[0,idx1+2,idx1+1]=t
                        C[1,idx0+5,idx1+1]=C[1,idx1+1,idx0+5]=k
                        C[1,idx0+6,idx1+2]=C[1,idx1+2,idx0+6]=k

        if self.type == 'hexagonal':
            raise NotImplementedError

        return C
