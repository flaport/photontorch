''' Ring Networks '''

#############
## Imports ##
#############

# Torch
import torch
from torch.nn import Parameter

## Other
import numpy as np
from collections import OrderedDict

## Relative
from .network import Network
from ..components.terms import Term
from ..components.component import Component
from ..components.connection import Connection
from ..components.directionalcouplers import DirectionalCouplerWithLength
from ..torch_ext.nn import Buffer
from ..torch_ext.autograd import block_diag
from ..torch_ext.tensor import where


#################################
## Directional Coupler Network ##
#################################

class DirectionalCouplerNetwork(Network, Component):
    ''' A network of directional couplers.

    A directional coupler with normal port ordering is repeated periodically in a grid,
    with the ports connected by waveguides:

    Network:
         0    1    2    3
        ..   ..   ..   ..
         1    1    1    1
    11: 0 2--0 2--0 2--0 2 : 4
         3    3    3    3
         |    |    |    |
         1    1    1    1
    10: 0 2--0 2--0 2--0 2 : 5
         3    3    3    3
        ..   ..   ..   ..
         9    8    7    6

    Legend:
        0: -> term locations
        1
       0 2 -> directional coupler
        3
        -- or | -> waveguides

    Note:
        Because of the connection order of the directional coupler (0<->1 and 2<->3), this
        network does not contain any loops and can thus be used as a weight matrix.
    '''
    def __init__(self,
                 shape,
                 dc,
                 wg,
                 couplings=None,
                 lengths=None,
                 phases=None,
                 terms=None,
                 name='dcnw'):
        ''' DirectionalCouplerNetwork

        Args:
            shape: (tuple): shape of the network.
            dc (DirectionalCoupler) : master directional coupler. All the directional
                couplers will take over the properties of this directional coupler,
                except when explicitly overwritten.
            wg (Waveguide): a waveguide containing all the properties of a single
                directional coupler arm, such as the length and the effective index.
            couplings=None (np.ndarray): Desired couplings for the grating couplers. This
                array should have the same length as the shape of the network. If None,
                all directional couplers default to the coupling of the master directional
                coupler.
            lengths=None (np.ndarray): Lengths of the directional couplers.
                If None all lengths will be equal to the length of the master
                directional coupler.
            phases=None (np.ndarray): optional. Phases introduced by the directional
                couplers. If None all phases will be coupled to the length of the
                direcional coupler.
            terms (dict): A dictionary with source and Detector locations in the form
                terms = {3:Source(), 15:Detector(), ...}. All not specified port locations
                will get a Connection as term, as to make it possible to connect to
                other networks (while keeping the port order).

        Note:
            The directional coupler and the waveguide will be combined in a
            DirectionalCouplerWithLength. This conversion happens internally and is not
            required up front.

        '''

        Component.__init__(self, name=name)

        # Add all the possible sources to this network:
        self._inject_sources()

        # Handle shape of network
        I, J = shape
        if I < 2 or J < 2:
            raise ValueError('2D Network is required.')

        # Get directional coupler with length
        master_wg = wg.copy()
        master_wg.name = 'wg'
        master_dc = dc.copy()
        master_dc.name = 'dc'
        self.dc = DirectionalCouplerWithLength(
            dc=master_dc,
            wg=master_wg,
            name='dcwl',
        )

        # Create directional coupler array
        self.dc_array = np.zeros((I, J), dtype=object)
        for i in range(I):
            for j in range(J):
                # Create copy of directional coupler:
                dc_copy = self.dc.copy()
                dc_copy.name = 'dc'
                # Put copy in dc array
                self.dc_array[i, j] = dc_copy

        # Override couplings
        if couplings is not None:
            self.couplings = couplings

        # Override lengths
        if lengths is not None:
            self.lengths = lengths

        # Override phases
        if phases is not None:
            self.phases = phases

        # Create term dictionary
        if terms is None:
            terms = {}
        self.num_terms = 8 + 2*(I-2) + 2*(J-2)
        self.terms = OrderedDict(terms)
        for t in range(self.num_terms):
            if t not in self.terms:
                self.terms[t] = Connection(name='t'+str(t))

        # save order of terms (to reorder in terms clockwise direction):
        self._order = Buffer(torch.from_numpy(np.hstack((
            np.arange(1, J+2, 1), # North row
            np.arange(J+3, 4+(J-2)+2*(J-2), 2), # East column
            [8+2*(I-2)+2*(J-2)-2, 8+2*(I-2)+2*(J-2)-1], # South-East Corner
            np.arange(4+(J-2)+2*(I-2), 8+2*(I-2)+2*(J-2)-2, 1)[::-1], #South Row
            np.arange(J+2, 4+(J-2)+2*(J-2), 2)[::-1], # left column
            [0], # half of north west corner
        ))).long())

        # PyTorch requires submodules to be registered as attributes
        # For the parameters to be found by autograd:
        self._register_components()

        # Other things to finish off:
        self.initialized = False

    def terminate(self, term=None):
        '''
        Terminate open conections with the term of your choice

        Args (Term): Which term to use. Defaults to Term.
        '''
        if term is None:
            term = Term()
        if self.is_cuda:
            term = term.cuda()
        for k, t in list(self.terms.items()):
            if isinstance(t, Connection):
                term = term.copy()
                term.name = t.name
                self.terms[k] = term
                setattr(self, term.name, term)
        return self

    @property
    def couplings(self):
        ''' Get array with couplings of the directional couplers '''
        I, J = self.dc_array.shape
        couplings = self.new_variable(np.zeros((I, J)))
        for i in range(I):
            for j in range(J):
                couplings.data[i, j] = self.dc_array[i, j].dc.kappa2.data[0]
        return couplings

    @couplings.setter
    def couplings(self, array):
        ''' Set all couplings of the network '''
        I, J = self.dc_array.shape
        for i in range(I):
            for j in range(J):
                self.dc_array[i,j].dc.kappa2.data = self.new_tensor([array[i,j]])

    @property
    def lengths(self):
        ''' Get array with couplings of the directional couplers '''
        I, J = self.dc_array.shape
        lengths = self.new_variable(np.zeros((I, J)))
        for i in range(I):
            for j in range(J):
                lengths.data[i, j] = self.dc_array[i, j].wg.length.data[0]
        return lengths

    @lengths.setter
    def lengths(self, array):
        ''' Set all lengths of the network '''
        I, J = self.dc_array.shape
        for i in range(I):
            for j in range(J):
                self.dc_array[i,j].wg.length.data = self.new_tensor([array[i,j]], dtype='double')

    @property
    def phases(self):
        ''' Get all the phases introduced by the waveguides
            Note, this function returns None if the master waveguide hase phase==None.
        '''
        I, J = self.dc_array.shape
        if self.dc.wg.phase is None:
            return None
        phases = self.new_variable(np.zeros((I, J)))
        for i in range(I):
            for j in range(J):
                phases.data[i, j] = self.dc_array[i, j].wg.phase.data[0]
        return phases

    @phases.setter
    def phases(self, array):
        ''' Set the phases of the waveguides
            Note: this only works if the master waveguide had phase!=None
        '''
        I, J = self.dc_array.shape
        if self.dc.wg.phase is None:
            raise AttributeError('Cannot set the phases of the waveguide manually, since '
                                 'they are coupled to the length')
        for i in range(I):
            for j in range(J):
                self.dc_array[i,j].wg.phase.data = self.new_tensor([array[i,j]])


    @property
    def C(self):
        ''' Combined Connection matrix of all the components in the network

        Returns:
            torch.FloatTensors with only 1's and 0's.

        '''
        Ns = np.cumsum([0]+[comp.num_ports for comp in self.components])
        free_idxs = [comp.free_idxs for comp in self.components]
        C = block_diag(*(comp.C for comp in self.components))

        # add connections
        for i1, j1, i2, j2  in self._parse_connections():
            idxs1 = free_idxs[i1]
            idxs2 = free_idxs[i2]
            i = Ns[i1] + idxs1[j1]
            j = Ns[i2] + idxs2[j2]
            C[i, j] = C[j, i] = 1.0

        # find and reorder term connection locations
        K = self.num_terms
        idxs = where(((C.sum(0) > 0) | (C.sum(1) > 0)).ne(1).data)[:K][self._order.data]

        # connect terms
        for k, i in enumerate(self.terms):
            idx = idxs[i] # index of the dc free port
            t_idx = Ns[-K+k-1] + free_idxs[-K+k][0] # index of the term (single) free port
            C[t_idx, idx] = C[idx, t_idx] = 1.0

        return C

    def _parse_connections(self):
        connections = []
        I, J = self.shape
        for i in range(I):
            for j in range(J):
                if i < I-1:
                    top = (i*J+j, 3)
                    bottom = ((i+1)*J+j, 1)
                    connections.append(top + bottom)
                if j < J-1:
                    left = (i*J+j, 2)
                    right = (i*J+j+1, 0)
                    connections.append(left + right)
        return connections

    @property
    def shape(self):
        ''' Get shape of directional coupler network '''
        return self.dc_array.shape

    @property
    def components(self):
        ''' Get all components of the directional coupler network as a list '''
        return tuple(self.dc_array.ravel()) + tuple(self.terms.values())

    def copy(self, couplings=None, lengths=None):
        ''' Copy the directional coupler network '''
        if couplings is None:
            couplings = self.couplings.data.cpu().numpy()
        if lengths is None:
            lengths = self.lengths.data.cpu().numpy()
        new = self.__class__(
            shape=self.shape,
            dc=self.dc.dc,
            wg=self.dc.wg,
            couplings=couplings,
            lengths=lengths,
            name=self.name,
        )
        new.terms = self.terms
        if self.initialized:
            new.initialize(self.env.copy())
        return new
