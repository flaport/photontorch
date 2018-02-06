''' Ring Networks '''

#############
## Imports ##
#############

## Other
import torch
import numpy as np

## Relative
from .network import Network
from ..components.terms import Source, Detector, Term
from ..components.connection import Connection
from ..components.component import Component
from ..utils.autograd import block_diag
from ..utils.tensor import where

#####################
## All Pass Filter ##
#####################

class DirectionalCouplerWithLength(Component):
    '''
    A directional coupler with length is a memory-containing component with 4 ports.

    A directional coupler has one trainable parameter: the coupling R.

    Connections
    ------------
    dircoup['ijkl']:
     k        l
      \______/
      /------\
     i        j

    Note
    ----
     * This version of a directional coupler is prefered over a wg-wg-wg-wg-dc network
       becuase it only has 4 ports in stead of 12.
     * This Component is part of the networks module, because it acts as a network.
    '''

    num_ports = 4

    def __init__(self, dircoup, wg, name=None):
        '''
        Directional Coupler initialization

        Parameters
        ----------
        dircoup : DirectionalCoupler instance without length
        wg : gives the full length of the directional coupler
        '''
        Component.__init__(self, name=None)
        self.wg = wg
        self.dircoup = dircoup

    def initialize(self, env):
        ''' Initialize Component '''
        self.wg.initialize(env)
        self.dircoup.initialize(env)

    @property
    def delays(self):
        return torch.cat((self.wg.delays, self.wg.delays))

    @property
    def rS(self):
        ''' real part of the scattering matrix '''
        k = self.dircoup.R**0.5 # coupling
        t = (1-self.dircoup.R)**0.5 # Transmission
        rS_wg_t = self.wg.rS*t
        iS_wg_k = self.wg.iS*k
        rS = self.new_variable([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        rS[:2,:2] = rS_wg_t # Transmission from i < - > j
        rS[2:,2:] = rS_wg_t # Transmission from k < - > l
        rS[::2,::2] = -iS_wg_k
        rS[1::2,1::2] = -iS_wg_k
        return rS

    @property
    def iS(self):
        ''' imag part of the scattering matrix '''
        k = self.dircoup.R**0.5 # coupling
        t = (1-self.dircoup.R)**0.5 # Transmission
        iS_wg_t = self.wg.iS*t
        rS_wg_k = self.wg.rS*k
        iS = self.new_variable([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        iS[:2,:2] = iS_wg_t # Transmission from i < - > j
        iS[2:,2:] = iS_wg_t # Transmission from k < - > l
        iS[::2,::2] = rS_wg_k
        iS[1::2,1::2] = rS_wg_k
        return iS

class DirectionalCouplerNetwork(Network, Component):
    def __init__(self, shape, dircoup=None, terms=None, initialize_randomly='seed', name='dircoupnw'):
        Component.__init__(self, name=name)

        if initialize_randomly:
            if initialize_randomly == 'seed':
                initialize_randomly = 1
            r = np.random.RandomState(seed=initialize_randomly)

        if dircoup is None:
            wg0 = Waveguide(length=10e-6, neff=2.86, name='wg0')
            dc0 = DirectionalCoupler(R=0.5)
            self.dircoup = DirectionalCouplerWithLength(dircoup=dc0, wg=wg0, name='+')
        else:
            self.dircoup = dircoup.copy()
            self.dircoup.name = '+'

        I,J = shape
        if I < 2 or J < 2:
            raise ValueError('2D Network is required.')

        self.components_array = np.zeros((I,J), dtype=object)
        for i in range(I):
            for j in range(J):
                dircoup_copy = self.dircoup.copy()
                if initialize_randomly:
                    dircoup_copy.dircoup.R = r.rand()
                self.components_array[i,j] = dircoup_copy

        if terms is None:
            terms = {}
        num_terms = 8 + 2*(I-2) + 2*(J-2)
        self.term_array = np.zeros(num_terms, dtype=object)
        for i in range(num_terms):
            self.term_array[i] = terms.get('default',Term)(name=str(i))
        for k, v in terms.items():
            if type(k) is int:
                self.term_array[k] = v

    def copy(self):
        new = self.__class__(shape=self.shape, dircoup=self.dircoup, name=self.name)
        new.term_array = self.term_array()
        return new

    def terminate(self, term=None):
        raise NotImplementedError

    @property
    def shape(self):
        return self.components_array.shape

    @property
    def components(self):
        return tuple(self.components_array.ravel()) + tuple(self.term_array)
    @components.setter
    def components(self, value):
        I, J = self.shape
        K = self.term_array.shape[0]
        for i in range(I):
            for j in range(J):
                self.components_array[i,j] = value[i*J+j]
        for k in range(K):
            self.term_array[k] = value[-K+k]

    @property
    def C(self):
        Ns = np.cumsum([0]+[comp.num_ports for comp in self.components])
        C = block_diag(*(comp.C for comp in self.components))
        # add connections
        for i1, j1, i2, j2  in self._parse_connections():
            i = Ns[i1] + j1
            j = Ns[i2] + j2
            C[i,j] = C[j,i] = 1.0

        K = self.term_array.shape[0]
        idxs = where(((C.sum(0)>0) | (C.sum(1)>0)).ne(1).data)
        for k, (idx, term) in enumerate(zip(idxs, self.term_array)):
            C[-K+k,idx] = C[idx,-K+k] = 1.0

        return C

    def _parse_connections(self):
        connections = []
        I,J = self.shape
        for i in range(I):
            for j in range(J):
                if i < I-1:
                    top = (i*J+j,3)
                    bottom = ((i+1)*J+j,1)
                    connections.append(top + bottom)
                if j < J-1:
                    left = (i*J+j,2)
                    right = (i*J+j+1,0)
                    connections.append(left + right)
        return connections

    @staticmethod
    def _parse_string(s):
        s.replace('+','d')
        lines = [_s.strip() for _s in s.lower().splitlines()]
        while lines[0] == '':
            lines = lines[1:]
        while lines[-1] == '':
            lines = lines[:-1]
        array = np.array([list(_s) for _s in lines], dtype=object)
        return array
