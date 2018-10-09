#############
## Imports ##
#############

# other
import numpy as np

# relative
from .network import Network
from ..components.mzis import Mzi
from ..components.mmis import PhaseArray
from ..components.terms import Source, Detector


#############
## Classes ##
#############

class _MixingPhaseArray(Network):
    ''' Helper Class for ClementsNxN '''
    def __init__(self, phases, length=1e-5, ng=2.86, trainable=True, name=None):
        N = phases.shape[0]
        num_mzis = N//2
        components = {}
        components['pa'] = PhaseArray(
            phases=phases,
            length=length,
            ng=ng,
            trainable=trainable,
        )

        for i in range(num_mzis):
            components['mzi%i'%i] = Mzi(
                length=0,
                phi=float(2*np.pi*np.random.rand()),
                theta=float(2*np.pi*np.random.rand()),
                trainable=trainable,
            )

        connections = []
        for i in range(num_mzis):
            connections += ['mzi%i:1:pa:%i'%(i, 2*i)]
            connections += ['mzi%i:2:pa:%i'%(i, 2*i+1)]

        # input connections:
        for i in range(0, num_mzis):
            connections += ['mzi%i:0:%i'%(i, 2*i)]
            connections += ['mzi%i:3:%i'%(i, 2*i+1)]
        if N%2:
            connections += ['pa:%i:%i'%(N-1, N-1)]

        super(_MixingPhaseArray, self).__init__(components, connections, name=name)

class _Capacity2ClemensNxN(Network):
    r''' Helper Class for ClementsNxN

        <- cap==2 ->
        0__  ______0
           \/
        1__/\__  __1
               \/
        2__  __/\__2
           \/
        3__/\______3

    '''
    def __init__(self, N=2, name=None):
        num_mzis = N-1

        # define components
        components = {}
        for i in range(num_mzis):
            components['mzi%i'%i] = Mzi(
                length=0,
                phi=float(2*np.pi*np.random.rand()),
                theta=float(2*np.pi*np.random.rand()),
                trainable=True,
            )

        # connections between mzis:
        connections = []
        for i in range(1, num_mzis-1, 2):
            connections += ['mzi%i:2:mzi%i:0'%((i-1),i)]
            connections += ['mzi%i:3:mzi%i:1'%(i,(i+1))]
        if num_mzis > 1 and N%2:
            connections += ['mzi%i:2:mzi%i:0'%(num_mzis-2, num_mzis-1)]

        # input connections:
        for i in range(0, num_mzis, 2):
            connections += ['mzi%i:0:%i'%(i, i)]
            connections += ['mzi%i:3:%i'%(i, i+1)]
        if N%2:
            connections += ['mzi%i:3:%i'%(N-2, N-1)]

        # output connections:
        k = i+2 + N%2
        connections += ['mzi%i:1:%i'%(0, k)]
        for i in range(1, num_mzis, 2):
            connections += ['mzi%i:1:%i'%(i, k+i)]
            connections += ['mzi%i:2:%i'%(i, k+i+1)]
        if N%2 == 0:
            connections += ['mzi%i:2:%i'%(N-2, 2*N-1)]

        # initialize network
        super(_Capacity2ClemensNxN, self).__init__(components, connections, name=name)

class ClementsNxN(Network):
    r''' A unitary matrix network.

    This unitary matrix implementation is based on the paper of W. R. Clements:
    https://www.osapublishing.org/optica/abstract.cfm?uri=optica-3-12-1460

    Network:
         <--- capacity --->
        0__  ______  ______[]__0
           \/      \/
        1__/\__  __/\__  __[]__1
               \/      \/
        2__  __/\__  __/\__[]__2
           \/      \/
        3__/\______/\______[]__3

        with:
            __[]__ = phase shift
            __  __
              \/    =  MZI
            __/\__

    '''
    def __init__(self, N=2, capacity=None, name=None):
        ''' Clements Network

        Args:
            N (int): matrix size (default 2)
            name (str): name of the network
        '''
        if capacity is None:
            capacity = N

        self.N = N
        self.capacity = capacity

        # create components
        components = {}
        for i in range(capacity//2):
            components['layer%i'%i] = _Capacity2ClemensNxN(N=N)
        if capacity%2==0:
            components['layer%i'%(capacity//2)] = PhaseArray(
                phases=2*np.pi*np.random.rand(N),
                length=0,
                trainable=True,
            )
        else:
            components['layer%i'%(capacity//2)] = _MixingPhaseArray(
                phases=2*np.pi*np.random.rand(N),
                length=0,
                trainable=True,
            )

        # create connections
        connections = []
        for i in range(capacity//2):
            for j in range(N):
                connections += ['layer%i:%i:layer%i:%i'%(i, N+j, i+1, j)]

        # initialize network
        super(ClementsNxN, self).__init__(components, connections, name=name)

    def terminate(self, term=None):
        ''' add sources to input nodes and detectors to output nodes

        Args:
            term=None. Term to use for termination. Default= sources for input nodes,
                detectors for output nodes
        '''
        if term is None:
            term = [Source(name='s%i'%i) for i in range(self.N)]
            term+= [Detector(name='d%i'%i) for i in range(self.N)]
        ret = super(ClementsNxN, self).terminate(term)
        ret.to(self.device)
        return ret
