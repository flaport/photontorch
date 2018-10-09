#############
## Imports ##
#############

# other
import numpy as np

# relative
from .network import Network
from ..components.mzis import Mzi
from ..components.mmis import PhaseArray
from ..components.terms import Source, Detector, Term


#############
## Classes ##
#############

class _ReckNxN(Network):
    ''' A helper class for ReckMxN '''
    def __init__(self, N=2, name=None):
        num_mzis = N - 1

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
        for i in range(num_mzis - 1):
            connections += ['mzi%i:3:mzi%i:1'%(i, i+1)]

        # input ports:
        for i in range(num_mzis):
            connections += ['mzi%i:0:%i'%(i,i)]
        connections += ['mzi%i:3:%i'%(i, i+1)]

        # output ports
        connections += ['mzi%i:1:%i'%(0, N)]
        for i in range(num_mzis):
            connections += ['mzi%i:2:%i'%(i, N+i+1)]

        super(_ReckNxN, self).__init__(components, connections, name=name)

class ReckMxN(Network):
    ''' A unitary matrix network.

    This unitary matrix implementation is based on The paper of M. Reck:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58

    Network:
                        .--- : 9
                        |
                        |
                        2
                   .---3 1-- : 8
                   |    0
                   |    |
                   2    2
              .---3 1--3 1-- : 7
              |    0    0
              |    |    |
              2    2    2
         .---3 1--3 1--3 1-- : 6
         |    0    0    0
         |    |    |    |
         2    2    2    2
    .---3 1--3 1--3 1--3 1-- : 5
    |    0    0    0    0
    |    |    |    |    |
   ..   ..   ..   ..   ..
    4    3    2    1    0

    '''
    def __init__(self, M=2, N=None, name=None):
        ''' Reck Network

        Args:
            M (int): first element of matrix shape
            N (int): second element of matrix shape (default N)
            name (str): name of the network
        '''
        if N is None:
            N = M

        if M > N:
            raise ValueError('M<N required')

        if M < 0:
            raise ValueError('M>0 required')

        if N < 2:
            raise ValueError('N>2 required')

        self.M = M
        self.N = N

        # define components
        components = {}
        for n in range(N-M-int(M==1)+2, N+1):
            components['r%i'%n] = _ReckNxN(N=n)
        components['pa'] = PhaseArray(
            phases=2*np.pi*np.random.rand(N),
            length=0,
            trainable=True,
        )

        # define conections
        connections = []
        for n in range(N-M+2, N):
            for i in range(n):
                connections += ['r%i:%i:r%i:%i'%(n, n+i, n+1, i)]

        for i in range(N):
            connections += ['r%i:%i:pa:%i'%(N, N+i, i)]

        super(ReckMxN, self).__init__(components, connections, name=name)

    def terminate(self, term=None, transposed=False):
        ''' add sources to input nodes, detectors to output nodes and terms to unused
        nodes

        Args:
            term=None. Term to use for termination. Default= sources for input nodes,
                detectors for output nodes and terms for unused nodes
            transposed (bool) = False. Switch sources and detectors around, so the input
                nodes become the output nodes and vice versa
        '''

        def _term(i):
            return Term(name='t%i'%i)

        def _source(i):
            return Source(name='s%i'%i) if not transposed else Detector(name='d%i'%i)

        def _detector(i):
            return Detector(name='d%i'%i) if not transposed else Source(name='s%i'%i)

        if term is None:
            term = []
            term += [_term(i) for i in range(self.N-self.M)]
            term += [_source(i) for i in range(self.M)]
            term += [_detector(i) for i in range(self.N)]

        ret = super(ReckMxN, self).terminate(term)
        ret.to(self.device)

        return ret

class ReckNxN(ReckMxN):
    ''' A unitary matrix network.

    This unitary matrix implementation is based on The paper of M. Reck:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58

    Network:
                        .--- : 9
                        |
                        |
                        2
                   .---3 1-- : 8
                   |    0
                   |    |
                   2    2
              .---3 1--3 1-- : 7
              |    0    0
              |    |    |
              2    2    2
         .---3 1--3 1--3 1-- : 6
         |    0    0    0
         |    |    |    |
         2    2    2    2
    .---3 1--3 1--3 1--3 1-- : 5
    |    0    0    0    0
    |    |    |    |    |
   ..   ..   ..   ..   ..
    4    3    2    1    0

    '''
    def __init__(self, N=2, name=None):
        ''' Reck Network

        Args:
            N (int): matrix size (NxN)
            name (str): name of the network
        '''
        super(ReckNxN, self).__init__(M=N, N=N, name=name)

class ReckMmi(ReckMxN):
    ''' An Mmi with weights represented by a unitary matrix network.

    This unitary matrix implementation is based on The paper of M. Reck:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58

    Network:
                        .--- : 9
                        |
                        |
                        2
                   .---3 1-- : 8
                   |    0
                   |    |
                   2    2
              .---3 1--3 1-- : 7
              |    0    0
              |    |    |
              2    2    2
         .---3 1--3 1--3 1-- : 6
         |    0    0    0
         |    |    |    |
         2    2    2    2
    .---3 1--3 1--3 1--3 1-- : 5
    |    0    0    0    0
    |    |    |    |    |
   ..   ..   ..   ..   ..
    4    3    2    1    0

    '''
    pass
