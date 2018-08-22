'''
# Matrix Networks

Directional coupler networks that act like a matrix multiplication.

'''

#############
## Imports ##
#############

# Standard Library
from copy import copy
from collections import OrderedDict

# Torch
import torch

# Other
import numpy as np

# Relative
from .network import Network
from .connector import Connector
from ..components.soas import LinearSoa
from ..components.waveguides import Waveguide
from ..components.directionalcouplers import DirectionalCoupler
from ..components.terms import Term, Source, Detector
from ..torch_ext.nn import BoundedParameter


############################
## Unitary Matrix Network ##
############################

class UnitaryMatrixNetwork(Network):
    ''' A (square) unitary matrix network.
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

    def __init__(self, shape, dc, wg, name='umn'):
        ''' UnitaryMatrixNetwork

        Args:
            shape: (tuple): shape of the unitary matrix (m <= n [for now]).
            dc (DirectionalCoupler) : master directional coupler. All the directional
                couplers will take over the properties of this directional coupler.
            wg (Waveguide): master waveguide. A waveguide containing all the properties of the connections
                between directional couplers. All waveguides in this network will take
                over the properties of this waveguide.
        '''
        self.shape = shape
        self.m, self.n = m, n = shape

        # checks
        if m > n:
            raise ValueError('m <= n is expected [for now]')
        if not isinstance(dc, DirectionalCoupler):
            raise TypeError('A DirectionalCoupler instance is expected for dc')
        if not isinstance(wg, Waveguide):
            raise TypeError('A Waveguide instance is expected for wg')

        # helper functions
        def new_wg(i, trainable):
            phase = 0.5 if wg.phase is None else wg.phase.data
            new_wg = Waveguide(length=wg.length, neff=wg.neff, loss=wg.loss, phase=phase, trainable=trainable, name='wg%i'%i)
            return new_wg
        def new_dc(i):
            new_dc = DirectionalCoupler(coupling=dc.coupling.data, trainable=True, name='dc%i'%i)
            return new_dc

        # Generate waveguides
        p = max(m,n)
        trainable_locations = np.zeros((2*p-1,p), dtype=bool)
        self.waveguides = np.zeros((2*p-1,p), dtype=object)
        self.waveguides.fill(None)
        k = -1
        for i in range(2*m):
            for j in range((2*n-i)//2):
                k+=1
                self.waveguides[i,j] = new_wg(k, trainable=i%2)
                trainable_locations[i,j] = i%2
            if i%2==0:
                self.waveguides[i,j] = new_wg(k, trainable=True)
                trainable_locations[i,j] = True
            elif i<2*n-1:
                self.waveguides[i,j+1] = self.waveguides[i-1,j+1]
        self.trainable_locations = list(zip(*np.where(trainable_locations)))

        # Generate directional couplers
        self.directionalcouplers = []
        k = -1
        for i in range(1,2*p,2):
            for j in range((2*n-i)//2):
                k += 1
                if self.waveguides[i,j] is not None:
                    self.directionalcouplers += [new_dc(k) if j<n else None]
                else:
                    self.directionalcouplers += [None]

        # Components dictionary:
        components = OrderedDict()
        for wg in self.waveguides.ravel():
            if wg is not None:
                components[wg.name] = wg
        for dc in self.directionalcouplers:
            if dc is not None:
                components[dc.name] = dc

        # Connections list:
        connections = [wg.name+':0:%i'%i for i, wg in enumerate(self.waveguides[0]) if wg is not None]
        connections += [wg.name+':1:%i'%(m+n-1-i) for i, wg in enumerate(list(self.waveguides[1::2,0]) + [self.waveguides[-1,0]]) if wg is not None]
        k = -1
        for i in range(1,2*p,2):
            for j in range((2*n-i)//2):
                k += 1

                if self.directionalcouplers[k] is None:
                    continue

                try:
                    connections += [self.waveguides[i-1,j].name + ':1:' + self.directionalcouplers[k].name + ':0']
                except:
                    t = Term(name=self.directionalcouplers[k].name+'_t0')
                    components[t.name] = t
                    connections += [t.name + ':0:' + self.directionalcouplers[k].name+':0']
                try:
                    connections += [self.waveguides[i,j].name+':0:' + self.directionalcouplers[k].name + ':1']
                except:
                    t = Term(name=self.directionalcouplers[k].name+'_t1')
                    components[t.name] = t
                    connections += [t.name + ':0:' + self.directionalcouplers[k].name+':1']
                try:
                    connections += [self.waveguides[i+1,j].name+':0:' + self.directionalcouplers[k].name + ':2']
                except:
                    t = Term(name=self.directionalcouplers[k].name+'_t2')
                    components[t.name] = t
                    connections += [t.name + ':0:' + self.directionalcouplers[k].name+':2']
                try:
                    connections += [self.waveguides[i,j+1].name+':1:' + self.directionalcouplers[k].name + ':3']
                except:
                    t = Term(name=self.directionalcouplers[k].name+'_t3')
                    components[t.name] = t
                    connections += [t.name + ':0:' + self.directionalcouplers[k].name+':3']

        # Termination Flag
        self.terminated = False

        # Initialize network
        super(UnitaryMatrixNetwork, self).__init__(components, connections, name=name)

    def terminate(self, term=None, name=None):

        env = self.env

        if self.terminated:
            return self # do nothing

        if term is None:
            term = OrderedDict()
            for i in range(self.n):
                term['s%i'%i] = Source(name='s%i'%i)
            for i in range(self.m):
                term['d%i'%i] = Detector(name='d%i'%i)
        if isinstance(term , Term):
            term = OrderedDict([(term.name+'_%i'%i,term.copy()) for i in range(self.m+self.n)])
        if isinstance(term, (list, tuple)):
            term = OrderedDict([(t.name,t) for t in term])
        if self.is_cuda:
            term = OrderedDict([(name,t.cuda()) for name, t in term.items()])

        self.components.update(term)

        input_waveguides = [wg for wg in self.waveguides[0] if wg is not None]
        output_waveguides = [wg for wg in list(self.waveguides[1::2,0]) + [self.waveguides[-1,0]] if wg is not None][::-1]
        connections = [wg.name+':0:'+t.name+':0' for t, wg in zip(list(term.values())[:self.n], input_waveguides)]
        connections += [wg.name+':1:'+t.name+':0' for t, wg in zip(list(term.values())[self.n:], output_waveguides)]

        self.connections[:self.m+self.n] = connections

        if name is None:
            name = self.name

        super(UnitaryMatrixNetwork, self).__init__(self.components, self.connections, name=name)

        if env is not None:
            self.initialize(env)

        return self

    def unterminate(self, name=None):

        env = self.env

        if not self.terminated:
            return self # do nothing

        connections = [wg.name+':0:%i'%i for i, wg in enumerate(self.waveguides[0]) if wg is not None]
        connections += [wg.name+':1:%i'%(self.m+self.n-1-i) for i, wg in enumerate(list(self.waveguides[1::2,0]) + [self.waveguides[-1,0]]) if wg is not None]

        self.connections[:self.m+self.n] = connections

        if name is None:
            name = self.name

        super(UnitaryMatrixNetwork, self).__init__(self.components, self.connections, name=name)

        if env is not None:
            self.initialize(env)

        return self

    def fit(self, U, max_epochs=1500, tol=1e-7, progress_bar=True, learning_rate=0.1):
        ''' Fit Unitary Matrix Network to a unitary matrix U '''

        # Perfom checks
        if U.shape != (self.m, self.n):
            raise ValueError('Shape mismatch for fit.')
        if not np.allclose(U.dot(U.T.conj()), np.eye(self.m)):
            raise ValueError('fit expects a unitary matrix')

        # Create Range (with or without progress bar)
        if progress_bar:
            from tqdm import trange
            r = trange(self.m)
        else:
            r = range(self.m)
            r.set_postfix = lambda **kwargs: None # dummy function

        # Terminate network
        was_terminated = self.terminated
        self.terminate()

        # Initialize terminated network
        self.initialize(self.env)

        # Helper function to retrieve relevant network parameters:
        def parameters(i):
            if i == self.n-1:
                selected_components = [self.waveguides[-1,0]]
            else:
                selected_components = [wg for wg in self.waveguides[2*i+1] if wg is not None]
                selections = np.arange(self.n-1,0,-1)
                start_idxs = np.cumsum(np.hstack([[0], selections[:-1]]))
                idx = start_idxs[i]
                s = selections[i]
                selected_components += self.directionalcouplers[idx:idx+s]
            for comp in selected_components:
                for p in comp.parameters():
                    yield p

        # Helper function to create a source
        def new_source(vector):
            source = np.zeros((2,1,1,self.nmc,vector.shape[1]))
            source[0,0,0,:vector.shape[0],:] = np.real(vector)
            source[1,0,0,:vector.shape[0],:] = np.imag(vector)
            source = torch.tensor(source, device=self.device)
            return source

        # Create source and target
        source = new_source(U.T.conj())
        target = new_source(np.eye(self.m,self.m))[...,:self.m,:]
        lossfunc = torch.nn.MSELoss()

        # Start training
        for i in r:
            loss = torch.tensor(1)
            optimizer = torch.optim.Adam(parameters(i), lr=learning_rate)
            for e in range(max_epochs):
                r.set_postfix(epoch='%i'%e, loss='{loss:.{x}f}'.format(loss=loss.item(), x=int(-np.log10(tol)+0.5)))
                if loss.item() < tol:
                    break
                optimizer.zero_grad()
                self.initialize(self.env)
                result = self(source, power=False)
                loss = lossfunc(result[...,-1-i,:], target[...,-1-i,:])
                loss.backward()
                optimizer.step()

        # Finalize
        if not was_terminated:
            self.unterminate()

        return self

    def transpose(self):
        ''' switch detectors and sources
        [this is an easy way around the fact that mxn matrices with m>n are not
        implemented]

        TODO: implement mxn matrices with m>n
        '''

        if not self.terminated:
            raise Exception('The network needs to be terminated')

        env = self.env

        components = OrderedDict()
        for name, comp in self.components.items():
            if isinstance(comp, Source):
                name = name.replace('s','d')
                comp = Detector(name)
            elif isinstance(comp, Detector):
                name = name.replace('d','s')
                comp = Source(name)
            components[name] = comp

        connections = []
        for conn in self.connections:
            conn = conn.replace('dc','@#').replace('d','$').replace('s','d').replace('$','s').replace('@#','dc')
            connections += [conn]

        super(UnitaryMatrixNetwork, self).__init__(components, connections, name=self.name)

        if env is not None:
            self.initialize(env)

        return self



####################
## General Matrix ##
####################

class MatrixNetwork(Network):
    ''' NOT FINISHED

        A general Matrix Network

        Every Matrix $M$ can be decomposed by the singular value decomposition as
        follows:

        ```
            math M = USV^*
        ```

        With U and V unitary matrices, and S the diagonal matrix containing the
        singular values.

        Practically, we implement this by using two UnitaryMatrixNetworks connected by an
        array of SOAs.

        Unused output ports of the U-network (in the case of a non-square matrix) will be
        terminated with a Term.
    '''

    def __init__(self, shape, dc, wg, soa, name='mn'):
        ''' MatrixNetwork

        Args:
            shape: (tuple): shape of the matrix.
            dc (DirectionalCoupler) : master directional coupler. All the directional
                couplers will take over the properties of this directional coupler.
            wg (Waveguide): master waveguide. A waveguide containing all the properties
                of the connections between directional couplers. All waveguides in this
                network will take over the properties of this waveguide.
            soa (SOA): master SOA. The amplifier to be used to implement the singular
                values
        Note:
            The parameters of the network will be randomly initialized. This will thus
            result in a unitary matrix with random values. You need to train the matrix
            for a specific output to set the matrix values.
        '''
        self.shape = shape
        self.m, self.n = m, n = shape

        # checks
        if m > n:
            raise ValueError('m <= n is expected [for now]')
        if not isinstance(dc, DirectionalCoupler):
            raise TypeError('A DirectionalCoupler instance is expected for dc')
        if not isinstance(wg, Waveguide):
            raise TypeError('A Waveguide instance is expected for wg')
        if not isinstance(soa, LinearSoa):
            raise TypeError('A LinearSoa instance is expected for soa')

        # helper functions
        def new_soa(i):
            return LinearSoa(amplification=soa.amplification.data[0].item(), trainable=True, name='soa%i'%i)

        # subnetworks
        U = UnitaryMatrixNetwork((m,m), dc, wg, name='U')
        V = UnitaryMatrixNetwork((m,n), dc, wg, name='V')

        # soas
        self.soas = [new_soa(i) for i in range(m)]

        # components
        components = OrderedDict([(soa.name, soa) for soa in self.soas])
        components['U'] = U
        components['V'] = V

        # connections
        connections = []
        connections += ['V:%i:%i'%(i, i) for i in range(n)]
        connections += ['U:%i:%i'%(m+i,i) for i in range(m)]
        connections += ['V:%i:soa%i:0'%(n+i, i) for i in range(m)]
        connections += ['U:%i:soa%i:1'%(i, i) for i in range(m)]

        super(MatrixNetwork, self).__init__(components, connections, name=name)

    def terminate(self, term=None, name=None):

        env = self.env

        if self.terminated:
            return self # do nothing

        if term is None:
            term = OrderedDict()
            for i in range(self.n):
                term['s%i'%i] = Source(name='s%i'%i)
            for i in range(self.m):
                term['d%i'%i] = Detector(name='d%i'%i)
        if isinstance(term , Term):
            term = OrderedDict([(term.name+'_%i'%i,term.copy()) for i in range(self.m+self.n)])
        if isinstance(term, (list, tuple)):
            term = OrderedDict([(t.name,t) for t in term])
        if self.is_cuda:
            term = OrderedDict([(name,t.cuda()) for name, t in term.items()])

        self.components.update(term)

        connections = ['V:'+str(i)+':'+t.name+':0' for i, t in enumerate(list(term.values())[:self.n])]
        connections += ['U:'+str(self.m+i)+':'+t.name+':0' for i, t in enumerate(list(term.values())[self.n:])]

        self.connections[:self.m+self.n] = connections

        if name is None:
            name = self.name

        super(MatrixNetwork, self).__init__(self.components, self.connections, name=name)

        if env is not None:
            self.initialize(env)

        return self

    def unterminate(self, name=None):

        env = self.env

        if not self.terminated:
            return self # do nothing

        connections += ['V:%i:%i'%(i, i) for i in range(n)]
        connections += ['U:%i:%i'%(m+i,i) for i in range(m)]

        self.connections[:self.m+self.n] = connections

        if name is None:
            name = self.name

        super(MatrixNetwork, self).__init__(self.components, self.connections, name=name)

        if env is not None:
            self.initialize(env)

        return self

    def fit(self, M, max_epochs=1500, tol=1e-7, progress_bar=True, learning_rate=0.1):
        ''' Fit Matrix Network to a matrix M '''

        env = self.env

        # Perfom checks
        if M.shape != (self.m, self.n):
            raise ValueError('Shape mismatch for fit.')

        # svd
        U, S, V = np.linalg.svd(M, full_matrices=False)

        # fit
        self.V.fit(V, max_epochs=max_epochs, tol=tol, progress_bar=progress_bar, learning_rate=learning_rate)
        self.U.fit(U, max_epochs=max_epochs, tol=tol, progress_bar=progress_bar, learning_rate=learning_rate)
        for i, s in enumerate(S):
            self.soas[i].amplification.data[0] = s

        # initialize
        if env is not None:
            self.initialize(env)

        # finish
        return self
