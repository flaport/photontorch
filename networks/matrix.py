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
            shape: (tuple): shape of the unitary matrix (should be square for now).
            dc (DirectionalCoupler) : master directional coupler. All the directional
                couplers will take over the properties of this directional coupler.
            wg (Waveguide): master waveguide. A waveguide containing all the properties of the connections
                between directional couplers. All waveguides in this network will take
                over the properties of this waveguide.
        '''
        self.name = name
        m,n = shape
        self.n = n

        # checks
        if m != n:
            raise ValueError('A unitary matrix should be square [for now]')
        if not isinstance(dc, DirectionalCoupler):
            raise TypeError('A DirectionalCoupler instance is expected for dc')
        if not isinstance(wg, Waveguide):
            raise TypeError('A Waveguide instance is expected for wg')

        # helper functions
        def new_wg(i, trainable):
            phase = 0.5 if wg.phase is None else wg.phase.data
            new_wg = Waveguide(length=wg.length, neff=wg.neff, loss=wg.loss, phase=phase, trainable=trainable, name='wg_%i'%i)
            return new_wg
        def new_dc(i):
            new_dc = DirectionalCoupler(coupling=dc.coupling.data, trainable=True, name='dc_%i'%i)
            return new_dc

        # Generate waveguides
        trainable_locations = np.zeros((2*n-1,n), dtype=bool)
        self.waveguides = np.zeros((2*n-1,n), dtype=object)
        self.waveguides.fill(None)
        k = -1
        for i in range(2*n-1):
            for j in range((2*n-i)//2):
                k+=1
                self.waveguides[i,j] = new_wg(k, trainable=i%2)
                trainable_locations[i,j] = i%2
            if i%2==0:
                self.waveguides[i,j] = new_wg(k, trainable=True)
                trainable_locations[i,j] = True
            else:
                self.waveguides[i,j+1] = self.waveguides[i-1,j+1]
        self.trainable_locations = list(zip(*np.where(trainable_locations)))

        # Generate directional couplers
        self.directionalcouplers = [new_dc(i) for i in range(n*(n+1)//2)]

        # Components dictionary:
        components = OrderedDict()
        for wg in self.waveguides.ravel():
            if wg is not None:
                components[wg.name] = wg
        for dc in self.directionalcouplers:
            if dc is not None:
                components[dc.name] = dc

        # Connections list:
        connections = [wg.name+':0:%i'%i for i, wg in enumerate(self.waveguides[0])]
        connections += [wg.name+':1:%i'%(2*n-1-i) for i, wg in enumerate(list(self.waveguides[1::2,0]) + [self.waveguides[-1,0]])]
        k = -1
        for i in range(1,2*n-1,2):
            for j in range((2*n-i)//2):
                k += 1
                connections += [
                    self.waveguides[i-1,j].name + ':1:' + self.directionalcouplers[k].name + ':0',
                    self.waveguides[i,j].name+':0:' + self.directionalcouplers[k].name + ':1',
                    self.waveguides[i+1,j].name+':0:' + self.directionalcouplers[k].name + ':2',
                    self.waveguides[i,j+1].name+':1:' + self.directionalcouplers[k].name + ':3',
                ]

        # Termination Flag
        self.terminated = False

        # Initialize network
        super(UnitaryMatrixNetwork, self).__init__(components, connections, name=name)

    def terminate(self, term=None, name=None):
        if self.terminated:
            return self # do nothing
        if term is None:
            term = OrderedDict()
            for i in range(self.n):
                term['s%i'%i] = Source(name='s%i'%i)
            for i in range(self.n):
                term['d%i'%i] = Detector(name='d%i'%i)
        if isinstance(term , Term):
            term = {term.name+'_%i'%i:term.copy() for i in range(n)}
        if isinstance(term, (list, tuple)):
            term = {t.name:t for t in term}
        if self.is_cuda:
            term = {name:t.cuda() for name, t in term.items()}
        self.components.update(term)

        connections = [wg.name+':0:'+t.name+':0' for t, wg in zip(list(term.values())[:self.n],self.waveguides[0])]
        connections += [wg.name+':1:'+t.name+':0' for t, wg in zip(list(term.values())[self.n:],(list(self.waveguides[1::2,0]) + [self.waveguides[-1,0]])[::-1])]

        self.connections[:2*self.n] = connections

        if name is None:
            name = self.name

        super(UnitaryMatrixNetwork, self).__init__(self.components, self.connections, name=name)

        return self


    def unterminate(self, name=None):
        if not self.terminated:
            return self # do nothing
        connections = [wg.name+':0:%i'%i for i, wg in enumerate(self.waveguides[0])]
        connections += [wg.name+':1:%i'%(2*n-1-i) for i, wg in enumerate(list(self.waveguides[1::2,0]) + [self.waveguides[-1,0]])]
        self.connections[:2*self.n] = connections

        if name is None:
            name = self.name

        super(UnitaryMatrixNetwork, self).__init__(self.components, self.connections, name=name)

        return self


    def fit(self, U, max_epochs=1500, tol=1e-7, progress_bar=True, learning_rate=0.1):
        ''' Fit Unitary Matrix Network to a unitary matrix U '''

        # Perfom checks
        if U.shape != (self.n, self.n):
            raise ValueError('Shape mismatch for fit.')
        if not(np.allclose(U.T.conj().dot(U), np.eye(self.n)) and np.allclose(U.dot(U.T.conj()), np.eye(self.n))):
            raise ValueError('fit expects a unitary matrix')

        # Create Range (with or without progress bar)
        if progress_bar:
            from tqdm import trange
            r = trange(self.n-1)
        else:
            r = range(self.n-1)
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
            source = self.tensor(source)
            return source

        # Create source and target
        source = new_source(U.T.conj())
        target = new_source(np.eye(self.n,self.n))[...,:self.n,:]
        lossfunc = torch.nn.MSELoss()

        # Start training
        for i in r:
            loss = torch.tensor(1)
            optimizer = torch.optim.Adam(parameters(i), lr=learning_rate)
            for _ in range(max_epochs):
                r.set_postfix(loss='{loss:.{x}f}'.format(loss=loss.item(), x=int(-np.log10(tol)+0.5)))
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


####################
## General Matrix ##
####################

class MatrixNetwork(Network):
    ''' A general Matrix Network

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

    def __init__(self, shape, dc, wg, soa, name=None):
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
        m, n = shape

        # Create unitary matrices
        U = UnitaryMatrixNetwork((m,m), dc, wg, name='Unw')
        V = UnitaryMatrixNetwork((n,n), dc, wg, name='Vnw')

        # Connection strings for unitary matrices:
        k = ord('a')
        Uin = ''.join(chr(k) for k in range(k, k+m))
        Uout = ''.join(chr(k) for k in range(k+m, k+2*m))
        Vin = ''.join(chr(k) for k in range(k+2*m, k+2*m+n))
        Vout = ''.join(chr(k) for k in range(k+2*m+n, k+2*m+2*n))

        # Looking at the singular value decomposition USV, we see that
        # we first have to describe the action of V, then amplify
        # and then we have to describe the action of U.

        # Connector
        conn = U[Uin+Uout]*V[Vin+Vout]

        # Inputs
        self.input_s = Vin

        # Amplification
        self.soas = []
        for vc, uc in zip(Vout, Uin):
            soa = soa.copy()
            soa.name = vc+uc
            self.soas.append(soa)
            conn = conn*soa[vc+uc]

        # Terms
        if m > n: # Add terms to unused inputs of U:
            for c in Uin[-(m-n):]:
                conn = conn * Term(name=c)[c]
            Uin = Uin[:-(m-n)]
        elif m < n: # Add terms to unused outputs of V:
            for c in Vout[-(n-m):]:
                conn = conn*Term(name=c)[c]
            Vout = Vout[:-(n-m)]

        # Outputs
        self.output_s = Uout

        # Initialize:
        self.conn = conn
        super(MatrixNetwork, self).__init__(conn, name=name)

    def copy(self):
        for dc in self.Vself.dcs.values():
            dc = dc.components[0]
            break
        for wg in self.Vself.hwgs.values():
            wg = wg.components[0]
            break
        for soa in self.soas:
            break
        new = self.__class__(self.shape, dc, wg, soa, name=self.name)
        new.conn.components = [comp.copy() for comp in self.conn.components]
        new.soas = [comp for comp in new.conn.components if isinstance(comp, soa.__class__)]

        super(MatrixNetwork, new).__init__(new.conn, name=self.name)

        return new

    def terminate(self):
        '''
        A Matrix Network is terminated by Sources on V and detectors on U
        '''
        is_cuda = self.is_cuda

        def maybe_cuda(comp):
            return comp if not is_cuda else comp.cuda()

        for c in self.input_s:
            self.conn = self.conn*maybe_cuda(Source(name=c))[c]
        for c in self.output_s:
            self.conn = self.conn*maybe_cuda(Detector(name=c))[c]

        super(MatrixNetwork, self).__init__(self.conn, name=self.name)

        self.is_cuda = is_cuda

        return self

    def unterminate(self):
        ''' Undo de termination of the network '''
        is_cuda = self.is_cuda

        s = self.conn.s.split(',')
        components = self.conn.components
        s, self.components = zip(*[(ss, comp) for (ss, comp) in zip(s, components) if not isinstance(comp, Term)])
        self.s = ','.join(s)
        self.conn = Connector(self.s, self.components)

        self.is_cuda = is_cuda
        return self

    def fit(self, M, max_epochs=1000, tol=1e-7, progress_bar=True, optimizer=None, lossfunc=None):
        ''' Fit Unitary Matrix Network to a unitary matrix U '''
        m, n = self.shape

        if self.shape != M.shape:
            raise ValueError('Shape mismatch for fit')

        U,S,V = np.linalg.svd(M)

        self.Vself.fit(
            U=V,
            max_epochs=max_epochs,
            tol=tol,
            progress_bar=progress_bar,
            optimizer=optimizer,
            lossfunc=lossfunc,
        )

        self.Uself.fit(
            U=U,
            max_epochs=max_epochs,
            tol=tol,
            progress_bar=progress_bar,
            optimizer=optimizer,
            lossfunc=lossfunc,
        )

        for amplification, soa in zip(S, self.soas):
            soa.amplification.data[0] = amplification

        return self

    @property
    def M(self):
        ''' Matrix represented by this matrix network '''
        m,n = self.shape
        return self.matmul(np.eye(n,n))

    def matmul(self, other):
        ''' Matrix multiplication of the network with a numpy array '''
        # we first check if the numpy array is 2D:
        if other.ndim != 2:
            raise ValueError('Matrix Multiplication of a Unitary Matrix Network with a numpy'
                             'array expects a 2D array.')
        if other.shape[1] != self.num_sources:
            raise ValueError('Shape mismatch between Unitary Matrix Network and numpy array')

        # Terminate if necessary
        terminated = self.terminated
        if not terminated:
            self.terminate()

        # Initialize with simplified environment
        old_env = self.env.copy()
        new_env = self.env.copy(
            use_delays=False,
            num_wl=1,
            num_timesteps=1,
            num_batches=other.shape[1],
        )
        self.initialize(new_env)

        # Create Source
        source = np.zeros((2,1,1,self.nmc,other.shape[1]))
        source[0,0,0,:other.shape[0],:] = np.real(other)
        source[1,0,0,:other.shape[0],:] = np.imag(other)
        source = self.tensor(source)

        # Propagate
        result = self(source, power=False)[:,0,0,:,:].data.cpu().numpy()

        # Unterminate if necessary
        if not terminated:
            self.unterminate()

        # Initialize with old environment
        self.initialize(old_env)

        # Return
        return result[0] + 1j*result[1]

    def __matmul__(self, other):
        ''' @ operator for python 3 '''
        return self.matmul(other)
