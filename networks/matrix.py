'''
# Matrix Networks

Directional coupler networks that act like a matrix multiplication.

'''

#############
## Imports ##
#############

# Standard Library
from collections import OrderedDict

# Torch
import torch

# Other
import numpy as np

# Relative
from .network import Network
from .connector import Connector
from ..components.terms import Term, Source, Detector
from ..torch_ext.nn import BoundedParameter


#############
## Unitary ##
#############

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

    def __init__(self, shape, dc, wg, name=None):
        ''' UnitaryMatrixNetwork

        Args:
            shape: (tuple): shape of the unitary matrix (should be square for now).
            dc (DirectionalCoupler) : master directional coupler. All the directional
                couplers will take over the properties of this directional coupler.
            wg (Waveguide): master waveguide. A waveguide containing all the properties of the connections
                between directional couplers. All waveguides in this network will take
                over the properties of this waveguide.

        Note:
            The parameters of the network will be randomly initialized. This will thus
            result in a unitary matrix with random values. You need to train the matrix
            for a specific output to set the matrix values.
        '''
        M,N = shape
        if M != N:
            raise ValueError('A Unitary matrix should be square')
        if M < 2:
            raise ValueError('The unitary matrix should be at least 2x2.')

        self.shape = shape

        self.dcs = dcs = OrderedDict()#np.empty((M-1,M-1), dtype=object); dcs.fill(None)
        self.hwgs = hwgs = OrderedDict()#np.empty((M, M), dtype=object); hwgs.fill(None)
        self.vwgs = vwgs = OrderedDict()#np.empty((M-1, M-1), dtype=object); vwgs.fill(None)

        def copy(obj, name):
            ''' copy an object with a different name '''
            new_obj = obj.copy()
            new_obj.name = name
            return new_obj

        # Output Nodes
        k = ord('A')
        self.input_s = input_s = ''.join(chr(k) for k in range(k, k+M))
        self.output_s = output_s = ''.join(chr(k) for k in range(k+M, k+2*M))

        # Directional Couplers
        k += 2*M
        for m in range(M-1):
            for n in range(m+1):
                s = chr(k) + chr(k+1) + chr(k+2) + chr(k+3)
                new_dc = copy(dc, 'dc')
                dcs[m,n] = new_dc[s]
                k += 4


        # Horizontal wgs
        # The horizontal waveguides usually have a trainable phase
        for m in range(M):
            for n in range(m+1):
                new_wg = copy(wg, 'wg')
                if (m,n) == (0,0):
                    hwgs[0,0] = new_wg[dcs[0,0].s[2]+output_s[0]]
                elif (m,n) == (M-1,M-1):
                    hwgs[M-1,M-1] = new_wg[dcs[M-2,N-2].s[-1]+input_s[-1]]
                elif m==n:
                    hwgs[m,m] = new_wg[dcs[m-1,m-1].s[-1]+dcs[m,m].s[2]]
                elif n == 0: # n > 0
                    hwgs[m,0] = new_wg[dcs[m-1,0].s[1]+output_s[m]]
                else:
                    hwgs[m,n] = new_wg[dcs[m-1,n].s[1]+dcs[m-1,n-1].s[-1]]

        # Vertical wgs
        # The vertical waveguides usually have a non-trainable phase.
        for m in range(M-1):
            for n in range(m+1):
                # Make waveguide untrainable (fixed length and phase):
                new_wg = wg.__class__(length=wg.length.data[0], neff=wg.neff, name='wg')
                if m == M-2:
                    vwgs[M-2,n] = new_wg[dcs[M-2,n].s[0]+input_s[n]]
                else:
                    vwgs[m,n] = new_wg[dcs[m,n].s[0]+dcs[m+1,n].s[2]]

        dc_list = list(dcs.values())
        vwgs_list = list(vwgs.values())
        hwgs_list = list(hwgs.values())

        conn = hwgs_list.pop(-1)
        for c in dc_list+vwgs_list[::-1]+hwgs_list[::-1]:
            conn = conn*c

        self.conn = conn

        super(UnitaryMatrixNetwork, self).__init__(conn, name=name)

    @classmethod
    def from_dicts(cls, shape, dcs, hwgs, vwgs, name=None):
        ''' Alternative class constructor '''
        for dc in dcs.values():
            dc = dc.components[0]
            break
        for wg in hwgs.values():
            wg = wg.components[0]
            break
        new = cls(shape, dc, wg, wg)
        new.dcs = dcs
        new.hwgs = hwgs
        new.vwgs = vwgs
        dc_list = list(new.dcs.values())
        vwgs_list = list(new.vwgs.values())
        hwgs_list = list(new.hwgs.values())

        conn = hwgs_list.pop(-1)
        for c in dc_list+vwgs_list[::-1]+hwgs_list[::-1]:
            conn = conn*c
        new.conn = conn
        super(UnitaryMatrixNetwork, new).__init__(conn, name=name)

        return new

    def copy(self):
        for dc in self.dcs.values():
            dc = dc.components[0]
            break
        for wg in self.hwgs.values():
            wg = wg.components[0]
            break
        new = self.__class__(self.shape, dc, wg)
        new.dcs = self.copydict(self.dcs)
        new.hwgs = self.copydict(self.hwgs)
        new.vwgs = self.copydict(self.vwgs)

        dc_list = list(new.dcs.values())
        vwgs_list = list(new.vwgs.values())
        hwgs_list = list(new.hwgs.values())

        conn = hwgs_list.pop(-1)
        for c in dc_list+vwgs_list[::-1]+hwgs_list[::-1]:
            conn = conn*c

        super(UnitaryMatrixNetwork, new).__init__(conn, name=self.name)

        return new

    def terminate(self):
        '''
        A Matrix Network is terminated by Sources on the bottom and detectors on the right
        '''
        for c in self.input_s:
            self.conn = self.conn*Source(name=c)[c]
        for c in self.output_s:
            self.conn = self.conn*Detector(name=c)[c]
        super(UnitaryMatrixNetwork, self).__init__(self.conn, name=self.name)

        return self

    def split(self, m):
        ''' Split the network in an upper half and a lower half at output m'''
        M, _ = self.shape
        if m > M:
            raise ValueError('cannot create a submatrix that is bigger than the original')

        def copy(conn):
            ''' shallow copy of a single component connector '''
            return conn.components[0][conn.s]

        # lower half of matrix
        hwgs = OrderedDict((k,copy(v)) for k,v in self.hwgs.items() if k[0]>m-2)
        vwgs = OrderedDict((k,copy(v)) for k,v in self.vwgs.items() if k[0]>m-3)
        dcs = OrderedDict((k,copy(v)) for k,v in self.dcs.items() if k[0]>m-3)

        for i, c in enumerate(self.output_s[:m-1]):
            s = dcs[m-2,i].s
            dcs[m-2,i].s = s[0] + s[1] + c + s[3]

        name = '_'.join([self.name.split('_')[0], 'btm%i'%m])
        btm = self.__class__.from_dicts(self.shape, dcs, hwgs, vwgs, name=self.name+'_btm%i'%m)

        # upper half of matrix
        hwgs = OrderedDict((k,copy(v)) for k,v in self.hwgs.items() if k[0]<=m-2)
        vwgs = OrderedDict((k,copy(v)) for k,v in self.vwgs.items() if k[0]<=m-3)
        dcs = OrderedDict((k,copy(v)) for k,v in self.dcs.items() if k[0]<=m-3)

        for i, c in enumerate(self.input_s[:m-2]):
            s = vwgs[m-3, i].s
            vwgs[m-3,i].s = s[0] + c

        c = self.input_s[-2]
        s = hwgs[m-2,m-2].s
        hwgs[m-2,m-2].s = s[0] + c

        i = ord(c) + 1
        for k, conn in hwgs.items():
            if k[1] == 0:
                conn.s = conn.s[0] + chr(i)
                i += 1

        name = '_'.join([self.name.split('_')[0], 'top%i'%m])
        top = self.__class__.from_dicts((m-1,m-1), dcs, hwgs, vwgs, name=name)

        if self.terminated:
            return top.terminate(), btm.terminate()

        return top, btm

    @staticmethod
    def copydict(dict):
        new_dict = dict.__class__() # can be an ordered dict...
        for k, v in dict.items():
            new_dict[k] = v.copy()
        return new_dict

    @staticmethod
    def dict2array(dict):
        M, N = [max(lst)+1 for lst in zip(*dict.keys())]
        array = np.empty((M,N), dtype=object)
        array.fill(None)
        for k, v in dict.items():
            array[k] = v
        return array

    @property
    def dc_array(self):
        return self.dict2array(self.dcs)

    @property
    def hwg_array(self):
        return self.dict2array(self.hwgs)

    @property
    def vwg_array(self):
        return self.dict2array(self.vwgs)

    def fit(self, U, max_epochs=1000, tol=1e-7, progress_bar=True, optimizer=None, lossfunc=None):
        ''' Fit Unitary Matrix Network to a unitary matrix U '''
        m, n = self.shape

        # Perfom checks
        if U.shape != self.shape:
            raise ValueError('Shape mismatch for fit.')
        if not(np.allclose(U.T.conj().dot(U), np.eye(m)) and np.allclose(U.dot(U.T.conj()), np.eye(m))):
            raise ValueError('fit expects a unitary matrix')

        # Create Range (with or without progress bar)
        if progress_bar:
            from tqdm import trange
            r = trange(m, 2, -1)
        else:
            r = range(m, 2, -1)
            r.set_postfix = lambda **kwargs: None

        # Create Optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.25)

        # Create Loss Function
        if lossfunc is None:
            lossfunc = torch.nn.MSELoss()

        # Create a copy of U for later use:
        U0 = U.copy()
        U = U.T.conj()

        # create simplified environment, but keep a reference to the old environment
        old_env = self.env.copy()
        env = self.env.copy(use_delays=False, num_wl=1, num_timesteps=1, num_batches=1)

        # We will need to create plenty of sources:
        def new_source(nw, array):
            source = np.zeros((2, 1, 1, nw.nmc, array.shape[1]))
            source[0, 0, 0, :array.shape[0], :] = np.real(array)
            source[1, 0, 0, :array.shape[0], :] = np.imag(array)
            source = nw.new_variable(source)
            return source

        # Start training:
        top = self
        for k in r:
            loss = 1
            env = env.copy(num_batches=k)
            top.initialize(env)
            source = new_source(top, U)
            target = top.new_variable(np.eye(k))
            top, btm = top.split(k)
            for i in range(max_epochs):
                r.set_postfix(loss='{loss:.{x}f}'.format(loss=loss, x=int(-np.log10(tol)+0.5)))
                btm.initialize(env)
                prediction = btm(source)[0,0,:,k-1:k]
                loss = lossfunc(prediction, target[:,k-1:k])
                loss.backward()
                optimizer.step()
                loss = loss.data[0]
                if loss < tol:
                    break
                del prediction

            # Update U:
            U = (btm@U)[:-1, :-1]

        # Loop is over. We now handle the k==2 case:
        env = env.copy(num_batches=2)
        top.initialize(env)
        source = new_source(top, U)
        target = top.new_variable(np.eye(2))
        for i in range(max_epochs):
            r.set_postfix(loss='{loss:.{x}f}'.format(loss=loss, x=int(-np.log10(tol)+0.5)))
            top.initialize(env)
            prediction = top(source)[0,0,:,:]
            loss = lossfunc(prediction, target)
            loss.backward()
            optimizer.step()
            loss = loss.data[0]
            if loss < tol:
                break

        # Last step is to adjust the output phase:
        # Replace U.T.conj() by the matrix represented by the network?
        out = np.diag(self@U0.T.conj())
        phase_correction = np.arctan2(np.imag(out), np.real(out))
        phases = [conn.components[0].phase.data for k, conn in self.hwgs.items() if k[1] == 0]
        for phase, pc in zip(phases, phase_correction):
            phase[0] -= pc


    @property
    def U(self):
        ''' Matrix represented by this matrix network '''
        m,n = self.shape
        if self.env.num_wl > 1:
            raise RuntimeError('Can only get matrix representation for a network with num_wl==1')
        rU = self._rC[0,:-m,m:].data.cpu().numpy().T
        iU = self._iC[0,:-m,m:].data.cpu().numpy().T
        return rU + 1j*iU

    def matmul(self, other):
        ''' Matrix multiplication of the network with a numpy array '''
        # we first check if the numpy array is 2D:
        if other.ndim != 2:
            raise ValueError('Matrix Multiplication of a Unitary Matrix Network with a numpy'
                             'array expects a 2D array.')
        if other.shape[0] != self.num_sources:
            raise ValueError('Shape mismatch between Unitary Matrix Network and numpy array')

        # Initialize with simplified environment
        env = self.env
        self.initialize(self.env.copy(use_delays=False,num_wl=1, num_timesteps=1, num_batches=other.shape[1]))

        # Create Source
        source = np.zeros((2,1,1,self.nmc,other.shape[1]))
        source[0,0,0,:other.shape[0],:] = np.real(other)
        source[1,0,0,:other.shape[0],:] = np.imag(other)
        source = self.new_variable(source)

        # Propagate
        result = self(source, power=False)[:,0,0,:,:].data.cpu().numpy()

        # Initialize with old environment
        self.initialize(env)

        # Return
        return result[0] + 1j*result[1]

    def __matmul__(self, other):
        ''' @ operator for python 3 '''
        return self.matmul(other)


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

        M, N = shape

        term = Term()
        U = UnitaryMatrixNetwork((M,M), dc, wg, name='Unetwork')
        V = UnitaryMatrixNetwork((N,N), dc, wg, name='Vnetwork')

        k = ord('a')
        U_in = ''.join(chr(k) for k in range(k, k+M))
        U_out = ''.join(chr(k) for k in range(k+M, k+2*M))
        V_in = ''.join(chr(k) for k in range(k+2*M, k+2*M+N))
        V_out = ''.join(chr(k) for k in range(k+2*M+N, k+2*M+2*N))

        U = U[U_in+U_out]
        V = V[V_in+V_out]

        if M > N:
            for i in range(M-N):
                U = U * term[U_out[-1-i]]
            U_out = U_out[:-(M-N)]
        elif M < N:
            for i in range(N-M):
                V = V * term[V_out[-1-i]]
            V_out = V_out[:-(N-M)]

        conn = U*V

        for u,v in zip(U_out, V_out):
            new_soa = soa.copy()
            new_soa.name = 'soa'
            conn = conn * new_soa[u+v]

        super(MatrixNetwork, self).__init__(conn, name=name)
