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

        self.dcs = dcs = OrderedDict()
        self.hwgs = hwgs = OrderedDict()
        self.vwgs = vwgs = OrderedDict()

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

        self.conn = self.dicts2connector(dcs, hwgs, vwgs)

        super(UnitaryMatrixNetwork, self).__init__(self.conn, name=name)

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

        new.conn = cls.dicts2connector(new.dcs, new.hwgs, new.vwgs)

        super(UnitaryMatrixNetwork, new).__init__(new.conn, name=name)

        return new

    @staticmethod
    def dicts2connector(dcs, hwgs, vwgs):
        dc_list = list(dcs.values())
        vwgs_list = list(vwgs.values())
        hwgs_list = list(hwgs.values())

        hwgs_list = [hwgs_list.pop(-1)] + hwgs_list
        conn = dc_list.pop(0)
        for c in dc_list+vwgs_list+hwgs_list:
            conn = conn*c

        return conn

    def copy(self):
        for dc in self.dcs.values():
            dc = dc.components[0]
            break
        for wg in self.hwgs.values():
            wg = wg.components[0]
            break
        new = self.__class__(self.shape, dc, wg, name=self.name)
        new.dcs = self.copydict(self.dcs)
        new.hwgs = self.copydict(self.hwgs)
        new.vwgs = self.copydict(self.vwgs)
        new.input_s = self.input_s
        new.output_s = self.output_s

        new.conn = self.dicts2connector(new.dcs, new.hwgs, new.vwgs)

        super(UnitaryMatrixNetwork, new).__init__(new.conn, name=self.name)

        return new

    def terminate(self):
        '''
        A Matrix Network is terminated by Sources on the bottom and detectors on the right
        '''
        is_cuda = self.is_cuda

        def maybe_cuda(comp):
            return comp if not is_cuda else comp.cuda()

        if self.terminated:
            return # Do nothing
        for c in self.input_s:
            self.conn = self.conn*maybe_cuda(Source(name=c))[c]
        for c in self.output_s:
            self.conn = self.conn*maybe_cuda(Detector(name=c))[c]
        super(UnitaryMatrixNetwork, self).__init__(self.conn, name=self.name)

        self.is_cuda = is_cuda

        return self

    def unterminate(self):
        ''' Undo de termination of the network '''
        is_cuda = self.is_cuda
        if not self.terminated:
            return # do nothing
        s = self.conn.s.split(',')
        components = self.conn.components
        s, self.components = zip(*[(ss, comp) for (ss, comp) in zip(s, components) if not isinstance(comp, Term)])
        self.s = ','.join(s)
        self.conn = Connector(self.s, self.components)
        self.hwgs = OrderedDict((k,v) for k,v in self.hwgs.items() if not isinstance(v, Term))
        self.vwgs = OrderedDict((k,v) for k,v in self.vwgs.items() if not isinstance(v, Term))
        self.dcs = OrderedDict((k,v) for k,v in self.dcs.items() if not isinstance(v, Term))
        self.is_cuda = is_cuda
        return self

    def split(self, m, terminate=None):
        ''' Split the network in an upper half and a lower half at output m'''
        M, _ = self.shape

        if m > M-1:
            raise ValueError('cannot create a submatrix that is bigger than the original')

        def copy(conn):
            ''' shallow copy of a single component connector '''
            return conn.components[0][conn.s]

        # lower half of matrix
        hwgs = OrderedDict((k,copy(v)) for k,v in self.hwgs.items() if k[0]>m-1)
        vwgs = OrderedDict((k,copy(v)) for k,v in self.vwgs.items() if k[0]>m-2)
        dcs = OrderedDict((k,copy(v)) for k,v in self.dcs.items() if k[0]>m-2)

        for i, c in enumerate(self.output_s[:m]):
            s = dcs[m-1,i].s
            dcs[m-1,i].s = s[0] + s[1] + c + s[3]

        name = '_'.join([self.name.split('_')[0], 'btm%i'%m])
        btm = self.__class__.from_dicts(self.shape, dcs, hwgs, vwgs, name=name)

        top = None
        if m > 1:
            # upper half of matrix
            hwgs = OrderedDict((k,copy(v)) for k,v in self.hwgs.items() if k[0]<=m-1)
            vwgs = OrderedDict((k,copy(v)) for k,v in self.vwgs.items() if k[0]<=m-2)
            dcs = OrderedDict((k,copy(v)) for k,v in self.dcs.items() if k[0]<=m-2)

            for i, c in enumerate(self.input_s[:m-1]):
                s = vwgs[m-2, i].s
                vwgs[m-2,i].s = s[0] + c

            c = self.input_s[-2]
            s = hwgs[m-1,m-1].s
            hwgs[m-1,m-1].s = s[0] + c

            i = ord(c) + 1
            for k, conn in hwgs.items():
                if k[1] == 0:
                    conn.s = conn.s[0] + chr(i)
                    i += 1

            name = '_'.join([self.name.split('_')[0], 'top%i'%m])
            top = self.__class__.from_dicts((m,m), dcs, hwgs, vwgs, name=name)

        if self.is_cuda:
            if top is not None:
                top.is_cuda = True
            btm.is_cuda = True
        if terminate is None:
            terminate = self.terminated
        if terminate:
            top = top if top is None else top.terminate()
            btm = btm.terminate()
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

    def fit(self, U, max_epochs=1500, tol=1e-7, progress_bar=True, optimizer=None, lossfunc=None):
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

        # Terminate if necessary:
        terminated = self.terminated
        if not terminated:
            self.terminate()

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
            env = env.copy(num_batches=k)
            top.initialize(env)
            source = new_source(top, U)
            target = top.new_variable(np.eye(k))
            top, btm = top.split(k-1)
            for i in range(max_epochs):
                btm.initialize(env)
                # Note: in general, using sqrt on top of the prediction
                # yields better results faster, however, there is a considerable
                # higher chance on nan's.
                prediction = btm(source)[0,0,:,k-1:k]
                loss = lossfunc(prediction, target[:,k-1:k])
                r.set_postfix(loss='{loss.data[0]:.{x}f}'.format(loss=loss, x=int(-np.log10(tol)+0.5)))
                if (loss != loss).any(): # this only happens when loss == nan
                    del prediction, loss
                    return
                loss.backward()
                optimizer.step()
                if (loss < tol).all():
                    del prediction, loss
                    break
                del prediction, loss

            # Update U:
            U = (btm.matmul(U))[:-1, :-1]

        # Loop is over. We now handle the k==2 case:
        env = env.copy(num_batches=2)
        top.initialize(env)
        source = new_source(top, U)
        target = top.new_variable(np.eye(2))
        for i in range(max_epochs):
            top.initialize(env)
            prediction = top(source)[0,0,:,:]
            loss = lossfunc(prediction, target)
            r.set_postfix(loss='{loss.data[0]:.{x}f}'.format(loss=loss, x=int(-np.log10(tol)+0.5)))
            if (loss != loss).any(): # this only happens when loss == nan
                del prediction, loss
                break
            loss.backward()
            optimizer.step()
            if (loss < tol).all():
                del prediction, loss
                break

        # Last step is to adjust the output phase:
        # Replace U.T.conj() by the matrix represented by the network?
        out = np.diag(self.matmul(U0.T.conj()))
        phase_correction = np.arctan2(np.imag(out), np.real(out))
        phases = [conn.components[0].phase.data for k, conn in self.hwgs.items() if k[1] == 0]
        for phase, pc in zip(phases, phase_correction):
            phase[0] -= pc

        # Unterminate if necessary
        if not terminated:
            self.unterminate()


    @property
    def U(self):
        ''' Matrix represented by this matrix network '''
        return self.matmul(np.eye(*self.shape))

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
        source = self.new_variable(source)

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
        for dc in self.Vnw.dcs.values():
            dc = dc.components[0]
            break
        for wg in self.Vnw.hwgs.values():
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

        self.Vnw.fit(
            U=V,
            max_epochs=max_epochs,
            tol=tol,
            progress_bar=progress_bar,
            optimizer=optimizer,
            lossfunc=lossfunc,
        )

        self.Unw.fit(
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
        source = self.new_variable(source)

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
