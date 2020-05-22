""" unitary matrix network based on the Clements network


Reference:
    https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743

"""
#############
## Imports ##
#############

# other
import numpy as np

# relative
from .network import Network
from ..components.mzis import Mzi
from ..components.waveguides import Waveguide
from ..components.terms import Source, Detector


#############
## Classes ##
#############


def _wg_factory():
    return Waveguide(phase=2 * np.pi * np.random.rand(), trainable=True)


def _mzi_factory():
    return Mzi(
        phi=2 * np.pi * np.random.rand(),
        theta=2 * np.pi * np.random.rand(),
        trainable=True,
    )


class _PhaseArray(Network):
    """ helper network for ClementsNxN """

    def __init__(
        self, N, wg_factory=_wg_factory, name=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            wg_factory (callable): function without arguments which creates the waveguides.
            name (optional, str): name of the component
        """
        self.N = int(N + 0.5)
        components = {}
        connections = []

        for i in range(self.N):
            components["wg%i" % i] = wg_factory()

        # input/output connections:
        for i in range(self.N):
            connections += ["wg%i:0:%i" % (i, i)]
            connections += ["wg%i:1:%i" % (i, self.N + i)]

        super(_PhaseArray, self).__init__(components, connections, name=name)


class _MixingPhaseArray(Network):
    """ helper network for ClementsNxN """

    def __init__(
        self, N, wg_factory=_wg_factory, mzi_factory=_mzi_factory, name=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            wg_factory (callable): function without arguments which creates the waveguides.
            mzi_factory (callable): function without arguments which creates the MZIs or any other general 
                4-port component with  ports defined anti-clockwise.
            name (optional, str): name of the component
        """
        self.N = int(N + 0.5)
        num_mzis = self.N // 2
        components = {}

        for i in range(self.N):
            components["wg%i" % i] = wg_factory()

        for i in range(num_mzis):
            components["mzi%i" % i] = mzi_factory()
        if self.N % 2:
            components["wg_"] = wg_factory()

        connections = []
        for i in range(num_mzis):
            connections += ["mzi%i:1:wg%i:0" % (i, 2 * i)]
            connections += ["mzi%i:2:wg%i:0" % (i, 2 * i + 1)]
        if self.N % 2:
            connections += ["wg_:1:wg%i:0" % (self.N - 1)]

        # input connections:
        for i in range(0, num_mzis):
            connections += ["mzi%i:0:%i" % (i, 2 * i)]
            connections += ["mzi%i:3:%i" % (i, 2 * i + 1)]
        if self.N % 2:
            connections += ["wg_:0:%i" % (self.N - 1)]

        # output connections:
        for i in range(self.N):
            connections += ["wg%i:1:%i" % (i, self.N + i)]

        super(_MixingPhaseArray, self).__init__(components, connections, name=name)


class _Capacity2ClementsNxN(Network):
    r""" Helper network for ClementsNxN::

        <- cap==2 ->
        0__  ______0
           \/
        1__/\__  __1
               \/
        2__  __/\__2
           \/
        3__/\______3

    """

    def __init__(
        self, N=2, wg_factory=_wg_factory, mzi_factory=_mzi_factory, name=None,
    ):
        """
        Args:
            N (int): number of input waveguides (= number of output waveguides)
            wg_factory (callable): function without arguments which creates the waveguides.
            mzi_factory (callable): function without arguments which creates the MZIs or any other general 
                4-port component with  ports defined anti-clockwise.
            name (optional, str): name of the component
        """
        num_mzis = N - 1

        # define components
        components = {}
        for i in range(num_mzis):
            components["mzi%i" % i] = mzi_factory()

        components["wg0"] = components["wg1"] = wg_factory()

        # connections between mzis:
        connections = []
        connections += ["mzi0:1:wg0:0"]
        for i in range(1, num_mzis - 1, 2):
            connections += ["mzi%i:2:mzi%i:0" % ((i - 1), i)]
            connections += ["mzi%i:3:mzi%i:1" % (i, (i + 1))]
        if num_mzis > 1 and N % 2:
            connections += ["mzi%i:2:mzi%i:0" % (num_mzis - 2, num_mzis - 1)]
        if N % 2:
            connections += ["wg1:1:mzi%i:3" % (N - 2)]
        else:
            connections += ["mzi%i:2:wg1:0" % (N - 2)]

        # input connections:
        for i in range(0, num_mzis, 2):
            connections += ["mzi%i:0:%i" % (i, i)]
            connections += ["mzi%i:3:%i" % (i, i + 1)]
        if N % 2:
            connections += ["wg1:0:%i" % (N - 1)]

        # output connections:
        k = i + 2 + N % 2
        connections += ["wg0:1:%i" % k]
        for i in range(1, num_mzis, 2):
            connections += ["mzi%i:1:%i" % (i, k + i)]
            connections += ["mzi%i:2:%i" % (i, k + i + 1)]
        if N % 2 == 0:
            connections += ["wg1:1:%i" % (2 * N - 1)]

        # initialize network
        super(_Capacity2ClementsNxN, self).__init__(components, connections, name=name)


class ClementsNxN(Network):
    r""" A unitary matrix network based on the Clements architecture.

    Network::

         <--- capacity --->
        0__  ______  ______[]__0
           \/      \/
        1__/\__  __/\__  __[]__1
               \/      \/
        2__  __/\__  __/\__[]__2
           \/      \/
        3__/\______/\______[]__3

        with:
        
           0__[]__1 = phase shift
           
           3__  __2
              \/    =  MZI
           0__/\__1

    Reference:
        https://www.osapublishing.org/optica/abstract.cfm?uri=optica-3-12-1460

    """

    def __init__(
        self,
        N=2,
        capacity=None,
        wg_factory=_wg_factory,
        mzi_factory=_mzi_factory,
        name=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            capacity (int): number of consecutive MZI layers (to span the full unitary space one needs capacity >=N).
            wg_factory (callable): function without arguments which creates the waveguides.
            mzi_factory (callable): function without arguments which creates the MZIs or any other general 
                4-port component with  ports defined anti-clockwise.
            name (optional, str): the name of the network (default: lowercase classname)
        """
        if capacity is None:
            capacity = N

        self.N = N
        self.capacity = capacity

        # create components
        components = {}
        for i in range(capacity // 2):
            components["layer%i" % i] = _Capacity2ClementsNxN(
                N=N, mzi_factory=mzi_factory, wg_factory=wg_factory,
            )
        if capacity % 2 == 0:
            components["layer%i" % (capacity // 2)] = _PhaseArray(
                self.N, wg_factory=wg_factory
            )
        else:
            components["layer%i" % (capacity // 2)] = _MixingPhaseArray(
                self.N, mzi_factory=mzi_factory, wg_factory=wg_factory,
            )

        # create connections
        connections = []
        for i in range(capacity // 2):
            for j in range(N):
                connections += ["layer%i:%i:layer%i:%i" % (i, N + j, i + 1, j)]

        # initialize network
        super(ClementsNxN, self).__init__(components, connections, name=name)

    def terminate(self, term=None):
        """ Terminate open conections with the term of your choice

        Args:
            term: (Term|list|dict): Which term to use. Defaults to Term. If a
                dictionary or list is specified, then one needs to specify as
                many terms as there are open connections.

        Returns:
            terminated network with sources on the left and detectors on the right.
        """
        if term is None:
            term = [Source(name="s%i" % i) for i in range(self.N)]
            term += [Detector(name="d%i" % i) for i in range(self.N)]
        ret = super(ClementsNxN, self).terminate(term)
        ret.to(self.device)
        return ret
