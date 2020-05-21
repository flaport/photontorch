"""
wg_factory=wg_factory,

The reck module implements a unitary matrix network based on the Reck network


Reference:
    https://journals.aps.org/prl/abstract/10.1103/NhysRevLett.73.58

"""
#############
## Imports ##
#############

# other
import numpy as np

# relative
from .network import Network
from .clements import _wg_factory, _mzi_factory

from ..components.mzis import Mzi
from ..components.terms import Source, Detector, Term


#############
## Classes ##
#############


class _ReckNxN(Network):
    """ A helper network for ReckNxN """

    def __init__(
        self, N=2, wg_factory=_wg_factory, mzi_factory=_mzi_factory, name=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            wg_factory (callable): function without arguments which creates the waveguides.
            mzi_factory (callable): function without arguments which creates the MZIs or
                any other general 4-port component with  ports defined anti-clockwise.
            name (str): name of the component

        """
        self.N = N
        num_mzis = N - 1

        # define components
        components = {}
        for i in range(num_mzis):
            components["mzi%i" % i] = mzi_factory()
        components["wg"] = wg_factory()

        # connections between mzis:
        connections = []
        for i in range(num_mzis - 1):
            connections += ["mzi%i:3:mzi%i:1" % (i, i + 1)]
        connections += ["mzi%i:3:wg:1" % (num_mzis - 1)]

        # input ports:
        for i in range(num_mzis):
            connections += ["mzi%i:0:%i" % (i, i)]
        connections += ["wg:0:%i" % (i + 1)]

        # output ports
        connections += ["mzi%i:1:%i" % (0, N)]
        for i in range(num_mzis):
            connections += ["mzi%i:2:%i" % (i, N + i + 1)]

        super(_ReckNxN, self).__init__(components, connections, name=name)


class ReckNxN(Network):
    """ A unitary matrix network based on the Reck Network.

    Network::

                                .--- : 9
                                |
                                |
                                2
                           .---3 1-- : 8
                           |    0
                           |    |
                           2    2
                      .---3 1--3 1-- : 7       N
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

                     N

    Reference:
        https://journals.aps.org/prl/abstract/10.1103/NhysRevLett.73.58


    """

    def __init__(
        self, N=2, wg_factory=_wg_factory, mzi_factory=_mzi_factory, name=None,
    ):
        """
        Args:
            N (int): number of input ports (the network represents an MxN matrix)
            wg_factory (callable): function without arguments which creates the waveguides.
            mzi_factory (callable): function without arguments which creates the MZIs or any other general
                4-port component with  ports defined anti-clockwise.
            name (str): name of the component

        Note:
            ``ReckMxN`` expects ``M >= N``. If M < N is desired, consider terminating with the ``transposed=True`` flag::

                reck7x3 = Reck(3, 7).terminate(transposed=True)
        """
        M = N  # TODO: handle M >= N some time
        self.N = N
        components = {}
        connections = []

        for m in range(M - 1):
            components["layer%i" % m] = _ReckNxN(
                N=N - m, wg_factory=wg_factory, mzi_factory=mzi_factory
            )
        components["wg"] = wg_factory()

        for m in range(1, M - 1):
            for n in range(N - m):
                connections += ["layer%i:%i:layer%i:%i" % (m - 1, N - m + 2 + n, m, n)]
        connections += ["layer%i:%i:wg:0" % (M - 2, N - (M - 2) + 1)]

        super(ReckNxN, self).__init__(components, connections, name=name)

    def terminate(self, term=None):
        """ Terminate open conections with the term of your choice

        Args:
            term: (Term|list|dict): Which term to use. Defaults to Term. If a
                dictionary or list is specified, then one needs to specify as
                many terms as there are open connections.

        Returns:
            terminated network with sources on the bottom and detectors on the right.
        """
        if term is None:
            term = [Source(name="s%i" % i) for i in range(self.N)]
            term += [Detector(name="d%i" % i) for i in range(self.N)]
        ret = super(ReckNxN, self).terminate(term)
        ret.to(self.device)
        return ret
