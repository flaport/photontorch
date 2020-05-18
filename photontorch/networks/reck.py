"""

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
from ..components.mzis import Mzi
from ..components.mmis import PhaseArray
from ..components.terms import Source, Detector, Term


#############
## Classes ##
#############


class _ReckNxN(Network):
    """ A helper network for ReckMxN """

    def __init__(
        self,
        N=2,
        length=1e-5,
        loss=0,
        neff=2.34,
        ng=3.40,
        wl0=1.55e-6,
        trainable=True,
        name=None,
    ):
        """
        Args:
            N (int): number of input waveguides (= number of output waveguides)
            length (float): length of the waveguides in the network in meters.
            loss (float): loss in the ring (dB/m).
            neff (float): effective index of the waveguides
            ng (float): group index of the waveguides
            wl0 (flota): center wavelength for th effective index in the waveguides
            trainable (bool): make parameters in the network trainable
            name (str): name of the component
        """
        num_mzis = N - 1

        # define components
        components = {}
        for i in range(num_mzis):
            components["mzi%i" % i] = Mzi(
                length=length,
                phi=0,
                theta=np.pi / 4,
                loss=loss,
                neff=neff,
                ng=ng,
                wl0=wl0,
                trainable=trainable,
            )

        # connections between mzis:
        connections = []
        for i in range(num_mzis - 1):
            connections += ["mzi%i:3:mzi%i:1" % (i, i + 1)]

        # input ports:
        for i in range(num_mzis):
            connections += ["mzi%i:0:%i" % (i, i)]
        connections += ["mzi%i:3:%i" % (i, i + 1)]

        # output ports
        connections += ["mzi%i:1:%i" % (0, N)]
        for i in range(num_mzis):
            connections += ["mzi%i:2:%i" % (i, N + i + 1)]

        super(_ReckNxN, self).__init__(components, connections, name=name)


class ReckMxN(Network):
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
                      .---3 1--3 1-- : 7       M
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
        self,
        N=2,
        M=None,
        length=1e-5,
        loss=0,
        neff=2.34,
        ng=3.40,
        wl0=1.55e-6,
        trainable=True,
        name=None,
    ):
        """
        Args:
            N (int): number of output (the network represents an MxN matrix)
            M (int): number of input ports (the network represents an MxN matrix).
                by default, M will be the same as N
            length (float): length of the waveguides in the network in meters.
            loss (float): loss in the ring (dB/m).
            neff (float): effective index of the waveguides
            ng (float): group index of the waveguides
            wl0 (flota): center wavelength for th effective index in the waveguides
            trainable (bool): make parameters in the network trainable
            name (str): name of the component
        """
        if M is None:
            M = N

        if N > M:
            raise ValueError("N<M required")

        if N < 0:
            raise ValueError("N>0 required")

        if M < 2:
            raise ValueError("M>2 required")

        self.N = N
        self.M = M

        # define components
        components = {}
        for n in range(M - N - int(N == 1) + 2, M + 1):
            components["r%i" % n] = _ReckNxN(
                N=n,
                length=length,
                loss=loss,
                neff=neff,
                ng=ng,
                wl0=wl0,
                trainable=trainable,
            )
        components["pa"] = PhaseArray(
            phases=2 * np.pi * np.random.rand(M), length=0, ng=0, trainable=trainable
        )

        # define conections
        connections = []
        for n in range(M - N + 2, M):
            for i in range(n):
                connections += ["r%i:%i:r%i:%i" % (n, n + i, n + 1, i)]

        for i in range(M):
            connections += ["r%i:%i:pa:%i" % (M, M + i, i)]

        super(ReckMxN, self).__init__(components, connections, name=name)

    def terminate(self, term=None, transposed=False):
        """ Terminate open conections with the term of your choice

        Args:
            term: (Term|list|dict): Which term to use. Defaults to Term. If a
                dictionary or list is specified, then one needs to specify as
                many terms as there are open connections.

        Returns:
            terminated network with sources on the left and detectors on the right.
        """

        def _term(i):
            return Term(name="t%i" % i)

        def _source(i):
            return (
                Source(name="s%i" % i) if not transposed else Detector(name="d%i" % i)
            )

        def _detector(i):
            return (
                Detector(name="d%i" % i) if not transposed else Source(name="s%i" % i)
            )

        if term is None:
            term = []
            term += [_term(i) for i in range(self.M - self.N)]
            term += [_source(i) for i in range(self.N)]
            term += [_detector(i) for i in range(self.M)]

        ret = super(ReckMxN, self).terminate(term)
        ret.to(self.device)

        return ret


class ReckNxN(ReckMxN):
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
                      .---3 1--3 1-- : 7       M
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
        self,
        N=2,
        length=1e-5,
        loss=0,
        neff=2.34,
        ng=3.40,
        wl0=1.55e-6,
        trainable=True,
        name=None,
    ):
        """
        N: int = 2: number of outputs/outputs (the network represents an NxN matrix)
        length: float = 1e-5: length of the waveguides in the network,
        loss: float = 0: loss of the waveguides in the network,
        neff: float = 2.34: effective index of the waveguides in the network,
        ng: float = 3.40: group index of the waveguides in the network,
        wl0: float = 1.55e-6: center wavelength of the waveguides in the network,
        trainable: bool = True: makes the MZIs in the network trainable
        name: str = None: the name of the network (default: lowercase classname)
        """
        super(ReckNxN, self).__init__(
            N=N,
            M=N,
            length=length,
            loss=loss,
            neff=neff,
            ng=ng,
            wl0=wl0,
            trainable=trainable,
            name=name,
        )


class ReckMmi(ReckMxN):
    """ An Mmi with weights represented by a unitary matrix network.

    This unitary matrix implementation is based on The paper of M. Reck:
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58

    Network::

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

    """

    pass
