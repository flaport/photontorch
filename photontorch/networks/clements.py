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
from ..components.waveguides import Waveguide
from ..components.mzis import Mzi
from ..components.mmis import PhaseArray
from ..components.terms import Source, Detector


#############
## Classes ##
#############


class _MixingPhaseArray(Network):
    """ helper network for ClementsNxN """

    def __init__(
        self,
        phases,
        length=1e-5,
        loss=0,
        neff=2.34,
        ng=None,
        wl0=1.55e-6,
        trainable=True,
        name=None,
    ):
        """
        Args:
            phases: array of phases to implement with photonic components.
            length (float): length of the waveguides in the network in meters.
            loss (float): loss in the ring (dB/m).
            neff (float): effective index of the waveguides
            ng (float): group index of the waveguides
            wl0 (flota): center wavelength for th effective index in the waveguides
            trainable (bool): make parameters in the network trainable
            name (str): name of the component
        """
        N = phases.shape[0]
        num_mzis = N // 2
        components = {}
        components["pa"] = PhaseArray(
            phases=phases, length=0, ng=0, trainable=trainable
        )

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

        connections = []
        for i in range(num_mzis):
            connections += ["mzi%i:1:pa:%i" % (i, 2 * i)]
            connections += ["mzi%i:2:pa:%i" % (i, 2 * i + 1)]

        # input connections:
        for i in range(0, num_mzis):
            connections += ["mzi%i:0:%i" % (i, 2 * i)]
            connections += ["mzi%i:3:%i" % (i, 2 * i + 1)]
        if N % 2:
            connections += ["pa:%i:%i" % (N - 1, N - 1)]

        super(_MixingPhaseArray, self).__init__(components, connections, name=name)


class _Capacity2ClemensNxN(Network):
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
        self,
        N=2,
        length=1e-5,
        loss=0,
        neff=2.34,
        ng=None,
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

        components["wg0"] = components["wg1"] = Waveguide(
            length=length,
            phase=0,
            loss=loss,
            neff=neff,
            ng=ng,
            wl0=wl0,
            trainable=False,
        )

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
        super(_Capacity2ClemensNxN, self).__init__(components, connections, name=name)


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
            __[]__ = phase shift
            __  __
              \/   =  MZI
            __/\__

    Reference:
        https://www.osapublishing.org/optica/abstract.cfm?uri=optica-3-12-1460

    """

    def __init__(
        self,
        N=2,
        capacity=None,
        length=1e-5,
        loss=0,
        neff=2.34,
        ng=3.4,
        wl0=1.55e-6,
        trainable=True,
        name=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            length (float): length of the waveguides in the network,
            loss (float): loss of the waveguides in the network,
            neff (float): effective index of the waveguides in the network,
            ng (float): group index of the waveguides in the network,
            wl0 (float): center wavelength of the waveguides in the network,
            trainable (bool): makes the MZIs in the network trainable
            name (str): the name of the network (default: lowercase classname)
        """
        if capacity is None:
            capacity = N

        self.N = N
        self.capacity = capacity

        # create components
        components = {}
        for i in range(capacity // 2):
            components["layer%i" % i] = _Capacity2ClemensNxN(
                N=N,
                length=length,
                loss=loss,
                neff=neff,
                ng=ng,
                wl0=wl0,
                trainable=trainable,
            )
        if capacity % 2 == 0:
            components["layer%i" % (capacity // 2)] = PhaseArray(
                phases=2 * np.pi * np.random.rand(N),
                length=0,
                ng=0,
                trainable=trainable,
            )
        else:
            components["layer%i" % (capacity // 2)] = _MixingPhaseArray(
                phases=2 * np.pi * np.random.rand(N),
                length=length,
                loss=loss,
                neff=neff,
                ng=ng,
                wl0=wl0,
                trainable=trainable,
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
