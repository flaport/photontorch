""" A collection of ring networks """

#############
## Imports ##
#############

## Standard Library
from copy import copy, deepcopy
from collections import OrderedDict

## Torch
import torch

## Other
import numpy as np

## Relative
from .network import Network
from .clements import _wg_factory, _mzi_factory, _PhaseArray

from ..components.terms import Detector, Source
from ..components.mzis import Mzi
from ..components.waveguides import Waveguide
from ..components.directionalcouplers import DirectionalCoupler
from ..torch_ext.nn import Buffer, Parameter
from ..torch_ext import block_diag


###################
## Ring Networks ##
###################


class _MixingPhaseArray(Network):
    """ helper network for RingNetwork """

    def __init__(
        self, N, wg_factory=_wg_factory, mzi_factory=_mzi_factory, name=None,
    ):
        """
        Args:
            N (int): number of input / output ports (the network represents an NxN matrix)
            wg_factory (callable): function without arguments which creates the waveguides.
            mzi_factory (callable): function without arguments which creates the MZIs or 
                any other general 4-port component with  ports defined anti-clockwise.
            name (str): name of the component
        """
        self.N = int(N + 0.5)
        if self.N % 2:
            raise ValueError("the number of phases should be even")
        num_mzis = self.N // 2
        components = {}

        for i in range(self.N):
            components["wg%i" % i] = wg_factory()

        for i in range(num_mzis):
            components["mzi%i" % i] = mzi_factory()

        connections = []
        for i in range(num_mzis):
            connections += ["mzi%i:3:wg%i:0" % (i, 2 * i)]
            connections += ["mzi%i:2:wg%i:0" % (i, 2 * i + 1)]

        # input connections:
        for i in range(0, num_mzis):
            connections += ["mzi%i:0:%i" % (i, 2 * i)]
            connections += ["mzi%i:1:%i" % (i, 2 * i + 1)]

        # output connections:
        for i in range(self.N):
            connections += ["wg%i:1:%i" % (i, self.N + i)]

        super(_MixingPhaseArray, self).__init__(components, connections, name=name)


class _UnclosedRingArray(Network):
    r""" Helper network for RingNetwork::

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
            mzi_factory (callable): function without arguments which creates the MZIs or 
                any other general 4-port component with  ports defined anti-clockwise.
            name (str): name of the component
        """

        if N % 2:
            raise ValueError("hidden size should be even")

        num_rings = N // 2

        num_mzis = 2 * num_rings - 1

        # define components
        components = {}
        for i in range(2):
            components["wg%i" % i] = wg_factory()
        for i in range(num_mzis):
            components["mzi%i" % i] = mzi_factory()

        # connections between mzis:
        connections = ["wg0:0:mzi0:3"]
        for i in range(1, num_mzis, 2):
            connections += ["mzi%i:0:mzi%i:2" % (i, i - 1)]
            connections += ["mzi%i:3:mzi%i:3" % (i, i + 1)]
        connections += ["wg1:0:mzi%i:2" % (num_mzis - 1)]

        # input connections:
        for i in range(0, num_mzis, 2):
            connections += ["mzi%i:0:%i" % (i, i)]
            connections += ["mzi%i:1:%i" % (i, i + 1)]

        # output connections:
        k = 2 * num_rings
        connections += ["wg0:1:%i" % k]
        for i in range(1, num_mzis, 2):
            connections += ["mzi%i:1:%i" % (i, k + i)]
            connections += ["mzi%i:2:%i" % (i, k + i + 1)]
        connections += ["wg1:1:%i" % (4 * num_rings - 1)]

        # initialize network
        super(_UnclosedRingArray, self).__init__(components, connections, name=name)


class RingNetwork(Network):
    r""" A ring network

    By changing the orientation of some of the MZIs in the Clements Network, a ring
    network can be obtained.

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

    """

    def __init__(
        self,
        N=2,
        capacity=2,
        wg_factory=_wg_factory,
        mzi_factory=_mzi_factory,
        name=None,
    ):
        """
        Args:
            N (int): number of input waveguides (= number of output waveguides)
            capacity (int): number of consecutive MZI layers (1 more than the consequtive number of ring layers)
            wg_factory (callable): function without arguments which creates the waveguides.
            mzi_factory (callable): function without arguments which creates the MZIs or 
                any other general 4-port component with  ports defined anti-clockwise.
            name (str): name of the component
        """
        self.N = N
        self.capacity = capacity

        if N % 2:
            raise ValueError(
                "the number of inputs/outputs of a ring network should be even"
            )

        # create components
        components = {}
        for i in range(capacity // 2):
            components["layer%i" % i] = _UnclosedRingArray(
                N=N, mzi_factory=mzi_factory, wg_factory=wg_factory
            )
        components["layer%i" % (capacity // 2)] = _MixingPhaseArray(
            N=N, mzi_factory=mzi_factory, wg_factory=wg_factory
        )

        # create connections
        connections = []
        for i in range(capacity // 2):
            for j in range(N):
                connections += ["layer%i:%i:layer%i:%i" % (i, N + j, i + 1, j)]

        # initialize network
        super(RingNetwork, self).__init__(components, connections, name=name)

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
        ret = super(RingNetwork, self).terminate(term)
        ret.to(self.device)
        return ret
