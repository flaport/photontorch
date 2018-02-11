''' Ring Networks '''

#############
## Imports ##
#############

## Other
import numpy as np

## Relative
from .network import Network
from .directionalcouplers import DirectionalCouplerNetwork
from ..components.waveguides import Waveguide
from ..components.directionalcouplers import DirectionalCoupler
from ..torch_ext.autograd import block_diag
from ..torch_ext.tensor import where


#####################
## All Pass Filter ##
#####################

class AllPass(Network):
    ''' All Pass Filter

    An AllPass filter is a memory-containging component with one input and one output.

    Connections
    -----------
    allpass['ij']
         ___
        /   \
        \___/
    i-----------j
    '''
    def __init__(self,
                 dircoup,
                 ring_wg,
                 in_wg=None,
                 pass_wg=None,
                 in_term=None,
                 pass_term=None,
                 name='allpass'):
        '''
        AllPass Filter Initialization

        Parameters
        ----------
        ring_wg : waveguide for the ring
        dircoup : directional coupler for the connection to the ring
        '''
        connector = dircoup['ikjl']*ring_wg['jl']

        i,j = connector.idxs
        if in_wg is not None:
            connector = in_wg['a'+i]*connector
        if pass_wg is not None:
            connector = pass_wg['b'+j]*connector

        i,j = connector.idxs
        if in_term is not None:
            connector = in_term[i]*connector
        if pass_term is not None:
            connector = pass_term[j]*connector

        Network.__init__(self, connector)


#####################
## Add Drop Filter ##
#####################

class AddDrop(Network):
    ''' Add Drop Filter

    An AddDrop filter is a memory-containging component with one input and one output.

    Connections
    -----------
    adddrop['ijkl']
    j----===----l
        /   \
        \___/
    i-----------k
    '''
    def __init__(self,
                 dircoup1,
                 dircoup2,
                 half_ring_wg,
                 in_wg = None,
                 pass_wg = None,
                 add_wg = None,
                 drop_wg = None,
                 in_term = None,
                 pass_term = None,
                 add_term = None,
                 drop_term = None,
                 name='adddrop'):
        '''
        AddDrop Filter Initialization

        Parameters
        ----------
        half_ring_wg : waveguide for a half ring (will be used twice to make the network)
        dircoup1 : bottom directional coupler for the connection to the ring
        dircoup2 : top directional coupler for the connection to the ring
        '''
        connector = dircoup1['abcd']*dircoup2['efgh']
        connector = half_ring_wg['ce']*half_ring_wg['df']*connector

        for i,j, wg in zip('wxyz',connector.idxs,[in_wg, pass_wg, drop_wg, add_wg]):
            if wg is not None:
                connector = wg[i+j]*connector

        for i, term in zip(connector.idxs,[in_term, pass_term, drop_term, add_term]):
            if term is not None:
                connector = term[i]*connector
        print(connector)

        Network.__init__(self, connector)


class RingNetwork(DirectionalCouplerNetwork):
    ''' A network of rings made out of directional couplers and waveguides.
    A directional coupler is repeated periodically in a grid, with the ports flipped
    in odd locations to make rings in the network. The ports are connected by waveguides

    Network
    -------
         1    2    3    5
        ..   ..   ..   ..
         1    1    1    1
    0 : 0 2--2 0--0 2--2 0 : 4
         3    3    3    3
         |    |    |    |
         3    3    3    3
    6 : 0 2--2 0--0 2--2 0 :10
         1    1    1    1
        ..   ..   ..   ..
         7    8    9   11

    Legend
    ------
        0: -> term locations
        1
       0 2 -> directional coupler
        3
       -- or | -> waveguides

    Note
    ----
    Because of the connection order of the directional coupler (0<->1 and 2<->3), this
    network does contain loops and can thus be used as a reservoir.
    '''

    def _parse_connections(self):
        connections = []
        I,J = self.shape
        for i in range(I):
            for j in range(J):
                if i < I-1:
                    if i%2 == 0:
                        top = (i*J+j,3)
                        bottom = ((i+1)*J+j,3)
                    if i%2 == 1:
                        top = (i*J+j,1)
                        bottom = ((i+1)*J+j,1)
                    connections.append(top + bottom)
                if j < J-1:
                    if j%2 == 0:
                        left = (i*J+j,2)
                        right = (i*J+j+1,2)
                    if j%2 == 1:
                        left = (i*J+j,0)
                        right = (i*J+j+1,0)
                    connections.append(left + right)
        return connections