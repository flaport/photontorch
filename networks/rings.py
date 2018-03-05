''' Ring Networks '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np

## Relative
from .network import Network
from .directionalcouplers import DirectionalCouplerNetwork

from ..environment.environment import Environment
from ..components.terms import Term, Detector, Source
from ..components.waveguides import Waveguide
from ..components.directionalcouplers import DirectionalCoupler
from ..components.connection import Connection

from ..torch_ext.nn import Buffer


#####################
## All Pass Filter ##
#####################

class AllPass(Network):
    r''' All Pass Filter

    An AllPass filter is a memory-containging component with one input and one output.

    Connections:
        allpass['ij']
             ___
            /   \
            \___/
        i-----------j
    '''
    def __init__(self,
                 dc,
                 ring_wg,
                 in_wg=None,
                 pass_wg=None,
                 in_term=None,
                 pass_term=None,
                 name=None):
        '''
        AllPass Filter Initialization

        Args:
            ring_wg : waveguide for the ring
            dc : directional coupler for the connection to the ring
        '''
        connector = dc['ikjl']*ring_wg['jl']

        i, j = connector.idxs
        if in_wg is not None:
            connector = in_wg['a'+i]*connector
        if pass_wg is not None:
            connector = pass_wg['b'+j]*connector

        i, j = connector.idxs
        if in_term is not None:
            connector = in_term[i]*connector
        if pass_term is not None:
            connector = pass_term[j]*connector

        Network.__init__(self, connector, name=name)


#####################
## Add Drop Filter ##
#####################

class AddDrop(Network):
    r''' Add Drop Filter

    An AddDrop filter is a memory-containging component with one input and one output.

    Connections:
        adddrop['ijkl']
        j----===----l
            /   \
            \___/
        i-----------k
    '''
    def __init__(self,
                 dc1,
                 dc2,
                 half_ring_wg,
                 in_wg=None,
                 pass_wg=None,
                 add_wg=None,
                 drop_wg=None,
                 in_term=None,
                 pass_term=None,
                 add_term=None,
                 drop_term=None,
                 name=None):
        '''
        AddDrop Filter Initialization

        Parameters
        ----------
        half_ring_wg : waveguide for a half ring (will be used twice to make the network)
        dc1 : bottom directional coupler for the connection to the ring
        dc2 : top directional coupler for the connection to the ring
        '''
        connector = dc1['abcd']*dc2['efgh']
        connector = half_ring_wg['ce']*half_ring_wg['df']*connector

        for i, j, wg in zip('wxyz', connector.idxs, [in_wg, pass_wg, drop_wg, add_wg]):
            if wg is not None:
                connector = wg[i+j]*connector

        for i, term in zip(connector.idxs, [in_term, pass_term, drop_term, add_term]):
            if term is not None:
                connector = term[i]*connector

        Network.__init__(self, connector, name=name)


class RingNetwork(DirectionalCouplerNetwork):
    ''' A network of rings made out of directional couplers and waveguides.
    A directional coupler is repeated periodically in a grid, with the ports flipped
    in odd locations to make rings in the network. The ports are connected by waveguides

    Network:
         0    1    2    3
        ..   ..   ..   ..
         1    1    1    1
    11: 0 2--2 0--0 2--2 0 : 4
         3    3    3    3
         |    |    |    |
         3    3    3    3
    10: 0 2--2 0--0 2--2 0 : 5
         1    1    1    1
        ..   ..   ..   ..
         9    8    7    6

    Legend:
        0: -> term locations
        1
       0 2 -> directional coupler
        3
       -- or | -> waveguides

    Note:
        Because of the connection order of the directional coupler (0<->1 and 2<->3), this
        network does contain loops and can thus be used as a reservoir.
    '''
    def __init__(self,
                 shape,
                 dc,
                 wg,
                 couplings=None,
                 lengths=None,
                 phases=None,
                 terms=None,
                 name='dcnw'):
        DirectionalCouplerNetwork.__init__(
            self,
            shape,
            dc,
            wg,
            couplings=couplings,
            lengths=lengths,
            phases=phases,
            terms=terms,
            name=name,
        )
        ''' RingNetwork

        Args:
            shape: (tuple): shape of the network.
            dc (DirectionalCoupler) : master directional coupler. All the directional
                couplers will take over the properties of this directional coupler,
                except when explicitly overwritten.
            wg (Waveguide): a waveguide containing all the properties of a single
                directional coupler arm, such as the length and the effective index.
            couplings=None (np.ndarray): Desired couplings for the grating couplers. This
                array should have the same length as the shape of the network. If None,
                all directional couplers default to the coupling of the master directional
                coupler.
            lengths=None (np.ndarray): Lengths of the directional couplers.
                If None all lengths will be equal to the length of the master
                directional coupler.
            phases=None (np.ndarray): optional. Phases introduced by the directional
                couplers. If None all phases will be coupled to the length of the
                direcional coupler.
            terms (dict): A dictionary with source and Detector locations in the form
                terms = {3:Source(), 15:Detector(), ...}. All not specified port locations
                will get a Connection as term, as to make it possible to connect to
                other networks (while keeping the port order).

        Note:
            The directional coupler and the waveguide will be combined in a
            DirectionalCouplerWithLength. This conversion happens internally and is not
            required up front.

        '''
        # Change default term order to term order of ring network
        I, J = self.shape
        k0 = (I%2 == 0) & (J%2 == 1)
        k1 = not k0
        self._order = Buffer(torch.from_numpy(np.hstack((
            np.arange(1, J, 1), # North row
            [J, J+1] if J%2 else [J+1, J], #North East Corner
            np.arange(J+3, 4+(J-2)+2*(J-2), 2), # East column
            [8+2*(I-2)+2*(J-2)-2+k0, 8+2*(I-2)+2*(J-2)-2+k1], # South-East Corner
            np.arange(4+(J-2)+2*(I-2), 8+2*(I-2)+2*(J-2)-2, 1)[::-1], #South Row
            np.arange(J+2, 4+(J-2)+2*(J-2), 2)[::-1], # left column
            [0], # half of north west corner
        ))).long())

    def _parse_connections(self):
        connections = []
        I, J = self.shape
        for i in range(I):
            for j in range(J):
                if i < I-1:
                    if i%2 == 0:
                        top = (i*J+j, 3)
                        bottom = ((i+1)*J+j, 3)
                    if i%2 == 1:
                        top = (i*J+j, 1)
                        bottom = ((i+1)*J+j, 1)
                    connections.append(top + bottom)
                if j < J-1:
                    if j%2 == 0:
                        left = (i*J+j, 2)
                        right = (i*J+j+1, 2)
                    if j%2 == 1:
                        left = (i*J+j, 0)
                        right = (i*J+j+1, 0)
                    connections.append(left + right)
        return connections

class RingNetworkExample(Network):
    r'''
                   S  D
                   |  |
                0  1  2--3  4--5
                |  |  |  |  |  |
                0  1  2  3  4  5
                |  |  |  |  |  |
         n--23--+--+--+--+--+--+--6--6
         |      |  |  |  |  |  |     |
         m--22--+--+--+--+--+--+--7--7
                |  |  |  |  |  |
         l--21--+--+--+--+--+--+--8--8--T
         |      |  |  |  |  |  |
         k--20--+--+--+--+--+--+--9--9--D
                |  |  |  |  |  |
         j--19--+--+--+--+--+--+--10-a--T
         |      |  |  |  |  |  |
         i--18--+--+--+--+--+--+--11-b--D
                |  |  |  |  |  |
                17 16 15 14 13 12
                h--g  f  e  d  c
                      |  |  |  |
                      D  T  D  T

    0-23 : RingNetwork Output Nodes
    0-a-n : Renamed Network Nodes
    -- or | : connections / waveguides
    + : directional coupler
    S : Source
    D : Detector
    T : Termination

    '''
    def __init__(self, name=None):
        # default environment
        _env = Environment(
            name = 'Default Environment', # name of the simulation environment
            t_start = 0, #[s] Start time of the simulation
            t_end = 1e-9, #[s] End time of the simulation
            dt = 2e-12, #[s] Timestep of the simulation
            wl = 1.55e-6, #[m] Wavelength of the simulation
            neff = 2.86, # Effective index of the waveguides
        )

        # design parameters
        r = np.random.RandomState(seed=0) # Random State for initializing the network lengths
        min_length = 15*_env.dt*_env.c/_env.neff #[m] Minimum length of the waveguides
        max_length = 0.010 #[m] Maximum length of the waveguides

        ## Directional Coupler Network
        dc_nw = RingNetwork(
            shape=(6,6), # shape of the network
            dc=DirectionalCoupler(0.5), # Base Directional Coupler
            wg=Waveguide(length=min_length, neff=_env.neff, length_bounds=(min_length, max_length)), # Base Waveguide
            lengths = min_length + r.rand(6,6)*(max_length-min_length), # override initial waveguide lengths
            name='dcnw',
        )

        # waveguide definition for the outside of the network:
        def wg():
            wg = Waveguide(
                length=2*(min_length + r.rand()*(max_length-min_length)),
                length_bounds = (2*min_length, 2*max_length),
                neff=2.86,
            )
            return wg

        Network.__init__(self,
            (dc_nw, '0123456789abcdefghijklmn'),
            (wg(), '23'),
            (wg(), '45'),
            (wg(), '67'),
            (wg(), 'hg'),
            (wg(), 'ji'),
            (wg(), 'kl'),
            (wg(), 'mn'),
            (Detector(name='f'), 'f'),
            (Detector(name='d'), 'd'),
            (Detector(name='b'), 'b'),
            (Detector(name='9'), '9'),
            (Detector(name='1'), '1'),
            (Source(), '0'),
            name=name,
        )

        # initialize
        self.initialize(_env)
