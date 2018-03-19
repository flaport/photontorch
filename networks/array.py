'''
# Array Network

a network, constructed of an array of Connectors.
'''

#############
## Imports ##
#############

# Torch
import torch

# Others
import numpy as np

# Relative
from .network import Network

###################
## Array Network ##
###################

class ArrayNetwork(Network):
    ''' Array network

    A network consisting of an array of Connectors

    The Connectors should be created by indexing the components with a string
    with the letters 'l','t','r','b','u' or 'd', each signifying which port will
    connect in which direction: left, top, right, bottom, up or down respectively.

    '''
    def __init__(self, array, name=None):
        ''' Array Network

        Args:
            array (np.ndarray(object)): array consisting of the connectors
            name=None (str): name of the network

        Note:
            Python 3 only, since unicode characters are used in intermediate steps.
        '''

        array = np.asarray(array, dtype=object)

        I,J = array.shape

        # create array with connection strings
        connections = np.zeros((I,J), dtype=object) # 2D array of lists or str

        l = 96 # we start with letter 'a'

        for i in range(I):
            for j in range(J):
                s = []
                num_ports = 0 if array[i,j] is None else len(array[i,j].s)
                for k in range(num_ports):
                    l += 1
                    s = s + [l]
                connections[i,j] = s


        # update connection strings

        # t-b connection strings:
        for i in range(I-1):
            for j in range(J):
                if array[i,j] is None or array[i+1,j] is None:
                    continue
                st = array[i,j].s
                sb = array[i+1,j].s
                if 'b' in st and 't' in sb:
                    l += 1
                    ib = st.index('b')
                    it = sb.index('t')
                    connections[i,j][ib] = connections[i+1,j][it] = l


        # l-r connection strings:
        for i in range(I):
            for j in range(J-1):
                if array[i,j] is None or array[i,j+1] is None:
                    continue
                sl = array[i,j].s
                sr = array[i,j+1].s
                if 'r' in sl and 'l' in sr:
                    l += 1
                    ir = sl.index('r')
                    il = sr.index('l')
                    connections[i,j][ir] = connections[i,j+1][il] = l

        for i in range(I):
            for j in range(J):
                connections[i,j] = ''.join([chr(c) for c in connections[i,j]])

        # connections string:
        s = ','.join([s for s in connections.ravel() if s != ''])

        # components list:
        components = [connection.components[0].copy() for connection in array.ravel() if connection is not None]

        # an array with the free ports:
        free_ports = connections.copy()
        k = 96 # we want to start at a.
        for i in range(I):
            for j in range(J):
                for c in free_ports[i,j]:
                    if s.count(c) == 2:
                        free_ports[i,j] = free_ports[i,j].replace(c,'.')
        for i in range(I):
            for j in range(J):
                for c in free_ports[i,j]:
                    if s.count(c) == 1:
                        k += 1
                        free_ports[i,j] = free_ports[i,j].replace(c,chr(k))

        self.free_ports = free_ports

        # initialize parent class:
        super(ArrayNetwork, self).__init__(s, *components)
