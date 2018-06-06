'''
# Connection Module

A Connection is a special kind of component that connects two ports without reflection.

It is generally recommended to make connections with the connection matrix, but if it's
not possible to do so directly, ac Connection object can be used.

This object can for example be used as a Term for a directional coupler network, such
that the directional coupler network can be connected to other components in a higher
level Network.

'''

#############
## Imports ##
#############

## Relative
from .component import Component


################
## Connection ##
################

class Connection(Component):
    ''' A connection connects two ports without delays and without reflection.

    Connections:
        connection['ij']:

        i-j

    Note:
        It is generally recommended to make connections with the connection matrix, but if it's
        not possible to do so directly, ac Connection object can be used.

    '''

    num_ports = 2

    def get_rS(self):
        ''' Real part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        return self.tensor([[[0, 1], [1, 0]]]*self.env.num_wl)

    def get_iS(self):
        ''' Imag part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        return self.tensor([[[0, 0], [0, 0]]]*self.env.num_wl)
