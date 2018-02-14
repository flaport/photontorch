''' Connection Module '''

#############
## Imports ##
#############

## Relative
from .component import Component


################
## Connection ##
################

class Connection(Component):
    '''
    A connection connects two ports without delays and without reflection.

    Connections
    -----------
    connection['ij']:

    i-j

    '''

    num_ports = 2

    @property
    def rS(self):
        '''
        Real part of the scattering matrix
        shape: (# num wavelengths, # num ports, # num ports)
        '''
        return self.new_variable([[[0, 1], [1, 0]]]*self.env.num_wl)

    @property
    def iS(self):
        '''
        Imag part of the scattering matrix
        shape: (# num wavelengths, # num ports, # num ports)
        '''
        return self.new_variable([[[0, 0], [0, 0]]]*self.env.num_wl)
