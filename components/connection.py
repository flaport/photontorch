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
        return self.new_variable([[0, 1], [1, 0]])

    @property
    def iS(self):
        return self.new_variable([[0, 0], [0, 0]])
