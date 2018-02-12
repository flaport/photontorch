''' Mirrors Module '''

#############
## Imports ##
#############

## Relative
from .component import Component


############
## Mirror ##
############

class Mirror(Component):
    '''
    A mirror is a memory-less component with one input and one output.

    A mirror has one trainable parameter: the reflectivity R.

    Connections
    -----------
    mirror['ij']:
        |
    i --|-- j
        |
    '''

    num_ports = 2

    def __init__(self, R=0.5, R_bounds=(0, 1), name=None):
        '''
        Mirror initialization

        Parameters
        ----------
        R : float. Reflectivity of the mirror (between 0 and 1)
        R_bounds : tuple of length 2: Bounds in which to optimize R.
                   If None, R will not be optimized.
        name : str. name of this specific mirror
        '''
        Component.__init__(self, name=name)

        self.R = self.new_bounded_parameter(
            data=R,
            bounds=R_bounds,
            requires_grad=True, # trainable between bounds
        )

    @property
    def rS(self):
        r = self.R**0.5
        S = self.new_variable([[1, 0],
                               [0, 1]])
        return r*S

    @property
    def iS(self):
        t = (1-self.R)**0.5
        S = self.new_variable([[0, 1],
                               [1, 0]])
        return t*S
