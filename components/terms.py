''' Module with network terminations. Network terminations can be absorbing (Term), a source (Source) or absorbing and detecting (Detector) '''

#############
## Imports ##
#############

## Other
import numpy as np

## Relative
from .component import Component


#################
## Termination ##
#################
class Term(Component):
    '''
    A term is a memory-less component with one single input. It terminates an unconnected node.
    '''

    num_ports = 1

    @property
    def rS(self):
        return self.new_variable([[0]], 'float')
    @property
    def iS(self):
        return self.new_variable([[0]], 'float')

############
## Source ##
############
class Source(Term):
    '''
    A source is a special kind of Term where power is injected in the system
    '''
    @property
    def sources_at(self):
        return self.new_variable([1], 'byte')


##############
## Detector ##
##############
class Detector(Term):
    '''
    A detector is a Term where the power is saved
    '''
    @property
    def detectors_at(self):
        return self.new_variable([1], 'byte')
