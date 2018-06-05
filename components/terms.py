'''
# Terminations

In order to be able to simulate a Network, the connection matrix should be fully connected.

Ports that are not connected to other components, should be connected to a Term.

Network terminations can be absorbing (Term), injecting (Source) or absorbing and
detecting (Detector)
'''

#############
## Imports ##
#############

## Relative
from .component import Component


#################
## Termination ##
#################
class Term(Component):
    ''' A term is a memory-less component with one single input.

    It terminates an unconnected node.

    '''

    num_ports = 1

    @property
    def rS(self):
        ''' Real part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        return self.tensor([[[0]]]*self.env.num_wl, 'float')
    @property
    def iS(self):
        ''' Imag part of the scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        return self.tensor([[[0]]]*self.env.num_wl, 'float')

############
## Source ##
############
class Source(Term):
    '''
    A source is a special kind of Term where power is injected in the system
    '''
    @property
    def sources_at(self):
        return self.tensor([1], 'byte')


##############
## Detector ##
##############
class Detector(Term):
    '''
    A detector is a Term where the power is saved
    '''
    @property
    def detectors_at(self):
        return self.tensor([1], 'byte')
