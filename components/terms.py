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

## Torch
import torch

## Relative
from .component import Component


#################
## Termination ##
#################
class Term(Component):
    ''' A term is a memory-less component with one single input.

    It terminates an unconnected node.

    Terms:
        --0

    '''

    num_ports = 1

    def get_S(self):
        ''' Scattering matrix with shape: (# wavelengths, # ports, # ports) '''
        return torch.zeros((2,self.env.num_wl,self.num_ports,self.num_ports), device=self.device)

############
## Source ##
############
class Source(Term):
    '''
    A source is a special kind of Term where power is injected in the system

    Terms:
        --0
    '''
    def get_sources_at(self):
        return torch.ones(1, dtype=torch.uint8, device=self.device)


##############
## Detector ##
##############
class Detector(Term):
    '''
    A detector is a Term where the power is saved

    Terms:
        --0
    '''
    def get_detectors_at(self):
        return torch.ones(1, dtype=torch.uint8, device=self.device)
