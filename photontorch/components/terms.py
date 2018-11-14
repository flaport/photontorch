"""
# Terminations

In order to be able to simulate a Network, the connection matrix should be fully connected.

Ports that are not connected to other components, should be connected to a Term.

Network terminations can be absorbing (Term), injecting (Source) or absorbing and
detecting (Detector)
"""

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
    """ A term is a memory-less component with a single input.

    It terminates an unconnected node.

    Terms:
        --0

    """

    num_ports = 1


############
## Source ##
############
class Source(Term):
    """
    A source is a special kind of Term where power is injected in the system

    Terms:
        --0
    """

    def set_sources_at(self, sources_at):
        sources_at[:] = 1


##############
## Detector ##
##############
class Detector(Term):
    """
    A detector is a Term where the power is saved

    Terms:
        --0
    """

    def set_detectors_at(self, detectors_at):
        detectors_at[:] = 1
