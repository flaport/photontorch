'''

# Two port component networks

Networks that consists solely of two-port components

'''


#############
## Imports ##
#############

# Standard Library
from copy import copy
from collections import OrderedDict

# Torch
import torch

# Other
import numpy as np

# Relative
from .network import Network
from ..components.terms import Source, Detector
from ..torch_ext.nn import Module, Buffer

######################
## Two Port Network ##
######################


class TwoPortNetwork(Network):
    ''' This class is used to define a network based on only two-port components. '''

    def __init__(self, twoportcomponents, conn_matrix, sources_at=None,
                 detectors_at=None, delays=None, name=None):
        '''
        Args:
            twoportcomponents (array|list): A list of components with just two ports
            conn_matrix (array[float]): a square 2D array where conn_matrix[i,j] means a
                connection from component i to component j
            sources_at (array|list[bool]): A list of locations of the sources
            detectors_at (array|list[bool]): A list of locations of the detectors
            delays (array|list[float]): Overrides the delays of the twoportcomponents.

        Note:
            * This network tries to emulate the behavior of the ConnMatrixNetwork of caphe.
            * delays should be a 1D list or array specifying the delay introduced by the
            twoportcomponents (in contrast to the delay matrix in caphe)
        '''
        Module.__init__(self)
        self.name = self.__class__.__name__.lower() if name is None else name

        self.device = torch.device('cpu')

        self.connections = conn_matrix
        self.components = OrderedDict()
        for i, comp in enumerate(twoportcomponents):
            name = comp.__class__.__name__.lower() + str(i)
            comp = copy(comp) # shallow copy
            comp.name = name
            self.components[name] = comp
            twoportcomponents[i] = comp

        self.twoportcomponents = twoportcomponents

        self.terms = OrderedDict()
        for i, (d, s) in enumerate(zip(detectors_at, sources_at)):
            if d:
                comp = Detector(name='d%i'%i)
                self.components[comp.name]=comp
                self.terms[2*i+1] = comp
            if s:
                comp = Source(name='s%i'%i)
                self.components[comp.name]=comp
                self.terms[2*i] = comp

        sources_at = [isinstance(term, Source) for term in self.terms.values()]
        detectors_at = [isinstance(term, Detector) for term in self.terms.values()]

        self.num_ports = 2*len(twoportcomponents) + len(self.terms)

        self.sources_at = Buffer(torch.zeros(self.num_ports, dtype=torch.uint8, device=self.device))
        self.sources_at[-len(self.terms):] = torch.tensor(sources_at)

        self.detectors_at = Buffer(torch.zeros(self.num_ports, dtype=torch.uint8, device=self.device))
        self.detectors_at[-len(self.terms):] = torch.tensor(detectors_at)

        self.functions_at = Buffer(torch.zeros(self.num_ports, dtype=torch.uint8, device=self.device))

        if delays is None:
            self.delays = Buffer(torch.cat([comp.get_delays() for comp in self.components.values()]))
        else:
            delays = torch.tensor(delays, dtype=torch.get_default_dtype(), device=self.device)
            self.delays = Buffer(torch.stack([delays, delays], -1).flatten())

        self._register_components()

        self._env = None
        self.order = self.get_order()
        self.C = Buffer(self.get_C())
        self.terminated = True
        self.initialized=False

    def terminate(self, term=None, name=None):
        raise RuntimeError('A TwoPortNetwork is always terminated by default')

    def unterminate(self):
        raise RuntimeError('A TwoPortNetwork is always terminated by default')

    def get_used_components(self):
        return self.components.keys()

    def get_sources_at(self):
        return self.sources_at

    def get_detectors_at(self):
        return self.detectors_at

    def get_functions_at(self):
        return self.functions_at

    def get_delays(self):
        return self.delays

    def get_order(self):
        return slice(None) # no reordering necessary

    def get_C(self):
        n = 2*self.connections.shape[0]

        rC = torch.tensor(np.real(self.connections), device=self.device, dtype=torch.get_default_dtype())
        iC = torch.tensor(np.imag(self.connections), device=self.device, dtype=torch.get_default_dtype())

        C = torch.zeros((2, self.num_ports,self.num_ports), device=self.device)

        C[0,1:n:2,0:n:2] = rC.t()
        C[1,1:n:2,0:n:2] = iC.t()
        C[0,0:n:2,1:n:2] = rC
        C[1,0:n:2,1:n:2] = iC

        for j, (i,term) in enumerate(self.terms.items()):
            if isinstance(term, (Source, Detector)):
                C[0,n+j,i] = C[0,i,n+j] = 1

        return C
