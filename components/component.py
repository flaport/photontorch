'''
# Base Component Submodule

The base component is a parent class meant for subclassing. It should not be used directly.

Each Component is generally defined by several key properties:

  * num_ports: The number of ports of the components.
  * C: The connection matrix for the component (usually all zero for base components)
  * rS: The real part of the Scattering matrix
  * iS: The imaginary part of the Scattering matrix
  * sources_at: Where there are sources in the component (usually all zero for base components)
  * detectors_at: Where there are detectors in the component (usually all zero for base components)
  * delays: delays introduced by the nodes of the component.

'''

#############
## Imports ##
#############

## Standard library
from copy import copy, deepcopy

## Torch
import torch

## Relative
from ..torch_ext import where
from ..torch_ext.nn import Module
from ..torch_ext.nn import Buffer


###############
## Component ##
###############

class Component(Module):
    ''' Generic base component.

        The base component is a parent class meant for subclassing.
        It should not be used directly.
    '''

    num_ports = 1 # Number of ports of the component

    def __init__(self, name=None):
        ''' Component

        Args:
            name (str): the name of the component (defaults to the classname in lowercase)
        '''
        Module.__init__(self)
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name

        # set attributes
        self._env = None
        self.C = Buffer(self.get_C())
        self.free_idxs = Buffer(self.get_free_idxs())
        self.sources_at = Buffer(self.get_sources_at())
        self.detectors_at = Buffer(self.get_detectors_at())

    def initialize(self, env):
        ''' Simulation initialization for the component.

        Befor a component can be used for simulation, it should be initialized with a
        simulation environment.

        Args:
            env (Environment): simulation environment.

        Note:
            Just initializing the component with a simulation environment is not enough
            to be able to start a simulation. A connection matrix for which
            C.sum(0) == C.sum(1) == all ones is required.
            You can achieve this by connecting the component into a network with some Terms.
        '''
        self._env = env
        if env.cuda is not None:
            if env.cuda and not self.is_cuda:
                self.cuda()
            elif not env.cuda and self.is_cuda:
                self.cpu()
        self.zero_grad()
        if (self.sources_at & self.detectors_at).any():
            raise ValueError('Sources and Detectors cannot be combined in the same node.')

        self.delays = self.get_delays()
        self.S = self.get_S()

    @property
    def env(self):
        '''@property
        Returns:
            The environment for which the component is initialized.
        '''
        return self._env

    def get_S(self):
        ''' Scattering matrix of the component

        Returns:
            torch tensor with shape (2, # wavelengths, # ports, # ports) and dtype float32

        Note:
            S[0] is the real part, S[1] is the imaginary part.
        '''

        raise NotImplementedError('The real part of the scattering matrix '
                                  'of %s does not exist'%self.name)

    def get_C(self):
        ''' Connection matrix of the component.

        Returns:
            torch tensor with shape (2, # ports, # ports) and dtype float32

        Note:
            C[0] is the real part, S[0] is the imaginary part
        '''
        return torch.zeros((2, self.num_ports, self.num_ports), device=self.device)

    def get_delays(self):
        ''' Delays introduced by the component.

        Returns:
            torch tensor with the delays in each node of the component.
                shape: (# ports, )
                dtype: float32
        '''
        return torch.zeros(self.num_ports, device=self.device)

    def get_sources_at(self):
        ''' The locations of the sources in the component.

        Returns:
            torch tensor containing the locations of the sources in the component.
                shape: (# ports, )
                dtype: uint8 [byte]
        '''
        return torch.zeros(self.num_ports, device=self.device, dtype=torch.uint8)

    def get_detectors_at(self):
        ''' The locations of the detectors in the component.

        Returns:
            torch tensor containing the locations of the detectors in the component.
                shape: (# ports, )
                dtype: uint8 [byte]
        '''
        return torch.zeros(self.num_ports, device=self.device, dtype=torch.uint8)

    def get_free_idxs(self):
        ''' Free indices to make connections to

        Returns:
            torch tensor containing the free indices in the components connection matrix
                shape: (# ports, )
                dtype: int64 [long]
        '''
        C = (abs(self.C)**2).sum(0)
        return torch.tensor(where(((C.sum(0) > 0) | (C.sum(1) > 0)).ne(1)), device=self.device)

    def __repr__(self):
        ''' String Representation of the component '''
        return self.name

    def __str__(self):
        ''' String Representation of the component '''
        return self.name

    def __getitem__(self, key):
        ''' Special __getitem__

        Each component contains a special __getitem__, which is solely used to connect
        components together. Indexing a component with a string index will create a
        connector with that same index. This connector will connect to any other connector
        where the same string index is used. Connected connectors can be made into a
        network. See Network for more invormation.

        Args:
            key (str): string key of the connection.

        '''
        from ..networks.connector import Connector
        return Connector(key, [self])

    def __deepcopy__(self, memo):
        ''' Create a copy of the component '''
        new = copy(self)
        if new.env is not None:
            del new.rS
            del new.iS
            del new.delays
        new = deepcopy(super(Component, new), memo)
        if new.env is not None:
            new.initialize(new.env)
        return new
