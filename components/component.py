''' Base Component Module '''

#############
## Imports ##
#############

## Torch
from torch.autograd import Variable

## Other
import numpy as np
from copy import deepcopy

## Relative
from ..torch_ext import where
from ..torch_ext.nn import Module
from ..networks.connector import Connector
from ..environment.environment import Environment


###############
## Component ##
###############

class Component(Module):
    '''
    Generic component.
    '''

    num_ports = 1 # Number of ports of the component

    def __init__(self, name=None):
        '''
        Initialization of the component.

        Note
        ----
        This is a parent class, not intended to be used directly.
        '''
        Module.__init__(self)
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name
        self.is_cuda = False
        self._env = Environment() # Set default environment

    def initialize(self, env):
        ''' Initialize Component For a simulation by giving it the simulation environment '''
        self._env = env
        if env.cuda is not None:
            if env.cuda and not self.is_cuda:
                self.cuda()
            elif not env.cuda and self.is_cuda:
                self.cpu()
        self.zero_grad()
        if (self.sources_at & self.detectors_at).any():
            raise ValueError('Sources and Detectors cannot be combined in the same node.')

    @property
    def env(self):
        '''
        Alternative initializer. This is here as a safeguard for when env would be
        set manually.
        '''
        return self._env
    @env.setter
    def env(self, value):
        self.initialize(value)

    @property
    def rS(self):
        '''
        Real Part of the Scattering matrix of the component.
        Should return a Variable containing a torch.FloatTensor.
        shape: (# num wavelengths, # num ports, # num ports)
        '''
        raise NotImplementedError('The real part of the scattering matrix '
                                  'of %s does not exist'%self.name)

    @property
    def iS(self):
        '''
        Imaginary Part of the Scattering matrix of the component.
        Should return a Variable containing a torch.FloatTensor.
        shape: (# num wavelengths, # num ports, # num ports)
        '''
        raise NotImplementedError('The imaginary part of the scattering matrix '
                                  'of %s does not exist'%self.name)

    @property
    def C(self):
        '''
        Connection matrix of the component.
        Should return a Variable containing a torch.FloatTensor consisting of only ones and zeros.
        shape: (# num ports, # num ports)
        '''
        return self.new_variable(np.zeros((self.num_ports, self.num_ports)), 'float')

    @property
    def delays(self):
        '''
        The delay introduced by the component.
        Should return a Variable containing a torch.FloatTensor
        containing the delays for each node in the component
        shape: (# num ports, )
        '''
        return self.new_variable(np.zeros(self.num_ports), 'float')

    @property
    def sources_at(self):
        '''
        The locations of the sources in the component.
        Should return a Variable containing a torch.ByteTensor
        containing the locations of the sources in the component
        shape: (# num ports, )
        '''
        return self.new_variable(np.zeros(self.num_ports), 'byte')

    @property
    def detectors_at(self):
        '''
        The locations of the detectors in the component.
        Should return a Variable containing a torch.ByteTensor
        containing the locations of the detectors in the component
        shape: (# num ports, )
        '''
        return self.new_variable(np.zeros(self.num_ports), 'byte')

    @property
    def free_idxs(self):
        ''' Get Free indices for connections '''
        C = self.C
        return where(((C.sum(0) > 0) | (C.sum(1) > 0)).ne(1).data)

    def copy(self):
        ''' Make a (deep) copy of the component '''
        new = self.__class__.__new__(self.__class__)
        new.__dict__['env'] = None
        for k, v in self.__dict__.items():
            if k == 'components': # special way of copying subcomponent of a network
                new.__dict__[k] = tuple([comp.copy() for comp in v])
            elif isinstance(v, Component):
                new.__dict__[k] = v.copy()
            elif isinstance(v, Variable):
                # Do nothing to variables that are not parameters or leafs
                # NOTE: parameters are in the ordered_dict _parameters and will
                # be copied by the deepcopy below.
                pass
            else:
                new.__dict__[k] = deepcopy(v)
        if new.__dict__['env'] is not None:
            new.initialize(new.env)
        return new

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __getitem__(self, s):
        return Connector(s, [self])
