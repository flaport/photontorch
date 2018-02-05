''' Base Component Module '''

#############
## Imports ##
#############

## Torch
import torch
from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable

## Other
import numpy as np
from copy import deepcopy

## Relative
from ..utils import where
from ..networks.connector import Connector


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
        self._cuda = False

    def initialize(self, env):
        ''' Initialize Component For a simulation by giving it the simulation environment '''
        self._env = env
        if env.cuda is not None:
            if env.cuda and not self._cuda:
                self.cuda()
            elif not env.cuda and self._cuda:
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
        self.initialize(env)

    def cuda(self):
        ''' Transform component to live on the GPU '''
        new = super(Component, self).cuda()
        new._cuda = True
        return new

    def cpu(self):
        ''' Transform component to live on the CPU '''
        new = super(Component, self).cpu()
        new._cuda = False
        return new

    def new_tensor(self, numpy_array, dtype='float'):
        '''
        Tensor constructor.
        '''
        dtype = {'byte':'uint8','float':'float32','double':'float64'}.get(dtype, dtype)
        tensor = torch.from_numpy(np.asarray(numpy_array, dtype=dtype))
        if self._cuda:
            return tensor.cuda()
        return tensor


    def new_parameter(self, numpy_array, dtype='float', requires_grad=True):
        '''
        Parameter constructor.
        Parameters are trainable [requires_grad=True]
        '''
        return Parameter(self.new_tensor(numpy_array, dtype=dtype), requires_grad=requires_grad)

    def new_variable(self, numpy_array, dtype='float', requires_grad=False):
        '''
        Variable constructor
        Variables are not trainable [requires_grad=False]
        '''
        return Variable(self.new_tensor(numpy_array, dtype=dtype), requires_grad=requires_grad)

    @property
    def rS(self):
        '''
        Real Part of the Scattering matrix of the component.
        Should return a Variable containing a torch.FloatTensor.
        '''
        raise NotImplementedError('The real part of the scattering matrix of %s does not exist'%self.name)

    @property
    def iS(self):
        '''
        Imaginary Part of the Scattering matrix of the component.
        Should return a Variable containing a torch.FloatTensor.
        '''
        raise NotImplementedError('The imaginary part of the scattering matrix of %s does not exist'%self.name)

    @property
    def C(self):
        '''
        Connection matrix of the component.
        Should return a Variable containing a torch.FloatTensor consisting of only ones and zeros.
        '''
        return self.new_variable(np.zeros((self.num_ports, self.num_ports)), 'float')

    @property
    def delays(self):
        '''
        The delay introduced by the component.
        Should return a Variable containing a torch.FloatTensor
        containing the delays for each node in the component
        '''
        return self.new_variable(np.zeros(self.num_ports), 'float')

    @property
    def sources_at(self):
        '''
        The locations of the sources in the component.
        Should return a Variable containing a torch.ByteTensor
        containing the locations of the sources in the component
        '''
        return self.new_variable(np.zeros(self.num_ports), 'byte')

    @property
    def detectors_at(self):
        '''
        The locations of the detectors in the component.
        Should return a Variable containing a torch.ByteTensor
        containing the locations of the detectors in the component
        '''
        return self.new_variable(np.zeros(self.num_ports), 'byte')

    @property
    def free_idxs(self):
        ''' Get Free indices for connections '''
        C = self.C
        return where(((C.sum(0) > 0) | (C.sum(1) > 0)).ne(1).data)

    def copy(self):
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



###################
## Other imports ##
###################

# This import needs to happen in the end to prevent circular imports.

## Relative
