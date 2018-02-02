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


###############
## Component ##
###############

class Component(Module):
    '''
    Generic component, defined by a scattering matrix and a connection matrix.
    '''
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
        self.env = env

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
        return self.new_variable(np.zeros(self.rS.size()), 'float')

    @property
    def delays(self):
        '''
        The delay introduced by the component.
        Should return a Variable containing a torch.FloatTensor
        containing the delays for each node in the component
        '''
        return self.new_variable([0]*self.rS.size(0), 'float')

    @property
    def sources_at(self):
        '''
        The locations of the sources in the component.
        Should return a Variable containing a torch.ByteTensor
        containing the locations of the sources in the component
        '''
        return self.new_variable([0]*self.rS.size(0), 'byte')

    @property
    def detectors_at(self):
        '''
        The locations of the detectors in the component.
        Should return a Variable containing a torch.ByteTensor
        containing the locations of the detectors in the component
        '''
        return self.new_variable([0]*self.rS.size(0), 'byte')

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __getitem__(self, s):
        return Connector(s, [self])



###################
## Other imports ##
###################

# These imports need to happen here because otherwise we have circular imports...

## Relative
from ..network.connector import Connector
