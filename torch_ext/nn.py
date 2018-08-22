'''
# Torch Neural Network Extensions

The Neural Network Extensions Consist out of three main objects:

  * `BoundedParameter`: A bounded parameter acts like a
`torch.nn.Parameter` that is bounded between a certain range. Under the hood it is
actually a `torch.nn.Module`, but for all intents and purposes it can be considered
to act like a `torch.nn.Parameter`.

  * `Buffer`: A special kind of tensor that automatically will
be added to the `._buffers` attribute of the Module. Buffers are typically used as
parameters of the model that do not require gradients or optimization, but that are
indispensible for the definition of your object.

  * `Module`: Extends `torch.nn.Module`, with some extra features (such as automatically)
registering a `Buffer` in its `._buffers` attribute, modified `.cuda()` calls and some
extra functionalities.

'''


#############
## Imports ##
#############

## Standard Library
import copy

## Torch
import torch
from torch.nn import * # we add to torch.nn
from torch.nn import Parameter
from torch.nn import Module as _Module_


############
## Buffer ##
############
class Buffer(torch.Tensor):
    ''' A variable of a module that is automatically registered in _buffers

    Each Module has an OrderedDict named _buffers. In this Dictionary, all model related
    parameters that do not require optimization are stored.

    This Class makes it easier to register a buffer in the Module. If an attribute of a
    module is set to a Buffer, it will automatically be added to the _buffers attribute.

    Note:
        For the automatic registration of the Buffer to work, you need to use the torch_ext
        subclass Module of torch.nn.Module.
    '''
    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __repr__(self):
        return 'Buffer containing:\n' + super(Buffer, self).__repr__()


######################
## Module Extension ##
######################

class Module(_Module_):
    ''' Torch.nn Module extension with some extra features:

    In PhotonTorch, often new variables need to be created on the fly. Therefore, three
    new functions (`tensor`, `buffer`, `parameter`) were created to easily
    create these, with the requested type and cuda flag.

    Buffers are automatically registerd in the _buffers attribute

    .cuda() calls perfom the cuda calls on all parameters, buffers and submodules.
    (this is different from the default PyTorch behavior, where just Parameters are
    converted to cuda.)

    .copy() creates a deep copy of the module

    '''

    def __init__(self):
        ''' Module Initialization '''
        super(Module, self).__init__()
        self.device = torch.device(type='cpu')

    def __setattr__(self, attr, value):
        # Check for buffers and if the value is a buffer, register it.
        if isinstance(value, Buffer):
            self.register_buffer(attr, value)
        else:
            super(Module, self).__setattr__(attr, value)

    def copy(self):
        ''' Make a deep copy of the module '''
        return copy.deepcopy(self)

    @property
    def is_cuda(self):
        ''' check if the model parameters live on the GPU '''
        return False if self.device.type == 'cpu' else True

    def to(self, *args, **kwargs):
        new = super(Module, self).to(*args, **kwargs)
        for k, v in self._modules.items():
            self._modules[k] = v.to(*args, **kwargs)
        new.device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        return new

    def cpu(self):
        ''' Transform the Module to live on the CPU '''
        return self.to(device='cpu')

    def cuda(self, device=None):
        ''' Transform the Module to live on the GPU

        Args:
            device (int): index of the GPU device.
        '''
        if device is None:
            device = 'cuda:0'
        elif isinstance(device, int):
            device = 'cuda:%i'%device
        return self.to(device=device)



#######################
## Bounded Parameter ##
#######################

class BoundedParameter(Module):
    ''' Bounded Parameter.

        A bounded parameter acts like a `torch.nn.Parameter` that is bounded between a
        certain range. Under the hood it is actually a `torch.nn.Module` for which certain
        common operations with normal variables (such as addition, multiplication, ...)
        are defined. This way it can act as a Parameter (with the only difference that
        it will be registerd in the ._modules dictionary of a model in stead of the
        ._parameters dictionary.)

        The parameter is bounded by specifying a set of `weights`, which are the inverse
        sigmoid of the parameter (min-max normalized between its bounds).
        This way, taking the sigmoid of the weights will always yield a bounded parameter.

    '''

    def __init__(self, data, bounds=None, requires_grad=True):
        ''' Bounded Parameter

        Args:
            data (torch.Tensor): data to construct the Bounded Parameter for
            bounds (tuple): the bounds of the Bounded Parameter
                (note that bounds[0] < bounds[1] is expected)
            requires_grad=True (bool): wether the Bounded Parameter needs optimization.

        Note:
            If no bounds are specified, the Bounded Parameter will act just like a normal
            Parameter or Buffer (depending if requires_grad is True or False)

        '''
        super(BoundedParameter, self).__init__()
        if not torch.is_tensor(data):
            raise RuntimeError('paramter data has to be a tensor, but got %s'%type(data).__name__)
        self.device = data.device
        self._datavar = None # To store the variable if no bounds are specified
        self.bounds = bounds # Store the bounds
        # If no bounds are specified, save the data directly into a Parameter
        if self.bounds is None:
            if requires_grad:
                self._datavar = Parameter(data=data, requires_grad=True)
            else:
                self._datavar = Buffer(data=data, requires_grad=False)
        elif self.bounds[0] == self.bounds[1]: # If the bounds are the same, no training can occur.
            self._datavar = Buffer(data=data, requires_grad=False)
        else: # If the bounds are valid, we normalize the data between 0 and 1 and calculate its
              # weights by taking the inverse sigmoid
            if ((data < bounds[0]) | (data > bounds[1])).any():
                raise ValueError('BoundedParameters data is not compatible with bounds')
            scaled_data = (data - bounds[0])/(bounds[1]-bounds[0])
            weights = -torch.log(1/scaled_data-1) # inverse sigmoid
            if requires_grad:
                self.weights = Parameter(data=weights, requires_grad=True)
            else:
                self.weights = Buffer(data=weights, requires_grad=False)

    def __repr__(self):
        ''' String representation of a bounded parameter '''
        if self.bounds is not None:
            b0 = '0' if self.bounds[0] == 0 else '%.2f'%self.bounds[0] if abs(self.bounds[0]) > 0.01 else '%.2e'%self.bounds[1]
            b1 = '0' if self.bounds[1] == 0 else '%.2f'%self.bounds[1] if abs(self.bounds[1]) > 0.01 else '%.2e'%self.bounds[1]
            bounds = '('+b0+','+b1+')'
            name = 'BoundedParameter with bounds '+bounds+':\n'
        else:
            name = 'BoundedParameter with no bounds:\n'
        return name + self.data.__repr__()

    def copy(self):
        ''' deep copy of the bounded parameter '''
        new = self.__class__(
            data = self.data.clone(),
            bounds = (self.bounds[0], self.bounds[1]),
            requires_grad = self.requires_grad,
        )
        return new

    @property
    def datavar(self):
        '''@property

        The tensor the bounded Parameter is trying to emulate

        Returns:
            torch.autograd.tensor
        '''
        if self._datavar is not None: # this happens when no bounds were specified
            return self._datavar
        else: # this happens when bounds were specified
            # Convert the weights to the requested variable
            scaled_data = torch.sigmoid(self.weights)
            data = (self.bounds[1]-self.bounds[0])*scaled_data + self.bounds[0]
            return data

    @property
    def data(self):
        '''@property

        The data contained by the Bounded Parameter

        Returns:
            torch.Tensor
        '''
        return self.datavar.data

    @data.setter
    def data(self, value):
        '''@property setter

        Set the tensor data of the Bounded Parameter

        Setting the data of the bounded parameter should work just like setting the data
        of a normal parameter

        Args:
            value (torch.Tensor): the data to set the Bounded Parameter to.
        '''
        if self._datavar is not None:
            self._datavar.data = value # this happens when no bounds were specified
        else: # this happens when bounds were specified
            # convert the data to the weights by first normalizing and then taking the
            # inverse sigmoid
            scaled_data = (value - self.bounds[0])/(self.bounds[1]-self.bounds[0])
            self.weights.data = -torch.log(1/scaled_data-1)
    def cos(self):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar.cos()
    def sin(self):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar.sin()
    def __add__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar + other
    def __radd__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return other + self.datavar
    def __sub__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar - other
    def __rsub__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return other - self.datavar
    def __mul__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar*other
    def __rmul__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return other*self.datavar
    def __div__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar/other
    def __rdiv__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return other/self.datavar
    def __truediv__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar/other
    def __rtruediv__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return other/self.datavar
    def __floordiv__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar//other
    def __rfloordiv__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal parameter '''
        return other//self.datavar
    def __pow__(self, other):
        ''' custom function that makes the bounded parameter act just like a normal Parameter '''
        return self.datavar**other
