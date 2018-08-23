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


#######################
## Bounded Parameter ##
#######################

class BoundedParameter(torch.nn.Parameter):
    def __new__(cls, data=None, bounds=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()
        if bounds is None:
            bounds = (0,1)
        # check bounds
        try:
            a, b = bounds
        except ValueError:
            raise ValueError('bounds should be a tuple with length 2')
        if b<a:
            raise ValueError('bounds should be a tuple with length 2 and with bounds[1] > bounds[2]')
        if (data < a).any() or (data>b).any():
            raise ValueError('some of your data is outside the specified bounds: [%i, %i]'%(a,b))
        new = torch.Tensor._make_subclass(cls, cls._inverse_sigmoid(data.data, (a,b)), requires_grad)
        new.bounds = (float(a), float(b))
        return new

    @staticmethod
    def _sigmoid(weights, bounds):
        a, b = bounds
        scaled_data = torch.sigmoid(weights)
        data = (b - a)*scaled_data + a
        return data

    @staticmethod
    def _inverse_sigmoid(data, bounds):
        a, b = bounds
        scaled_data = (data - a)/(b - a)
        weights = -torch.log(1/scaled_data - 1)
        return weights

    def __repr__(self):
        tensor_repr = torch.Tensor.__repr__(self._sigmoid(self.data, self.bounds))
        return 'BoundedParameter in [%.2f, %.2f] representing:\n'%self.bounds+tensor_repr


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

    '''

    def __init__(self):
        ''' Module Initialization '''
        super(Module, self).__init__()
        self.device = torch.device(type='cpu')

    def __setattr__(self, attr, value):
        # Check for buffers and if the value is a buffer, register it.
        if isinstance(value, Buffer):
            self.register_buffer(attr, value)
        if isinstance(value, BoundedParameter):
            _attr = '_'+attr
            self.register_parameter(_attr, value)
            value_property = property(lambda self: self._parameters[_attr]._sigmoid(self._parameters[_attr], self._parameters[_attr].bounds))
            self.__class__ = type('ModuleWithBoundedParameter', (self.__class__,), {attr:value_property}) # subclass on the fly
        else:
            super(Module, self).__setattr__(attr, value)

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
