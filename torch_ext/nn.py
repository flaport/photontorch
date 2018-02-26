''' Torch.nn extensions '''

#############
## Imports ##
#############

## Torch
import torch
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import Module as _Module_
## Other
import numpy as np

## Relative
from .types import NUMPY_TYPES
from .types import TORCH_TYPES
from .types import SIMPLE_TYPES


#######################
## Bounded Parameter ##
#######################


class BoundedParameter(_Module_):
    def __init__(self, data, bounds=None, requires_grad=True):
        _Module_.__init__(self)
        if not torch.is_tensor(data):
            raise RuntimeError('Bounded Parameter expects a Tensor')
        self._datavar = None
        self.bounds = bounds
        if self.bounds is None:
            self._datavar = Parameter(data=data, requires_grad=requires_grad)
        elif self.bounds[0] == self.bounds[1]:
            self._datavar = Buffer(data=data, requires_grad=False)
        else:
            scaled_data = (data - bounds[0])/(bounds[1]-bounds[0])
            weights = -torch.log(1/scaled_data-1)
            self.weights = Parameter(data=weights, requires_grad=requires_grad)
    def __repr__(self):
        type = SIMPLE_TYPES[self.data.__class__]
        if self.bounds is not None:
            if type is float:
                if abs(min(self.bounds)) < 1e-3 and abs(min(self.bounds)) > 0:
                    bounds = '(%.2e,%.2e)'%self.bounds
                else:
                    bounds = '(%.3f,%.3f)'%self.bounds
            else:
                bounds = '(%s,%s)'%[str(i) for i in self.bounds]
            name = 'BoundedParameter with bounds '+bounds+':'
        else:
            name = 'BoundedParameter with no bounds:'
        return name + self.data.__repr__()
    @property
    def datavar(self):
        if self._datavar is not None:
            return self._datavar
        else:
            scaled_data = torch.sigmoid(self.weights)
            data = (self.bounds[1]-self.bounds[0])*scaled_data + self.bounds[0]
            return data
    @property
    def data(self):
        return self.datavar.data
    @data.setter
    def data(self, value):
        if self._datavar is not None:
            self._datavar.data = value
        else:
            scaled_data = (value - self.bounds[0])/(self.bounds[1]-self.bounds[0])
            self.weights.data = -torch.log(1/scaled_data-1)
    def cos(self):
        return self.datavar.cos()
    def sin(self):
        return self.datavar.sin()
    def __add__(self, other):
        return self.datavar + other
    def __radd__(self, other):
        return other + self.datavar
    def __sub__(self, other):
        return self.datavar - other
    def __rsub__(self, other):
        return other - self.datavar
    def __mul__(self, other):
        return self.datavar*other
    def __rmul__(self, other):
        return other*self.datavar
    def __div__(self, other):
        return self.datavar/other
    def __div__(self, other):
        return other/self.datavar
    def __truediv__(self, other):
        return self.datavar/other
    def __rtruediv__(self, other):
        return other/self.datavar
    def __floordiv__(self, other):
        return self.datavar//other
    def __floordiv__(self, other):
        return other//self.datavar
    def __pow__(self, other):
        return self.datavar**other

class Buffer(Variable):
    ''' A variable of a module that is registered in _buffers '''
    def __repr__(self):
        return 'Buffer containing:' + self.data.__repr__()


######################
## Module Extension ##
######################

class Module(_Module_):
    ''' Torch.nn Module extension for bounded parameters '''
    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        # normal zero grad
        for p in self.parameters():
            if p.grad is not None:
                if p.grad.volatile:
                    p.grad.data.zero_()
                else:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())
            # if in the training cycle, the some parameters were out of bounds:
            if hasattr(p, 'bounds') and p.bounds is not None:
                p.data[p.data<p.bounds[0]] = p.bounds[0]
                p.data[p.data>p.bounds[1]] = p.bounds[1]

    def __setattr__(self, attr, value):
        if isinstance(value, Buffer):
            self.register_buffer(attr, value)
        else:
            super(Module, self).__setattr__(attr, value)

    def zeros(self, shape, dtype='float', cuda=None):
        '''
        Create an empty torch tensor with a certain type

        Arguments
        ---------
        *shape : shape of the new tensor
        type = 'torch.FloatTensor' : type of the new tensor
        '''
        Tensor = TORCH_TYPES[dtype]
        tensor = Tensor(*shape).zero_()
        if self.is_cuda and (cuda is None or cuda is True):
            tensor = tensor.cuda()
        return tensor

    def ones(self, shape, dtype='float', cuda=None):
        '''
        Create an empty torch tensor with a certain type

        Arguments
        ---------
        *shape : shape of the new tensor
        type = 'torch.FloatTensor' : type of the new tensor
        '''
        return self.zeros(shape, dtype=dtype, cuda=cuda) + 1

    def new_tensor(self, data, dtype='float', cuda=None):
        '''
        Tensor constructor.
        '''
        dtype = NUMPY_TYPES.get(dtype, dtype)
        if not isinstance(data, torch.Tensor):
            data = np.asarray(data, dtype=dtype)
            if data.ndim == 0:
                data = data[None]
            data = torch.from_numpy(data)
        if self.is_cuda and (cuda is None or cuda is True):
            return data.cuda()
        return data

    def new_parameter(self, data, dtype='float', cuda=None, requires_grad=True):
        '''
        Parameter constructor.
        Parameters are trainable [requires_grad=True]
        '''
        data = self.new_tensor(data, dtype=dtype, cuda=cuda)
        if not requires_grad:
            return Buffer(data, requires_grad=False)
        return Parameter(data, requires_grad=True)

    def new_bounded_parameter(self, data, bounds=None, dtype='float',
                              cuda=None, requires_grad=True):
        '''
        Bounded Parameter Constructor
        Bounded Parameters are trainable [requires_grad=True]
        '''
        if bounds is None:
            return self.new_parameter(data, dtype=dtype, cuda=cuda, requires_grad=requires_grad)
        bparam = BoundedParameter(
            self.new_tensor(data, dtype=dtype, cuda=cuda),
            bounds=bounds,
            requires_grad=requires_grad
        )
        return bparam

    def new_variable(self, data, dtype='float', cuda=None, requires_grad=False):
        '''
        Variable constructor
        Variables are not trainable [requires_grad=False]
        '''
        return Variable(self.new_tensor(data, dtype=dtype, cuda=cuda), requires_grad=requires_grad)

    def cuda(self, device=None):
        ''' Transform component to live on the GPU '''
        new = super(Module, self).cuda(device=device)
        new.is_cuda = True
        return new

    def cpu(self):
        ''' Transform component to live on the CPU '''
        new = super(Module, self).cpu()
        new.is_cuda = False
        return new
