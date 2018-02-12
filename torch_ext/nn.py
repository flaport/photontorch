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


#######################
## Bounded Parameter ##
#######################

class BoundedParameter(object):
    '''
    Bounded Parameter

    A bounded parameter is recognised by the Module extension to create a new parameter
    (the weights). Next a reference (property) is stored to the class instance by some
    dark magic that creates a variable on the fly.

    More concretely lets say you define a bounded parameter as follows:

    def __init__(self, R, R_bounds, ...):
        ...
        self.R = BoundedParameter(data=R, bounds=R_bounds)
        ...

    This is effectively very similar to the following:

    def __init__(self, R, R_bounds, ...)
        ...
        self.R_bounds = R_bounds
        self._R = Parameter(data=-log(1/R-1))
        ...
    @property
    def R(self):
        return sigmoid(self._R)

    With the addition of some setters also defined.
    '''
    def __init__(self, data=None, bounds=None, requires_grad=True):
        ''' Bounded Parameter __init__
        data : (torch.Tensor) : The data of the Parameter
        bounds : (tuple) : The bounds of the parameter
        requires_grad (bool) : If the parameter is trainable.
        '''
        self.data = data
        self.bounds = bounds
        self.requires_grad = requires_grad


######################
## Module Extension ##
######################

class Module(_Module_):
    ''' Torch.nn Module extension for bounded parameters '''
    def register_bounded_parameter(self, name, bparam):
        '''
        Some setattr, getattr magic to create an unbounded Parameter weight and
        Some class properties pointing to the scaled version of that weight.
        '''
        cls = self.__class__
        setattr(self, name+'_bounds', tuple(bparam.bounds))
        assert getattr(self, name+'_bounds')[0] < getattr(self, name+'_bounds')[1]
        assert len(getattr(self, name+'_bounds')) == 2
        setattr(cls, name+'_min', property(lambda self: getattr(self, name+'_bounds')[0]))
        def set_min(self, value):
            ''' Set minimum of bounded parameter '''
            setattr(self, name+'_bounds', (value, getattr(self, name+'_bounds')[1]))
        setattr(cls, name+'_min', getattr(cls, name+'_min').setter(set_min))
        setattr(cls, name+'_max', property(lambda self: getattr(self, name+'_bounds')[1]))
        def set_max(self, value):
            ''' Set maximum of bounded parameter '''
            setattr(self, name+'_bounds', (getattr(self, name+'_bounds')[0], value))
        setattr(cls, name+'_max', getattr(cls, name+'_max').setter(set_max))
        def get(self):
            ''' Get bounded parameter '''
            min = getattr(self, name+'_bounds')[0]
            max = getattr(self, name+'_bounds')[1]
            return (max-min)*torch.sigmoid(getattr(self, '_'+name)) + min
        def set(self, value):
            ''' Set bounded parameter '''
            eps = 1e-8
            min = getattr(self, name+'_bounds')[0]
            max = getattr(self, name+'_bounds')[1]
            if not isinstance(value, torch.Tensor) or not isinstance(value, Variable):
                value = np.array(value)
                if value.ndim == 0:
                    value = value[None]
                value = self.new_tensor(value)
            data = ((1-2*eps)*((value - min)/(max - min) + eps))
            setattr(
                self, '_'+name,
                Parameter(-torch.log(1/data-1), # inverse sigmoid
                          requires_grad=bparam.requires_grad),
            )
        set(self, bparam.data)
        setattr(cls, name, property(get))
        setattr(cls, name, getattr(cls, name).setter(set))

    def __setattr__(self, attr, value):
        ''' custom setattr to take BoundedParameter into account '''
        if isinstance(value, BoundedParameter):
            self.register_bounded_parameter(attr, value)
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
        return Parameter(self.new_tensor(data, dtype=dtype, cuda=cuda), requires_grad=requires_grad)

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
