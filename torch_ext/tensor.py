'''
# Photontorch Tensor Tools

Some non-differentiable, but useful functions that act on (or create) torch tensors.

'''

#############
## Imports ##
#############

## Torch
import torch
from torch.autograd import Variable
from torch.nn import Parameter

## Relative
from .types import TORCH_TYPES
from .nn import BoundedParameter
from .nn import Buffer


###############
## Functions ##
###############

def is_variable(var):
    ''' Check if an object is Variable-like

    Args:
        var (object): Object to check

    Returns:
        bool
    '''
    return isinstance(var, (Variable, Parameter, BoundedParameter, Buffer))

def zeros(shape, type='float', cuda=False):
    ''' Create an empty torch tensor filled with zeros.

    Args:
        shape (tuple): shape of the new tensor
        dtype (str): type of the new tensor
        cuda (bool): if the new tensor should be cuda or not.
    '''
    Tensor = TORCH_TYPES[type]
    tensor = Tensor(*shape).zero_()
    if cuda:
        return tensor.cuda()
    return tensor

def where(bytetensor):
    ''' Get indices of places where bytetensor > 0.

    Args:
        bytetensor (torch.ByteTensor): tensor to index

    Note:
        Only works for 1D ByteTensors for now.
    '''
    idxs = torch.zeros_like(bytetensor).long()
    torch.arange(0, idxs.size(0), 1, out=idxs)
    idxs = idxs[bytetensor]
    return idxs
