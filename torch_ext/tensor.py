''' Torch tensor function extensions '''

#############
## Imports ##
#############

## Torch
import torch

## Relative
from .types import TORCH_TYPES


###############
## Functions ##
###############

def zeros(shape, **kwargs):
    '''
    Create an empty torch tensor with a certain type

    Arguments
    ---------
    *shape : shape of the new tensor
    type = 'torch.FloatTensor' : type of the new tensor
    '''
    type = kwargs.pop('type', 'torch.FloatTensor')
    Tensor = TORCH_TYPES[type]
    tensor = Tensor(*shape).zero_()
    return tensor

def where(bytetensor):
    '''
    Get indices of places where bytetensor > 0.
    NOTE: Only works for 1D ByteTensors for now.
    '''
    idxs = torch.zeros_like(bytetensor).long()
    torch.arange(0, idxs.size(0), 1, out=idxs)
    idxs = idxs[bytetensor]
    return idxs
