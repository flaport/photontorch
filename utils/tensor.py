''' Torch tensor function extensions '''

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np


###############
## Constants ##
###############

TENSOR_TYPES = {
    'torch.FloatTensor':torch.FloatTensor,
    'torch.DoubleTensor':torch.DoubleTensor,
    'torch.HalfTensor':torch.HalfTensor,
    'torch.ByteTensor':torch.ByteTensor,
    'torch.CharTensor':torch.CharTensor,
    'torch.ShortTensor':torch.ShortTensor,
    'torch.IntTensor':torch.IntTensor,
    'torch.LongTensor':torch.LongTensor,
    'torch.cuda.FloatTensor':torch.cuda.FloatTensor,
    'torch.cuda.DoubleTensor':torch.cuda.DoubleTensor,
    'torch.cuda.HalfTensor':torch.cuda.HalfTensor,
    'torch.cuda.ByteTensor':torch.cuda.ByteTensor,
    'torch.cuda.CharTensor':torch.cuda.CharTensor,
    'torch.cuda.ShortTensor':torch.cuda.ShortTensor,
    'torch.cuda.IntTensor':torch.cuda.IntTensor,
    'torch.cuda.LongTensor':torch.cuda.LongTensor,
    torch.FloatTensor:torch.FloatTensor,
    torch.DoubleTensor:torch.DoubleTensor,
    torch.HalfTensor:torch.HalfTensor,
    torch.ByteTensor:torch.ByteTensor,
    torch.CharTensor:torch.CharTensor,
    torch.ShortTensor:torch.ShortTensor,
    torch.IntTensor:torch.IntTensor,
    torch.LongTensor:torch.LongTensor,
    torch.cuda.FloatTensor:torch.cuda.FloatTensor,
    torch.cuda.DoubleTensor:torch.cuda.DoubleTensor,
    torch.cuda.HalfTensor:torch.cuda.HalfTensor,
    torch.cuda.ByteTensor:torch.cuda.ByteTensor,
    torch.cuda.CharTensor:torch.cuda.CharTensor,
    torch.cuda.ShortTensor:torch.cuda.ShortTensor,
    torch.cuda.IntTensor:torch.cuda.IntTensor,
    torch.cuda.LongTensor:torch.cuda.LongTensor,
}

###############
## Functions ##
###############

def zeros(*shape, **kwargs):
    '''
    Create an empty torch tensor with a certain type

    Arguments
    ---------
    *shape : shape of the new tensor
    type = 'torch.FloatTensor' : type of the new tensor
    '''
    type = kwargs.pop('type','torch.FloatTensor')
    Tensor = TENSOR_TYPES[type]
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