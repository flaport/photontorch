'''
# Type Conversions

Common type converstions between Python, Numpy and PyTorch.

'''

#############
## Imports ##
#############

## Torch
import torch


###########
## Torch ##
###########

TORCH_TYPES = {
    float:torch.FloatTensor,
    int:torch.IntTensor,
    bool:torch.ByteTensor,
    'uint8':torch.ByteTensor,
    'float':torch.FloatTensor,
    'float32':torch.FloatTensor,
    'float64':torch.DoubleTensor,
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

###########
## Numpy ##
###########
NUMPY_TYPES = {
    float:'float',
    int:'int',
    bool:'bool',
    'byte':'uint8',
    'float':'float32',
    'double':'float64',
    'torch.FloatTensor':'float32',
    'torch.DoubleTensor':'float64',
    'torch.ByteTensor':'uint8',
    'torch.IntTensor':'int',
    'torch.LongTensor':'long',
    'torch.cuda.FloatTensor':'float32',
    'torch.cuda.DoubleTensor':'float64',
    'torch.cuda.ByteTensor':'uint8',
    'torch.cuda.IntTensor':'int',
    'torch.cuda.LongTensor':'long',
    torch.FloatTensor:'float32',
    torch.DoubleTensor:'float64',
    torch.ByteTensor:'uint8',
    torch.IntTensor:'int',
    torch.LongTensor:'long',
    torch.cuda.FloatTensor:'float32',
    torch.cuda.DoubleTensor:'float64',
    torch.cuda.ByteTensor:'uint8',
    torch.cuda.IntTensor:'int',
    torch.cuda.LongTensor:'long',
}

############
## Simple ##
############

SIMPLE_TYPES = {
    'byte':bool,
    'float':float,
    'double':float,
    'torch.FloatTensor':float,
    'torch.DoubleTensor':float,
    'torch.ByteTensor':bool,
    'torch.IntTensor':int,
    'torch.LongTensor':int,
    'torch.cuda.FloatTensor':int,
    'torch.cuda.DoubleTensor':float,
    'torch.cuda.ByteTensor':bool,
    'torch.cuda.IntTensor':int,
    'torch.cuda.LongTensor':int,
    torch.FloatTensor:float,
    torch.DoubleTensor:float,
    torch.ByteTensor:bool,
    torch.IntTensor:int,
    torch.LongTensor:int,
    torch.cuda.FloatTensor:float,
    torch.cuda.DoubleTensor:float,
    torch.cuda.ByteTensor:float,
    torch.cuda.IntTensor:int,
    torch.cuda.LongTensor:int,
}