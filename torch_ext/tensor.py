'''
# Photontorch Tensor Tools

Some non-differentiable, but useful functions that act on (or create) torch tensors.

'''

#############
## Imports ##
#############

## Torch
import torch


###############
## Functions ##
###############

def where(bytetensor):
    ''' Get indices of places where bytetensor > 0.

    Args:
        bytetensor (torch.ByteTensor): tensor to index

    Note:
        Only works for 1D ByteTensors for now.
    '''
    idxs = torch.zeros_like(bytetensor).long()
    torch.arange(0, idxs.shape[0], 1, out=idxs)
    idxs = idxs[bytetensor]
    return idxs
