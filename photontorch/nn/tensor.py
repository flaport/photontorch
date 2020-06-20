""" Photontorch Tensor Tools

Some non-differentiable, but useful functions that act on (or create) torch tensors.

"""

#############
## Imports ##
#############

## Torch
import torch


###############
## Functions ##
###############


def where(bytetensor):
    """ Get indices of places where bytetensor > 0.

    Args:
        bytetensor (ByteTensor): tensor to index

    Returns:
        Tensor[int64]: indices where the bytetensor is True

    Note:
        Only works for 1D ByteTensors for now.

    """
    idxs = torch.arange(
        0, bytetensor.shape[0], 1, dtype=torch.int64, device=bytetensor.device
    )
    return idxs[bytetensor]
