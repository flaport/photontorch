'''
# Torch Autograd Extensions

The autograd extensions supply three main functions:

  * `block_diag`: a differentiable implementation of a block diagonal matrix
performed over a batch of matrices.

The classes defined below are called inside these functions and should not be used directly.

'''

#############
## Imports ##
#############

## Torch
import torch
from torch.autograd import Function

## Other
import numpy as np


####################
## Block Diagonal ##
####################

def block_diag(*inputs):
    ''' Block Diagonal Matrix

    Create a block diagonal matrix from provided input tensors.

    Args:
        *inputs (torch.Tensors): n-dimensional tensors for which the last two indices are
            equal (square).

    Returns:
        torch.Tensor: n-dimensional tensor for which the last two dimensions are the sum
            of the last two dimensions of the input tensor.

    Example:
        block_diag(torch.Tensor(3,4,6,6), torch.Tensor(3,4,2,2)).shape == torch.Size([3,4,8,8])

    '''
    # get indices and size of the submatrices
    sizes = [m.shape[-1] for m in inputs]
    idxs = list(np.cumsum([0]+sizes))
    total_size = int(idxs[-1])

    # get return matrix
    m = inputs[0]
    shape = m.shape[:-2] + (total_size, total_size)
    M = torch.zeros(shape, dtype=m.dtype, device=m.device)

    # fill return matrix
    for (i, j), m in zip(zip(idxs[:-1], idxs[1:]), inputs):
        M[...,i:j,i:j] = m

    # return matrix
    return M
