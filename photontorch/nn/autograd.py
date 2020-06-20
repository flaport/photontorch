""" Torch Autograd Extensions """

#############
## Imports ##
#############

## Torch
import torch

## Other
import numpy as np


####################
## Block Diagonal ##
####################


def block_diag(*inputs):
    """ Block Diagonal Matrix

    Create a block diagonal matrix from provided input tensors.

    Args:
        *inputs (Tensors): n-dimensional tensors for which the last two
            indices are equal (square).

    Returns:
        Tensor: n-dimensional tensor for which the last two dimensions are the sum
            of the last two dimensions of the input tensor.

    Example:
        >>> a = torch.randn(3,4,6,6)
        >>> b = torch.randn(3,4,2,2)
        >>> c = block_diag(a, b)
        >>> c.shape
        torch.Size([3, 4, 8, 8])

    """
    # get indices and size of the submatrices
    sizes = [m.shape[-1] for m in inputs]
    idxs = list(np.cumsum([0] + sizes))
    total_size = int(idxs[-1])

    # get return matrix
    m = inputs[0]
    shape = m.shape[:-2] + (total_size, total_size)
    M = torch.zeros(shape, dtype=m.dtype, device=m.device)

    # fill return matrix
    for (i, j), m in zip(zip(idxs[:-1], idxs[1:]), inputs):
        M[..., i:j, i:j] = m

    # return matrix
    return M
