'''
# Torch Autograd Extensions

The autograd extensions supply three main functions:

  * `pinv`: a differentiable implementation of the pseudo inverse of a matrix
  * `block_diag`: a differentiable implementation of a block diagonal matrix
  * `batch_block_diag`: a differentiable implementation of a block diagonal matrix
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

## Relative
from .tensor import zeros


####################
## Block Diagonal ##
####################

def block_diag(*inputs):
    ''' Block Diagonal Matrix

    Create a block diagonal matrix from provided input matrices.

    Args:
        *inputs (2D torch.Tensors): Square Matrices to construct the block diagonal matrix from.

    Returns:
        2D torch.Tensor: The Block diagonal matrix.

    '''
    return BlockDiag().apply(*inputs)

class BlockDiag(Function):
    ''' Block Diagonal Matrix

    Create a block diagonal matrix from provided input matrices.

    Note:
        Do not use this torch Autograd Function Class directly. Use `block_diag` in stead.

    '''
    @staticmethod
    def forward(ctx, *inputs):
        ''' BlockDiag forward

        The forward method creates the block diagonal matrix and saves the variables
        for a backward pass in the context variable.

        Args:
            ctx (torch context): An object storing the information for backward pass.
            *inputs (2D torch.Tensors): Square Matrices to construct the block diagonal matrix from.

        Returns:
            2D torch.Tensor: The block diagonal matrix

        Note:
            This function is used by PyTorch internally and should not be used by the user.
            Use the `.apply` function in stead.

        '''
        # we assume all inputs are square. TODO: implement a check for this
        # Get total size of block diagonal matrix
        sizes = [m.size(0) for m in inputs]
        idxs = list(np.cumsum([0]+sizes))
        total_size = int(idxs[-1])
        # Get start and end indices of blocks in matrix
        ctx.idxs = list(zip(idxs[:-1], idxs[1:]))
        # Get type of new matrix and create empty new matrix with total_size as shape
        M = zeros((total_size, total_size), type=inputs[0].type())
        # Fill Blocks
        for (i, j), matrix in zip(ctx.idxs, inputs):
            M[i:j, i:j] = matrix
        return M

    @staticmethod
    def backward(ctx, grad_output):
        ''' BlockDiag backward

        The backward pass selects the relevant submatrices from the gradients of the
        block diagonal matrix and returns them as the requested gradients of the input
        matrices.

        Args:
            ctx (torch context): An object storing the indices of the submatrices
            grad_output: The gradients of the block diagonal matrix.

        Returns:
            The gradients of the submatrices of the block diagonal matrix.

        Note:
            This function is used by PyTorch internally and should not be used by the user.

        '''
        outputs = []
        for i, j in ctx.idxs:
            output = grad_output[i:j, i:j]
            outputs.append(output)
        return tuple(outputs)


#################################
## Batch Block Diagonal Matrix ##
#################################

def batch_block_diag(*inputs):
    ''' Block Diagonal Matrix

    Create a block diagonal matrix from the provided batch input matrices.

    Args:
        *inputs (3D torch.Tensors): Tensors to construct the batch block diagonal matrix from.
            - The shape of the inputs should be (#batches, dim, dim).
            - The first dimension size (the batch size) of all inputs should be the same.

    Returns:
        3D torch.Tensor: The Batch Block diagonal matrix.

    '''
    return BatchBlockDiag().apply(*inputs)

class BatchBlockDiag(Function):
    ''' Batch Block Diagonal Matrix

    Create a batch block diagonal matrix from provided input matrices.

    Note:
        Do not use this torch Autograd Function Class directly. Use `batch_block_diag` in stead.

    '''
    @staticmethod
    def forward(ctx, *inputs):
        ''' BatchBlockDiag forward

        The forward method creates the batch block diagonal matrix and saves the variables
        for a backward pass in the context variable.

        Args:
            ctx (torch context): An object storing the information for backward pass.
            *inputs (3D torch.Tensors): Tensors of Batch Square Matrices to construct the
                batch block diagonal matrix from.

        Returns:
            3D torch.Tensor: The batch block diagonal matrix

        Note:
            This function is used by PyTorch internally and should not be used by the user.
            Use the `.apply` function in stead.

        '''
        # we assume all inputs are square. and the same for each batch
        # TODO: implement a check for this
        # Get total size of block diagonal matrix
        batch_size = inputs[0].size(0)
        sizes = [m.size(1) for m in inputs]
        idxs = list(np.cumsum([0]+sizes))
        total_size = int(idxs[-1])
        # Get start and end indices of blocks in matrix
        ctx.idxs = list(zip(idxs[:-1], idxs[1:]))
        # Get type of new matrix and create empty new matrix with total_size as shape
        M = zeros((batch_size, total_size, total_size), type=inputs[0].type())
        # Fill Blocks
        for (i, j), matrix in zip(ctx.idxs, inputs):
            M[:, i:j, i:j] = matrix
        return M

    @staticmethod
    def backward(ctx, grad_output):
        ''' BlockDiag backward

        The backward pass selects the relevant submatrices from the gradients of the
        batch block diagonal matrix and returns them as the requested gradients of the input
        matrices.

        Args:
            ctx (torch context): An object storing the indices of the submatrices
            grad_output: The gradients of the batch block diagonal matrix.

        Returns:
            The gradients of the submatrices of the batch block diagonal matrix.

        Note:
            This function is used by PyTorch internally and should not be used by the user.

        '''
        outputs = []
        for i, j in ctx.idxs:
            output = grad_output[:, i:j, i:j]
            outputs.append(output)
        return tuple(outputs)
