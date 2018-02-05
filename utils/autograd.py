''' Torch autograd function extensions '''

#############
## Imports ##
#############

## Torch
import torch
from torch.autograd import Variable
from torch.autograd import Function

## Other
import numpy as np

## Relative
from .tensor import zeros


###################
## PseudoInverse ##
###################

## PseudoInverse Function
def pinv(input, rcond=1e-8):
    '''
    Calculate the generalized inverse of 2D Torch FloatTensor using the
    singular-value decomposition (SVD) and including all
    *large* singular values.

    Parameters
    ----------
    input : (M, N) tensor. Matrix to be pseudo-inverted.
    rcond : float. Cutoff for small singular values.

    Returns
    -------
    B : (N, M) FloatTensor. The pseudo-inverse of `input`.
    '''
    return PseudoInverse().apply(input)

class PseudoInverse(Function):
    '''
    The Moore-Penrose pseudo inverse Functional

    Note
    ----
    Do not use this torch Function directly.
    Use `pinv` instead.
    '''
    @staticmethod
    def forward(ctx, input, rcond=1e-8):
        '''
        The forward method calculates the pseudo-inverse and
        saves the necessary variables for a backward pass
        '''
        cutoff = rcond*input.max()
        # numpy way:
        # np_inverse = np.linalg.pinv(np.array(input.cpu().numpy(), dtype=np.float32), rcond=rcond)
        # inverse = torch.from_numpy(np_inverse)

        # torch way
        U,S,V = torch.svd(input)
        selection = (torch.abs(S) <= cutoff)
        if selection.all():
            invS = torch.diag(torch.zeros_like(S))
        else:
            invS = torch.diag((1./S)*(~selection).type(input.type()))
        inverse = V.mm(invS.mm(U.transpose(1,0)))

        # save the inverse for the backward pass
        ctx.inverse = inverse

        # return the inverse
        return inverse

    @staticmethod
    def backward(ctx, grad_output):
        '''
        The backward pass uses the formula for a (normal) inverse:
        inv'(M) = -inv(M)@M'@inv(M).

        We assume this formula also works for the pseudo-inverse.
        '''

        # some debugging with torch.autograd.gradcheck shows that we need
        # to transpose the inverse, for the desired backward pass to work.
        # So far, I have no idea why this is the case.
        inverse = Variable(ctx.inverse, requires_grad=True).transpose(1,0) # we need a variable to work with

        # return the gradient
        return -inverse.mm(grad_output.mm(inverse))


####################
## Block Diagonal ##
####################

def block_diag(*inputs):
    '''
    Create a block diagonal matrix from provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    Parameters
    ----------
    A, B, C, ... : 2D square Torch FloatTensor Variables

    Returns
    -------
    D : 2d square Torch FloatTensor Variable with A, B, C, ... on the diagonal
    '''
    return BlockDiag().apply(*inputs)

class BlockDiag(Function):
    '''
    Construct a block diagonal matrix from a sequence of inputs

    Note
    ----
    Do not use this torch Function directly.
    Use `block_diag` instead.
    '''
    @staticmethod
    def forward(ctx, *inputs):
        '''
        The forward method creates the block diagonal matrix and
        saves the locations of the submatrices for the backward pass
        '''
        # we assume all inputs are square. TODO: implement a check for this
        # Get total size of block diagonal matrix
        sizes = [m.size(0) for m in inputs]
        idxs = list(np.cumsum([0]+sizes))
        total_size = int(idxs[-1])
        # Get start and end indices of blocks in matrix
        ctx.idxs = list(zip(idxs[:-1], idxs[1:]))
        # Get type of new matrix and create empty new matrix with total_size as shape
        M = zeros(total_size, total_size, type=inputs[0].type())
        # Fill Blocks
        for (i, j), matrix in zip(ctx.idxs, inputs):
            M[i:j,i:j] = matrix
        return M

    @staticmethod
    def backward(ctx, M):
        '''
        The backward pass selects the relevant submatrices of the gradient
        '''
        outputs = []
        for i,j in ctx.idxs:
            output = M[i:j,i:j]
            outputs.append(output)
        return tuple(outputs)