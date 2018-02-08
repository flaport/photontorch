''' Extensions and utilities for the photontorch package '''


## custom torch differentiable functions

from .autograd import block_diag


## Custom Tensor Functions

from .tensor import zeros
from .tensor import where


## Other functions (not differentiable):

from .functions import inv_sigmoid