''' Extensions and utilities for the photontorch package '''


## custom torch differentiable functions
from .autograd import block_diag


## custom neural network functions
from .nn import Module
from .nn import Parameter
from .nn import BoundedParameter


## Custom Tensor Functions
from .tensor import zeros
from .tensor import where

