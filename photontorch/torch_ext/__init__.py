""" Torch Extensions for PhotonTorch

Since PhotonTorch is a photonic simulation framework in the first place,
we require some extra functionalities that PyTorch does not offer out of
the box.

Below you can find a short summary:

  * ``block_diag``: a differentiable implementation of a block diagonal matrix
      performed over a batch of matrices.
  * ``BoundedParameter``: A bounded parameter is a special kind of
      ``torch.nn.Parameter`` that is bounded between a certain range.
  * ``Buffer``: A special kind of tensor that automatically will
      be added to the ``._buffers`` attribute of the Module. Buffers are typically
      used as parameters of the model that do not require gradients.
  * ``Module``: Extends ``torch.nn.Module``, with some extra
      features, such as automatically registering a ``Buffer`` in its
      ``._buffers`` attribute, modified ``.cuda()`` calls and some extra
      functionalities.
  * ``BitStreamGenerator``: A simple class that generates random bitstreams.
  * ``BERLoss``: A Module that calculates the bit error rate between two bitstreams.
  * ``MSELoss``: A Module that calculates the mean squared error between two bitstreams.

"""


## custom torch differentiable functions
from .autograd import block_diag


## custom neural network functions [not imported]
# torch_ext.nn

## Custom Tensor Functions
from .tensor import where

# custom functional additions
from .functional import BERLoss
from .functional import MSELoss
from .functional import BitStreamGenerator
