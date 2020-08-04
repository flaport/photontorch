neural network (nn) extensions
==============================

Since PhotonTorch is a photonic simulation framework in the first place, we
require some extra functionalities that PyTorch does not offer out of the box.

Below you can find a short summary:

* `photontorch.nn.Buffer`: A special kind of tensor that automatically will be
  added to the `._buffers` attribute of the Module. Buffers are typically used
  as parameters of the model that do not require gradients.

* `photontorch.nn.Module`: Extends `torch.nn.Module`, with some extra features,
  such as automatically registering a `[Buffer](.nn.Buffer)` in its `._buffers`
  attribute, modified `.cuda()` calls and some extra functionalities.

* `photontorch.nn.BoundedParameter`: A bounded parameter is a special kind of
  `torch.nn.Parameter` that is bounded between a certain range. Under the hood
  it registers an unbounded weight in the `photontorch.nn.Module` and a class
  property calculating the desired parameter on the fly when it is asked by
  performing a scaled sigmoid on the weight.

* `photontorch.nn.MSELoss`: mean squared error loss function which takes
  latency differences between input stream and target stream into account.

* `photontorch.nn.BERLoss`: bit error rate loss function which takes
  latency differences between input stream and target stream into account. The
  resulting BER is not differentiable.

* `photontorch.nn.BitStreamGenerator`: a bitstream generator with proper
  lowpass filtering.


nn
--

.. automodule:: photontorch.nn.nn
   :members:
   :undoc-members:
   :show-inheritance:

