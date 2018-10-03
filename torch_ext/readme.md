[comment]: # (This is and automatically generated readme file)
[comment]: # (To edit this file, edit the docstring in the __init__.py file)
[comment]: # (And run the documentation: python -m photontorch.documentation)

# Torch Extensions for PhotonTorch

Since PhotonTorch is a photonic simulation framework in the first place,
we require some extra functionalities that PyTorch does not offer out of
the box.

Below you can find a short summary:

## [Autograd](autograd) Extensions
  * `block_diag`: a differentiable implementation of a block diagonal matrix

## [Neural Network](nn) Extensions
  * `[Buffer](nn.Buffer)`: A special kind of tensor that automatically will
be added to the `._buffers` attribute of the Module. Buffers are typically used as
parameters of the model that do not require gradients.

  * `[Module](nn.Module)`: Extends `torch.nn.Module`, with some extra features, such as
automatically registering a `[Buffer](.nn.Buffer)` in its `._buffers` attribute, modified
`.cuda()` calls and some extra functionalities.

  * `[BoundedParameter](nn.BoundedParameter)`: A bounded parameter is a special kind of
`torch.nn.Parameter` that is bounded between a certain range. Under the hood it registers
an unbounded weight in our torch_ext.nn.Module and a class property calculating the
desired parameter on the fly when it is asked by performing a scaled sigmoid on the weight.

## [Tensor](tensor) functions
Some non-differentiable, but useful functions that act on (or create) torch tensors.

