networks
========

The Network is the core of Photontorch.

This is where everything comes together.  The Network is a special kind of
torch.nn.Module, where all subcomponents are automatically initialized and
connected in the right way.

reduction of S-matrix
---------------------

Each component can be described in terms of it's S matrix. For such a
component, we have that the output fields :math:`\bf x_{\rm out}` are connected to
the input fields :math:`x_{\rm in}` through a scattering matrix:

.. math::

    x_{\rm out} = S \cdot x_{\rm in}

For a network of components, the field vectors :math:`x` will just be stacked on top of each other,
and the S-matrix will just be the block-diagonal matrix of the S-matrices of the
individual components. However, to connect the output fields to each other, we need
a connection matrix, which connects the output fields of the individual components
to input fields of other components in the fields vector :math:`x`:

.. math::

    x_{\rm in} = C \cdot x_{\rm out}

a simulation (without delays) can thus simply be described by:

.. math::

    x(t+1) = C\cdot S\cdot x(t)

However, when delays come in the picture, the situation is a bit more complex.
We then split the fields vector :math:`x` in a memory_containing part (mc) and a
memory-less part (ml):

.. math::

    \begin{pmatrix}x^{\rm mc} \\x^{\rm ml} \end{pmatrix}(t+1) =
    \begin{pmatrix} C^{\rm mcmc} & C^{\rm mcml} \\ C^{\rm mlmc} & C^{\rm mlml} \end{pmatrix}
    \begin{pmatrix} S^{\rm mcmc} & S^{\rm mcml} \\ S^{\rm mlmc} & S^{\rm mlml} \end{pmatrix}
    \cdot\begin{pmatrix}x^{\rm mc} \\x^{\rm ml} \end{pmatrix}(t)

Usually, we are only interested in the memory-containing nodes, as memory-less nodes
should be connected together and act all at once. After some matrix algebra we arrive at

.. math::

    \begin{align}
    x^{\rm mc}(t+1) &= \left( C^{\rm mcmc} + C^{\rm mcml}\cdot S^{\rm mlml}\cdot
    \left(1-C^{\rm mlml}S^{\rm mlml}\right)^{-1} C^{\rm mlmc}\right)S^{\rm mcmc} x^{\rm mc}(t) \\
    &= C^{\rm red} x^{\rm mc}(t),
    \end{align}

Which defines the reduced connection matrix used in the simulations.

complex matrix inverse
----------------------

PyTorch still does not allow complex valued Tensors. Therefore, the above
equation was completely rewritten with matrices containing the real and
imaginary parts.  This would be fairly straightforward if it were not for the
matrix inverse in the reduced connection matrix:

.. math::

    \begin{align}
    P^{-1} = \left(1-C^{\rm mlml}S^{\rm mlml}\right)^{-1}
    \end{align}

unfortunately for complex matrices :math:`P^{-1} \neq {\rm real}(P)^{-1} + i{\rm
imag}(P)^{-1}`, the actual case is a bit more complicated.

It is however, pretty clear from the equations that the :math:`{\rm real}(P)^{-1}`
will always exist, and thus we can write for the real and imaginary part of
:math:`P^{-1}`:


.. math::

    \begin{align}
    {\rm real}(P^{-1}) &= \left({\rm real}(P) + {\rm imag}(P)\cdot {\rm real}(P)^{-1}
    \cdot {\rm imag}(P)\right)^{-1}\\
    {\rm real}(P^{-1}) &= -{\rm real}(P^{-1})\cdot {\rm imag}(P) \cdot {\rm real}(P)^{-1}
    \end{align}

This equation is valid, even if :math:`{\rm imag}(P)^{-1}` does not exist.

network
-------

.. automodule:: photontorch.networks.network
   :members:
   :undoc-members:
   :show-inheritance:

clements
--------

.. automodule:: photontorch.networks.clements
   :members:
   :undoc-members:
   :show-inheritance:

reck
----

.. automodule:: photontorch.networks.reck
   :members:
   :undoc-members:
   :show-inheritance:

rings
-----

.. automodule:: photontorch.networks.rings
   :members:
   :undoc-members:
   :show-inheritance:

