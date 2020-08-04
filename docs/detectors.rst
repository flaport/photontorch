detectors
=========

lowpassdetector
---------------

responsivity
^^^^^^^^^^^^

The detected photo-current :math:`I_p~[C/s]` can be described in terms of the
optical power :math:`P_o~[W]` through the quantum efficiency :math:`\eta` and
the carrier frequency :math:`f=c/\lambda~[1/s]` of the light:

.. math::

    \begin{align*}
    I_p = \eta \frac{q}{h f}P_o = \eta \frac{q \lambda}{h c}P_o
    \end{align*}

However, it's often more useful to describe it in terms of a single quantity
:math:`r~[A/W]`:

.. math::

    \begin{align*}
     I_p &= rP_o,
    \end{align*}

where we defined :math:`r~[A/W]` as the responsivity of the detector. At a
wavelength of :math:`1550nm` and for a quantum efficiency of :math:`\eta=1`, we
have:

.. math::

    \begin{align*}
    r &= \eta \frac{q \lambda}{h c} = 1.25 A/W,
    \end{align*}

hence a default value of 1 for the responsivity is appropriate.

implementation
^^^^^^^^^^^^^^

.. automodule:: photontorch.detectors.lowpassdetector
   :members:
   :undoc-members:
   :show-inheritance:

photodetector
-------------

responsivity
^^^^^^^^^^^^

As mentioned above, a responsivity of :math:`r=1 A/W` is a valid value, but almost
always it will be lower.  If the detected signal is not as noisy as expected,
you probably used a too high responsivity.

thermal noise
^^^^^^^^^^^^^
On top of the detected current, thermal noise should be added. The thermal
noise variance :math:`\sigma_t^2` [:math:`C^2/s^2`] is given by:

.. math::

    \begin{align*}
        \sigma_t^2 &= N_t^2 f_c,
    \end{align*}

with :math:`f_c~[1/s]` the cutoff frequency of the signal and
:math:`N_t~[C/\sqrt s]` the thermal noise spectral density. This represenation
(which is the representation used in VPI for example) requires you to use a
quantity with :math:`N_t~[C/\sqrt s]` with weird units. We therefore prefer
another representation (see for example
`wikipedia <https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise>`_):

.. math::

    \begin{align*}
    \sigma_t^2 = \frac{4kT}{R_L} f_c
    \end{align*}

To find a good default value for the load resistance :math:`R_L`, we can equate
this equation with the previous one using the default VPI value for
:math:`N_t=10^{-11}C/\sqrt s`:

.. math::

    \begin{align}
    R_L = 166 \Omega
    \end{align}

where we assumed for the room temparture :math:`T=300K`.

shot noise
^^^^^^^^^^
The shot noise variance :math:`\sigma_t^2` [:math:`C^2/s^2`] for a PIN
photodetector is given by:

.. math::

    \begin{align*}
    \sigma_s^2 = 2 q (I_p + I_d)f_c
    \end{align*}

with :math:`q~[C]` the elementary charge and :math:`I_d` the dark current of
the photodetector. Notice that this power is dependent on the photocurrent
:math:`I_p~[C/s]` itself.

In some representations, :math:`\mu_p=\left< I_p \right>~[C/s]` is used
instead. We choose not to take this average to have a more accurate
reprentation. Moreover, low-pass filtering already takes a kind of moving
average into account.

total noise
^^^^^^^^^^^
The total post-detection noise variance :math:`\sigma_p^2` [:math:`C^2/s^2`] now depends on
the thermal noise variance :math:`\sigma_t^2` and the shot noise variance
:math:`\sigma_s^2`:

.. math::

    \begin{align*}
    \sigma_p^2 &= \sigma_t^2 + \sigma_s^2
    \end{align*}

implementation
^^^^^^^^^^^^^^

.. automodule:: photontorch.detectors.photodetector
   :members:
   :undoc-members:
   :show-inheritance:

