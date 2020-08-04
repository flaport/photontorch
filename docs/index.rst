Photontorch
===========

:ref:`examples` · :ref:`documentation` · :ref:`genindex` · :ref:`modindex`

Photontorch is a photonic simulator for highly parallel simulation and
optimization of photonic circuits in time and frequency domain.
Photontorch features CUDA enabled simulation and optimization of
photonic circuits. It leverages the deep learning framework PyTorch to
view the photonic circuit as essentially a recurrent neural network.
This enables the use of native PyTorch optimizers to optimize the
(physical) parameters of the circuit.


Installation
------------

Stable version
^^^^^^^^^^^^^^

Photontorch can be installed with pip::

    pip install photontorch

Development version
^^^^^^^^^^^^^^^^^^^
During development or to use the most recent Photontorch version,
clone the repository and link with pip::

    git clone https://git.photontorch.com/photontorch.git
    ./install-git-hooks.sh # Unix [Linux/Mac/BSD/...]
    install-git-hooks.bat  # Windows
    pip install -e photontorch

During development, use pytest to run the tests from within the root
of the git-repository::

    pytest tests


Dependencies
------------

Required dependencies
^^^^^^^^^^^^^^^^^^^^^

    - Python 2.7 (linux only) or 3.6+.  It's recommended to use the `Anaconda <https://www.anaconda.com/download>`_ distribution.
    - `pytorch <https://pytorch.org>`_ >=1.5.0:  ``conda install pytorch`` (see `pytorch.org <https://pytorch.org>`_ for more installation options for your CUDA version)
    - `numpy <https://numpy.org>`_ : ``conda install numpy``
    - `scipy <https://scipy.org>`_ : ``conda install scipy``

Recommended dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

    - `tqdm <https://pypi.python.org/pypi/tqdm>`_ : ``conda install tqdm`` [progress bars]
    - `matplotlib <https://matplotlib.org>`_ : ``conda install matplotlib`` [network visualization]
    - `networkx <https://networkx.github.io>`_ : ``conda install networkx`` [visualization]
    - `pytest <http://docs.pytest.org>`_ : ``conda install pytest`` [to run tests]
    - `pandoc <https://pandoc.org>`_: ``conda install pandoc`` [to generate docs]
    - `sphinx <https://www.sphinx-doc.org>`_ : ``pip install sphinx nbsphinx`` [to generate docs]


Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   examples
   photontorch


Reference
---------
If you're using Photontorch in your work or feel in any way inspired by it,
please be so kind to cite us in your work.

Floris Laporte, Joni Dambre, and Peter Bienstman. *"Highly parallel simulation
and optimization of photonic circuits in time and frequency domain based on the
deep-learning framework PyTorch."* Scientific reports 9.1 (2019): 5918.


Where to go from here?
----------------------
Check out the first example: `A brief introduction to Photontorch <examples/00_introduction_to_photontorch.html>`_.


License
-------

Photontorch is available under an
`Academic License <https://github.com/flaport/photontorch/blob/master/LICENSE>`_.
This means that there are no restrictions on the usage in a purely
non-commercial or academic context. For commercial applications you can
always contact the authors.

Copyright © 2020, Floris Laporte - Universiteit Gent - Ghent University -
`Academic License <https://github.com/flaport/photontorch/blob/master/LICENSE>`_.
