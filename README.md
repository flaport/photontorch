# Photontorch

Photontorch is a photonic simulator for highly parallel simulation and
optimization of photonic circuits in time and frequency domain.
Photontorch features CUDA enabled simulation and optimization of
photonic circuits. It leverages the deep learning framework PyTorch to
view the photonic circuit as essentially a recurrent neural network.
This enables the use of native PyTorch optimizers to optimize the
(physical) parameters of the circuit.

- Floris Laporte ( [floris.laporte@ugent.be](mailto:floris.laporte@gmail.com) )
- Peter Bienstman ( [peter.bienstman@ugent.be](mailto:peter.bienstman@ugent.be) )
- Website: [http://photontorch.com](http://photontorch.com)

## Installation

### Stable version

Photontorch can be installed with pip:

```
pip install photontorch
```

### Development version

During development or to use the most recent Photontorch version,
clone the repository and link with pip:

```
git clone https://git.photontorch.com/photontorch.git
./install-git-hooks.sh # Unix [Linux/Mac/BSD/...]
install-git-hooks.bat  # Windows
pip install -e photontorch
```

During development, use pytest to run the tests from within the root
of the git-repository:

```
pytest tests
```

## Documentation

Read the full documentation here: [https://docs.photontorch.com](https://docs.photontorch.com)

## Dependencies

### Required dependencies

- Python 2.7 (Linux only) or 3.6+. It's recommended to use the [Anaconda](http://www.anaconda.com/download/) distribution.
- [`pytorch>=1.5.0`](http://pytorch.org/): `conda install pytorch` (see [pytorch.org](https://pytorch.org) for more installation options for your CUDA version)
- [`numpy`](http://www.numpy.org/): `conda install numpy`
- [`scipy`](http://www.scipy.org/): `conda install scipy`

### Optional (but recommended) dependencies

- [`tqdm`](http://pypi.python.org/pypi/tqdm): `conda install tqdm`
- [`networkx`](http://networkx.github.io): `conda install networkx`
- [`matplotlib`](http://matplotlib.org/): `conda install matplotlib`
- [`pytest`](http://docs.pytest.org/): `conda install pytest`
- [`sphinx`](https://www.sphinx-doc.org): `pip install sphinx nbsphinx`

## Reference

If you're using Photontorch in your work or feel in any way inspired by it,
we ask you to cite us in your work:

Floris Laporte, Joni Dambre, and Peter Bienstman. _"Highly parallel simulation
and optimization of photonic circuits in time and frequency domain based on the
deep-learning framework PyTorch."_ Scientific reports 9.1 (2019): 5918.

## License

Photontorch is available under an [Academic License](LICENSE). This
means that there are no restrictions on the usage in a purely
non-commercial or academic context. For commercial applications you
can always contact the authors.

Copyright © 2020, Floris Laporte - Universiteit Gent - Ghent University - [Academic License](LICENSE)
