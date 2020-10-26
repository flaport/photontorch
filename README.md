# Photontorch

Photontorch is a photonic simulator for highly parallel simulation and
optimization of photonic circuits in time and frequency domain.
Photontorch features CUDA enabled simulation and optimization of
photonic circuits. It leverages the deep learning framework PyTorch to
view the photonic circuit as essentially a recurrent neural network.
This enables the use of native PyTorch optimizers to optimize the
(physical) parameters of the circuit.

- Floris Laporte [[floris.laporte@ugent.be](mailto:floris.laporte@gmail.com)]
- Website: [photontorch.com](http://photontorch.com)

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

- Python 2.7 (Linux only) or 3.6+. It's recommended to use the [Anaconda](http://www.anaconda.com/download) distribution.
- [`pytorch>=1.5.0`](http://pytorch.org): `conda install pytorch` (see [pytorch.org](https://pytorch.org) for more installation options for your CUDA version)
- [`numpy`](http://www.numpy.org): `conda install numpy`
- [`scipy`](http://www.scipy.org): `conda install scipy`

### Optional (but recommended) dependencies

- [`tqdm`](https://github.com/tqdm/tqdm): `conda install tqdm` [progress bars]
- [`networkx`](http://networkx.github.io): `conda install networkx` [network visualization]
- [`matplotlib`](http://matplotlib.org): `conda install matplotlib` [visualization]
- [`pytest`](http://docs.pytest.org): `conda install pytest` [to run tests]
- [`pandoc`](https://pandoc.org): `conda install pandoc` [to generate docs]
- [`sphinx`](https://www.sphinx-doc.org): `pip install sphinx nbsphinx` [to generate docs]
- [`torch-lfilter`](https://github.com/flaport/torch_lfilter): `pip install torch-lfilter` [faster lfilter for detectors]

## Reference

If you're using Photontorch in your work or feel in any way inspired by it,
we ask you to cite us in your work:

Floris Laporte, Joni Dambre, and Peter Bienstman. _"Highly parallel simulation
and optimization of photonic circuits in time and frequency domain based on the
deep-learning framework PyTorch."_ Scientific reports 9.1 (2019): 5918.

## Known issues

- Complex tensor support. Complex tensors are not supported in
  PyTorch/Photontorch. Wherever complex tensors would be applicable,
  Photontorch expects a real-valued tensor with the real and imag part
  stacked in the first dimension. The Photontorch issue can be
  followed [here](https://github.com/flaport/photontorch/issues/4) and
  the PyTorch issue [here](https://github.com/pytorch/pytorch/issues/755).
- Sparse tensor support. A lot of memory usage can probably be avoided
  when transitioning to a sparse tensor representation for the connection matrices and
  scatter matrices. The Photontorch issue can be followed [here](https://github.com/flaport/photontorch/issues/5)

## License

Photontorch used to be available under a custom Academic License, but Since October
2020, Photontorch is now fully open source and available under the [AGPLv3](LICENSE). 

Copyright © 2020, Floris Laporte - UGent - [AGPLv3](LICENSE)

