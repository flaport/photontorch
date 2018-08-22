# PhotonTorch

## Introduction
PhotonTorch is a photonic simulation framework based on the deep learning framework PyTorch.

## Features
PhotonTorch features CUDA enabled optimization of photonic circuits. It leverages the
deep learning framework PyTorch to view the photonic circuit as essentially a recurrent
neural network. This enables the use of native PyTorch optimizers to optimize the
(physical) parameters of your circuit.

## Roadmap
* Ring Networks
* Network visualizer
* ~~Complex Connection Matrices~~
* ~~Connection Matrix Network~~
* ~~Matrix Networks~~
* ~~Bounded Parameter~~
* ~~Copying components~~
* ~~Network termination~~
* ~~Network of networks~~
* ~~CUDA enabled simulations~~
* ~~Possibility to ignore delays~~
* ~~Multi-wavelength simulations / training~~
* ~~Batched simulations / training~~

## Dependencies
* Python 2.7 (linux only), 3.6 or 3.7: from [Anaconda](http://www.anaconda.com/download/) [recommended]
* [`pytorch>=0.4.0`](http://pytorch.org/): `conda install pytorch -c pytorch` [linear algebra with backpropagation]
* [`numpy`](http://www.numpy.org/): `conda install numpy` [linear algebra]
* [`scipy`](http://www.scipy.org/): `conda install scipy` [basic signal processing]
* [`pytest`](http://docs.pytest.org/): `conda install pytest` [testing]
* [`tqdm`](http://pypi.python.org/pypi/tqdm): `conda install tqdm` [progress bars]
* [`matplotlib`](http://matplotlib.org/): `conda install matplotlib` [visualization]

## Documentation
You can generate the documentation by typing [only with python 3]
```
python -m photontorch.documentation.generate
```
in the folder that contains photontorch.

## Where to start
If you don't know where to start, start going through the notebooks in the examples folder.

## Tests
The tests are currently failing due to a big refactoring of the codebase.

## Copyright
© Floris Laporte
