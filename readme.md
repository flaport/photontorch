# PhotonTorch

## Introduction
PhotonTorch is a photonic simulation engine based on the deep learning framework PyTorch.

## Features
PhotonTorch features CUDA enabled optimization of photonic circuits. It leverages the
deep learning framework PyTorch to view the photonic circuit as essentially a recurrent
neural network. This enables the use of native PyTorch optimizers to optimize the
(physical) parameters of your circuit.
## Roadmap
* Network visualizer
* Connection Matrix Network
* ~~Ring Networks~~
* ~~Trainable Delays~~
* ~~Bounded Parameter~~
* ~~Copying components~~
* ~~Network termination~~
* ~~Network of networks~~
* ~~CUDA enabled simulations~~
* ~~Possibility to ignore delays~~
* ~~Directional Coupler Networks~~
* ~~Batched simulations / training~~
* ~~Multi-wavelength simulations / training~~

## Dependencies
#### Required
* [`numpy`](http://www.numpy.org/)
* [`pytorch`](http://pytorch.org/)

#### Optional
* [`tqdm`](https://pypi.python.org/pypi/tqdm) (for progress bars)
* [`matplotlib`](https://matplotlib.org/) (for visualization)


## Copyright

© Floris Laporte
