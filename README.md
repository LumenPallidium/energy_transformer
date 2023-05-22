# Description

This is a tiny repository, but I have been reusing the energy transformer across multiple projects so I wanted it to live a pip-installable existence (for myself and others if they are interested).

This repository contains an implementation of [energy transformers](https://openreview.net/pdf?id=4nrZXPFN1c4), which may be the only Pytorch implementation at the moment. The jax implementation [can be found here](https://github.com/bhoov/energy-transformer-jax), this repository is a straightforward port of it, with some consolidation and adaption for PyTorch. The main file includes an example with the full self-supervised masked image reconstruction training used in the paper (except on CIFAR instead of Imagenet for speed). This example is optional and requires some extra (common) packages not installed during setup.

Briefly, an energy transformer is a variant of the transformer which runs a variant of attention in parallel with a Hopfield network. It is effectively recurrent, iteratively acting on its input as it descends the gradient of its energy function. The paper above contains the full mathematical details of the energy function. Note that, unlike a conventional transformer, this model has no feedforward layer: inputs have postional embedding added, then they are normalized and passed through the network; the input is iteratively modified by subtracting the network output then running the residual through the network (including normalization) again.

The Modern Hopfield variants (SoftmaxModernHopfield and BaseModernHopfield) that are used in the energy transformer are also available for import.

# Installation

To install this package, run:

```
pip install git+https://github.com/LumenPallidium/energy_transformer.git
```

The only requirement is Pytorch (>=2.0). If you run the optional masked image reconstruction pipeline example, you will also need torchvision, einops, matplotlib, and tqdm.The above PIP install command will install Pytorch, but I would reccomend installing on your own independently, so you can configure any neccesary environments, CUDA tools, etc.
