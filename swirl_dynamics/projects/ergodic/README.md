## Stable Autoregressive Models for Chaotic Dynamical Systems via Ergodicity Inducing Regularization

This repository contains the code for the paper "[DySLIM: Dynamics Stable Learning by Invariant Measure for Chaotic Systems](https://arxiv.org/abs/2402.04467)".

## Abstract

We present ideas derived from ergodic theory for stabilizing learning dynamical system models.
Specifically, we introduce regularization that encourages the learned models to preserve the same invariant measure as that of true system.
We analyze our approach on several well known dynamical systems, including the Lorenz 63 model, the Kuramoto–Sivashinsky (KS) equation, and Navier Stokes equation.

## Getting started

We provide a [demo](./colabs/demo.ipynb) illustrating how to run the experiments in a notebook environment.

## Training

For training you can use the configuration files provided in the config folder.

- `configs/lorenz63.py` for the Lorenz63 system
- `configs/ks_1d.py` for the Kuramoto–Sivashinsky system
- `configs/ns_2d.py` for Kolmogorov flow

The main entry point into running the experiments is `main.py`.

One can run the experiments with the following command:
```python
python main.py \
    --config=<path_to_config> \
    --workdir=<path_to_project_dir>

```

## Datasets

### Lorenz 63
For the Lorenz 63 system we generate data 'on-the-fly', and hence there is no need to download a dataset.

### KS
The dataset used for KS experiments are simulated using `jax-cfd` ([github](https://github.com/google/jax-cfd)).
We have made the generated datasets available to download (in `hdf5` format) on google cloud.

To download the datasets, first follow the [instructions](https://cloud.google.com/storage/docs/gsutil_install) to install the `gsutil` tool.
Then simply run the following command (see [here](https://cloud.google.com/storage/docs/gsutil/commands/cp) for more CLI options) to copy the dataset to a local directory

```.bash
gsutil cp gs://gresearch/swirl_dynamics/hdf5/pde/1d/`ks_trajectories.hdf5`
```
