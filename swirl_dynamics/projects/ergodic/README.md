## Stable Autoregressive Models for Chaotic Dynamical Systems via Ergodicity Inducing Regularization

## Abstract

We present ideas derived from ergodic theory for stabilizing learning dynamical system models.
Specifically, we introduce regularization that encourages learned models to preserve the same invariant measure as that of true system.
We analyze our approach on several well known dynamical systems, including the Lorenz 63 model and the Kuramoto–Sivashinsky (KS) equation.

## Getting started

We provide a [demo](./colabs/demo.ipynb) illustrating how to run the experiments in a notebook environment.

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
