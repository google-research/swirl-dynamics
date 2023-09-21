# Evolve Smoothly, Fit Consistently: Learning Smooth Latent Dynamics For Advection-Dominated Systems

## Abstract

We present a data-driven, space-time continuous framework to learn surrogate models for complex physical systems described by advection-dominated partial differential equations. Those systems have slow-decaying Kolmogorov $$n$$-width that hinders standard methods, including reduced order modeling, from producing high-fidelity simulations at low cost. In this work, we construct hypernetwork-based latent dynamical models directly on the parameter space of a compact representation network. We leverage the expressive power of the network and a specially designed consistency-inducing regularization to obtain latent trajectories that are both low-dimensional and smooth. These properties render our surrogate models highly efficient at inference time. We show the efficacy of our framework by learning models that generate accurate multi-step rollout predictions at much faster inference speed compared to competitors, for several challenging examples.

## Datasets

The datasets used for this work are simulated using `jax-cfd` ([github](https://github.com/google/jax-cfd)). We have made the generated datasets available to download (in `hdf5` format) on google cloud.

To download the datasets, first follow the [instructions](https://cloud.google.com/storage/docs/gsutil_install) to install the `gsutil` tool. Then simply run the following command (see [here](https://cloud.google.com/storage/docs/gsutil/commands/cp) for more CLI options) to copy the dataset to a local directory

```.bash
gsutil cp gs://gresearch/swirl_dynamics/hdf5/pde/1d/{dataset_path}
```
where `{dataset_path}` is one of the paths listed in the table below. The dataset will be downloaded to your current directory.

| Dataset                  | `{dataset_path}`
| :---------------------   | :--------------------------
| Burgers'                 | `burgers_trajectories.hdf5`
| Kuramoto-Sivashinsky     | `ks_trajectories.hdf5`
| Korteweg-De Vries        | `kdv_trajectories.hdf5`

## Getting Started

We provide a [demo](./colabs/demo.ipynb) illustrating how to run the code in a notebook environment.

## Citation

If you extend or use our work, please cite the [paper](https://arxiv.org/abs/2301.10391):

```
@inproceedings{wan2023evolve,
  title={Evolve Smoothly, Fit Consistently: Learning Smooth Latent Dynamics For Advection-Dominated Systems},
  author={Zhong Yi Wan and Leonardo Zepeda-Nunez and Anudhyan Boral and Fei Sha},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```
