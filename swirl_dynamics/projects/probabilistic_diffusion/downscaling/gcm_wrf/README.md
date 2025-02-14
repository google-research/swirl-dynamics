# Dynamical-generative downscaling of climate model ensembles

This repository includes the source code used to design, train, and evaluate the
Regional Residual Diffusion-based Downscaling (R2-D2) model described in the
paper "Dynamical-generative downscaling of climate model ensembles"
(Lopez-Gomez et al, [2024](https://doi.org/10.48550/arXiv.2410.01776)). The
model weights of a pre-trained R2-D2 model instance, as well as pre-processed
input and evaluation data, are available on [Google Cloud](https://console.cloud.google.com/storage/browser/dynamical_generative_downscaling).

R2-D2 is a generative diffusion model designed to downscale coarse-resolution
climate data over a limited-area region, providing higher-resolution samples of
the input fields within the same domain. Generative downscaling is an efficient
alternative to dynamical downscaling that enables capturing the uncertainty of
the downscaling process, unlike deterministic statistical downscaling methods.

The method relies on loosely paired trajectories of coarse-resolution climate
fields and corresponding high-fidelity, high-resolution versions of those
fields. These paired trajectories can be generated from dynamically downscaled
climate projections, which follow the evolution of a coarse-resolution input, or
by forcing a coarse-resolution model to align with a high-fidelity,
high-resolution observational dataset. This paper focuses on the former
approach, which is applicable to future climate downscaling.

## Application to climate projections over the Western United States

This paper explores the application of R2-D2 to downscale future climate
projections over the western United States under the SSP3-7.0 forcing scenario.
We demonstrate that R2-D2 instances, trained on data from a single climate model
projection, can accurately downscale projections from other climate models. This
is achieved provided that the inputs to R2-D2 are first dynamically downscaled
to an intermediate coarse resolution by a regional climate model. This
dynamical-generative downscaling framework enables accurate downscaling of
climate model ensembles at a fraction of the cost of pure dynamical downscaling.

The data used to train and evaluate R2-D2 comes from the WUS-D3
dataset, described by Rahimi et al [(2024)](https://doi.org/10.5194/gmd-17-2265-2024).
The WUS-D3 dataset is openly available [here](https://registry.opendata.aws/wrf-cmip6/)
All references to the dataset should follow the guidelines provided by the
authors of the dataset.

In our [study](https://doi.org/10.48550/arXiv.2410.01776), we consider in
particular the bias corrected paired trajectories of 8 different Earth System
Models (ESMs): CanESM5, MPI-ESM1-2-HR, EC-Earth3-Veg, UKESM1-0-LL, ACCESS-CM2,
MIROC6, TaiESM1, and NorESM2-MM. Bias correction follows the procedure described
by Risser et al [(2024)](https://doi.org/10.1029/2023GL105979).

## Getting started

The easiest way to learn how to use the model and navigate the codebase is
through the demo Colab notebooks. The starting point for those interested in
performing inference with the model is the Google Colab notebook
*OS_downscaling_inference.ipynb*. This initial notebook presents a step-by-step
guide to installing the code, the expected format of input and output datasets,
instantiating an R2-D2 model given model weights, and performing inference.

Colab *OS_multimodel_distribution_analysis.ipynb* inspects pre-generated
downscaled datasets and showcases a few of the evaluation metrics used in
Lopez-Gomez et al ([2024](https://doi.org/10.48550/arXiv.2410.01776)).

## Data

We consider data at hourly resolution over the western United States, at
45 km resolution (`hourly_d01`), and at 9 km resolution (`hourly_d02`). We
include in parentheses the nomenclature used for these datasets in our source
code.

For each forcing ESM, we use high-resolution data (`hourly_d02.zarr`), which
serves as the label, and low-resolution data interpolated to the high-resolution
grid (`hourly_d01_cubic_interpolated_to_d02.zarr`), which serves as the input.
R2-D2 models are trained to predict samples of the distribution of the
difference between the high-resolution data and the interpolated low-resolution
data as the target. For normalization, we also compute the mean and standard
deviation of each field over a specified period, in our case, 2020-2090.

We include samples of this dataset over the evaluation period on [Google Cloud](https://console.cloud.google.com/storage/browser/dynamical_generative_downscaling).

## Modeling framework

The R2-D2 model is a generative diffusion model. The learned denoising network
is parameterized by a neural network with a UNet architecture. The modeling
framework used to design the model is `JAX`. The
[`swirl-dynamics`](https://github.com/google-research/swirl-dynamics) codebase
contains numerous implementations of popular neural network architectures
(e.g., UNet, ViT), as well as model templates for probabilistic diffusion. The
package `orbax` is used for checkpointing. The model was trained and
evaluated using NVIDIA A100 GPUs, so using architectures with similar or greater
memory (e.g., NVIDIA H100) is recommended to avoid out-of-memory issues.

[This](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/probabilistic_diffusion/colabs/demo.ipynb) demo Colab notebook illustrates how
to instantiate, train, and perform inference with probabilistic diffusion models
using `swirl-dynamics`.

## License

Licensed under the Apache License, Version 2.0.
