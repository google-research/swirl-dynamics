# Dynamical-generative downscaling of climate model ensembles

This repository includes the source code used to design, train, and evaluate the
Regional Residual Diffusion-based Downscaling (R2-D2) model described in the
paper "Dynamical-generative downscaling of climate model ensembles"
(Lopez-Gomez et al, [2024](https://doi.org/10.48550/arXiv.2410.01776)). The
model weights of a pre-trained R2-D2 model, as well as pre-processed input and
evaluation data, are available on [Google Cloud](https://console.cloud.google.com/storage/browser/dynamical_generative_downscaling).

The R2-D2 model is designed to take coarse-resolution climate data over a
limited-area region and provide plausible high-resolution samples of the input
fields over the same domain. The method relies on loosely paired trajectories of
coarse-resolution climate fields and a higher-fidelity, high-resolution version
of the same fields for training. Such paired trajectories may be obtained from
dynamically downscaled climate projections, which track the trajectory of a
coarse-resolution input, or by forcing a coarse-resolution model to track a
high-fidelity, high-resolution observational dataset. This paper explores the
first approach, which is applicable to future climate downscaling.

## Application to climate projections over the Western United States

The paper explores the application of R2-D2 to downscale future
climate projections over the western United States, under the SSP3-7.0 forcing
scenario. The data used to train and evaluate the models comes from the WUS-D3
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

The easiest way to learn how to use the models and navigate the codebase is
through the demo colabs. The starting point for those interested in performing
inference with the model is the Google Colab *OS_downscaling_inference.ipynb*.
This first colab presents a step-by-step guide of installing the code, the
expected format of input and output datasets, instantiating an R2-D2 model given
model weights, and performing inference.

Colab *OS_multimodel_distribution_analysis.ipynb* inspects pre-generated
downscaled datasets and showcases a few of the evaluation metrics used in
Lopez-Gomez et al ([2024](https://doi.org/10.48550/arXiv.2410.01776)).

## Data

We consider data at hourly resolution over the western United States, at
45 km resolution (`hourly_d01`), and at 9 km resolution (`hourly_d02`). We
include in parentheses the nomenclature used for these datasets in our source
code.

For each forcing ESM, we use high-resolution data (`hourly_d02.zarr`), which
serves as the label; and low-resolution data interpolated to the high-resolution
grid (`hourly_d01_cubic_interpolated_to_d02.zarr`), which serves as the input.
Residual models are trained using the difference between the high-resolution
data and the interpolated low-resolution data as the target. For normalization,
we also compute the mean and standard deviation of each field over a certain
period of time, in our case 2020-2090.

We include samples of this dataset over the evaluation period on [Google Cloud](https://console.cloud.google.com/storage/browser/dynamical_generative_downscaling).

## Modeling framework

The R2-D2 model is a generative diffusion model. The learned denoising network
is parameterized by a neural network with a UNet architecture. The modeling
framework used to design the model is `JAX`. The
[`swirl-dynamics`](https://github.com/google-research/swirl-dynamics) codebase
contains numerous implementations of popular neural network architectures
(e.g., UNet, ViT), as well as model templates for probabilistic diffusion.
The package `orbax` is used for checkpointing. The model was trained and
evaluated using NVIDIA A100 GPUs, so using architectures with similar or greater
memory (e.g., NVIDIA H100) is recommended to avoid out-of-memory issues.

## License

Licensed under the Apache License, Version 2.0.
