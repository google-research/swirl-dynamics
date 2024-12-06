# Dynamical-generative downscaling of climate model ensembles

This repository includes the source code used to design, train, and evaluate the
Regional Residual Diffusion-based Downscaling (R2-D2) model described in the
paper "Dynamical-generative downscaling of climate model ensembles" [arXiv](
https://doi.org/10.48550/arXiv.2410.01776).

The R2-D2 model is designed to take coarse-resolution climate data over a
limited-area region and provide plausible high-resolution samples of the input
fields over the same domain. The method relies on paired trajectories of
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
dataset, described by Rahimi et al [(2023)](https://doi.org/10.5194/gmd-17-2265-2024).
The WUS-D3 dataset is openly available [here](https://registry.opendata.aws/wrf-cmip6/)
All references to the dataset should follow the guidelines provided by the
authors of the dataset.

In our [study](https://doi.org/10.48550/arXiv.2410.01776), we consider in
particular the bias corrected paired trajectories of 8 different Earth System
Models (ESMs): CanESM5, MPI-ESM1-2-HR, EC-Earth3-Veg, UKESM1-0-LL, ACCESS-CM2,
MIROC6, TaiESM1, and NorESM2-MM. Bias correction follows the procedure described
by Risser et al [(2024)](https://doi.org/10.1029/2023GL105979).


## Data

We consider data at hourly resolution over the western United States, at
45 km resolution (`d01` in the bucket), and at 9 km resolution (`d02`). We
include in parantheses the nomenclature used for these datasets in our source
code in the following paragraph.

For each forcing ESM, we use high-resolution data (`hourly_d02.zarr`), which
serves as the label; and low-resolution data interpolated to the high-resolution
grid (`hourly_d01_cubic_interpolated_to_d02.zarr`), which serves as the input.
Residual models are trained using the difference between the high-resolution
data and the interpolated low-resolution data as the target.

For normalization, we also want to compute the mean and standard deviation over
a certain period of time, for example 2020-2090.

## Modeling framework

The R2-D2 model is a generative diffusion model. The learned denoising network
is parameterized by a neural network with a UNet architecture. The modeling
framework used to design the model is `JAX`. The package `orbax` is used for
checkpointing. The model was trained and evaluated using NVIDIA A100 GPUs, so
using architectures with similar or greater memory (e.g., NVIDIA H100) is
recommended to avoid out-of-memory issues.

