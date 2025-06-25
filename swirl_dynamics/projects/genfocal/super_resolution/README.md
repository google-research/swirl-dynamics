# GenFocal: Super-resolution with Diffusion Models

This folder contains the modules and scripts to run the super-resolution (SR)
step in [GenFocal](https://arxiv.org/abs/2412.08079).

The entry-point scripts are:

- `run_training.py`: Runs training for a SR denoising model.
- `run_sampling.py`: Runs sampling that applies a trained SR model to generate
high-resolution trajectories over an extended period of time.

We provide the configurations that lead to the results in paper in the `configs`
subfolder, specifying input files and parameter values:

- `conus_train.yaml`: training configuration for the Conterminous United States
(CONUS) region.
- `nao_train.yaml`: training configuration for the North Atlantic Ocean (NAO)
region.
- `conus_sample.yaml`: sampling configuration for CONUS.
- `nao_sample.yaml`: sampling configuration for NAO.

These configs can be directly passed to the entry-point scripts via a command
line flag.
The config schemas are defined in `configs/schema.py`, whose docstrings provide
detailed explanations for all parameters.

We also provide a demo colab in `colabs/era5_demo.ipynb`,
which contains instructions to install necessary dependencies, run sampling
and inspect outputs.

## Workflow

### Model training

#### Data preparation

Our SR training data comprises three components:

- Low-resolution (1.5-degree spatial and daily mean) input data.
- Corresponding high-resolution (0.25-degree spatial and bi-hourly) outputs. By
default, we assume this dataset has been pre-processed offline:
  - From the raw high-resolution data, we subtract the corresponding
  interpolated (linear in space and replicated in time) low-resolution input.
  This way we model the residual between the low- and high-resolution data,
  preventing interference with any non-stationary, climate change signal in
  the low-resolution input.
  - The residuals are normalized by their climatology, which is specific to
  the time of day and day of the year, computed over a reference
  historical period.
- Low-resolution reference statistics, including mean and standard deviation,
aggregated over all days within the reference period.

All datasets are assumed to be stored in `zarr` format. We provide some example
datasets in a [Google Cloud bucket](https://console.cloud.google.com/storage/browser/genfocal/super_resolution/training/example_datasets).

To download the example datasets, use
[gsutil](https://cloud.google.com/storage/docs/gsutil_install):

```bash
mkdir -p data/super_resolution/training

# Low-res
gsutil -m -q cp -R gs://genfocal/super_resolution/training/example_datasets/era5_1p5deg_1980-2009_global_t2m_w10m_q1000_msl_z200_z500.zarr data/super_resolution/training

# Low-res stats
gsutil -m -q cp -R gs://genfocal/super_resolution/training/example_datasets/era5_1p5deg_stats_global_t2m_w10m_q1000_msl_z200_z500.zarr data/super_resolution/training

# High-res normalized residuals (CONUS)
gsutil -m -q cp -R gs://genfocal/super_resolution/training/example_datasets/era5_0p25deg_residual_1980-2009_conus_t2m_w10m_q1000_msl.zarr data/super_resolution/training
```

#### Run training

To train a SR model, make sure that all paths referred to in the config file
exist and run command:

```bash
python run_training.py --config=configs/conus_train.yaml
```

The model parameters in the included config files are full replicas of the ones
used to produce the paper results. They have been verified to run on an NVIDIA
H100 GPU with 80GB vRAM.

### Sampling

#### Data preparation

Input data for sampling consists of:

- Low-resolution (1.5-degree spatial and daily mean) input data.
- Low-resolution reference statistics, including mean and standard deviation.
- High-resolution climatology to denormalize the modeled residuals; this is the
same climatology used in the second offline pre-processing step (normalization)
mentioned above.

We provide example files in the
[Google Cloud Bucket subdirectory](https://console.cloud.google.com/storage/browser/genfocal/super_resolution/inference/example_inputs),
which can be downloaded by running:

```bash
mkdir -p data/super_resolution/inference

# Low-res
gsutil -m -q cp -R gs://genfocal/super_resolution/inference/example_inputs/era5_1p5deg_2015_global_t2m_w10m_q1000_msl_z200_z500.zarr data/super_resolution/inference

# Low-res stats
gsutil -m -q cp -R gs://genfocal/super_resolution/inference/example_inputs/era5_1p5deg_stats_global_t2m_w10m_q1000_msl_z200_z500.zarr data/super_resolution/inference

# High-res residual climatology (CONUS)
gsutil -m -q cp -R gs://genfocal/super_resolution/inference/example_inputs/era5_0p25deg_residual_clim_conus_t2m_w10m_q1000_msl.zarr data/super_resolution/inference
```

where the input low-resolution data are the ground truth low-resolution ERA5 in
2015.

The last necessary piece is a checkpoint for a trained SR model:
```bash
mkdir -p data/super_resolution/checkpoints
gsutil -m -q cp -R gs://genfocal/super_resolution/checkpoints/conus_7d_t2m_w10m_q1000_msl data/super_resolution/checkpoints
```

#### Run sampling

To generate samples with a trained SR model, make sure that all paths referred
to in the config file exist and run command:

```bash
python run_sampling.py --config=configs/conus_sample.yaml
```

This example config generates a 7-day trajectory using one accelerator device
(GPU or TPU). The memory requirements for sampling is much less than that for
training - the script has been verified to run on NVIDIA A100/V100 GPU with a
minimum of 16GB RAM, as well as TPU V5e.

The sampling script supports running sampling over an extended period of time
with consistency, by making use of multiple accelerator devices to generate
overlapping date chunks in parallel. Refer to section I.4.3 in the paper and the
docstring of `sampling.TrajectorySamplerParallel` for details.

### Results

We provide access to the full end-to-end inference results
(GenFocal debiasing & super-resolution):

| Region | Date range             | Chunk  | Dataset (common prefix `gs://genfocal/results/e2e_inference/`) |
|--------|------------------------|--------|--------------------------------------------------|
| CONUS  | 2010-2019, Jun-Jul-Aug | Pixel  | `conus_jja10s_800members_pixel_chunks.zarr`  |
| CONUS  | 2010-2019, Jun-Jul-Aug | Member | `conus_jja10s_800members_member_chunks.zarr` |
| NAO    | 2019-2019, Aug-Sep-Oct | Member | `nao_aso10s_800members_member_chunks.zarr`   |

All results contain 800 members (100 LENS members, 8 SR samples per LENS member)
over test period 2010-2019. We offer results in two possible chunking schemes,
for (time, member, longitude, latitude) dimensions:

- Pixel: each chunk contains all times and members for a single (longitude,
latitude) coordinate. This chunking scheme makes it convenient to perform
analysis of pixel-wise distribution and statistics.
- Member: each chunk contains all longitude and latitude coordinates within the
region, for a single member and a limited time range. This chunking scheme is
convenient for visualizing the field in the entire spatial domain.

Note that libraries like `Xarray` often performs lazy reads to the indices
requested. It is therefore highly important to choose a chunking scheme based on
the exact need for optimized performance.

### Tutorials

For users new to diffusion-based generative models or curious to learn more,
please refer to [our introductory colabs](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion/colabs), which contains detailed explanations
and examples. Some guidance on how to make use of the versatile `swirl-dynamics`
templates are also provided.

## License

Licensed under the Apache License, Version 2.0.
