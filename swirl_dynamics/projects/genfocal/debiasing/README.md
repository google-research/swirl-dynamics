# GenFocal: Debiasing with Rectified Flows

![Diagram of the debiasing step of GenFocal](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/genfocal/debiasing/figures/genfocal_debiasing.png)

This folder contains the code necessary for performing the debiasing step in [GenFocal](https://arxiv.org/abs/2412.08079).

The code depends heavily on the `swirl-dynamics` template for all its
scaffolding. The dataloaders make extensive use of `xarray` for lazy reads from
disk and of the modularity of `pygrain`. This allows for writing a couple of
'DataSources' and then using `pygrain` transformations to transform the data
(such as normalization, batching, etc.).

The three main scripts are:

  - `run_train.py`: This is the main training script.
  - `run_validation.py`: This is a script for running a lightweight evaluation
  of the model using the latest checkpoint (specified in a config file). The
  script will debias the specified period (by default 2000-2009) for a small
  number of LENS2 members and run a few statistical metrics. The results are
  saved in a dictionary and stored in an HDF5 file.
  - `run_inference.py`: This script runs the debiasing during the target period (by default 2010-2019) for the given LENS2 members and stores the results in a [Zarr](https://zarr.dev/) file.

We also provide configuration files for each step. The models in these files are significantly smaller than the ones used in [GenFocal](https://arxiv.org/abs/2412.08079). The original config files can be found in the checkpoints as a `config.json` file.

The functionality of the other files is summarized below:

  - `dataloaders.py`: This file contains all the necessary scaffolding for
  loading the data using `xarray` and `zarr` as a backbone.
  - `models.py`: This file wraps the neural network using the `swirl-dynamics`
  framework. It interfaces the model with the loss function and evaluation
  protocols.
  - `inference_utils.py`: This file contains several utility functions for
  building the different dataloaders from the configuration files and for
  performing sampling using an ODE solver.
  - `pygrain_transforms.py`: This file contains all the transformations
  necessary to build the dataloaders. They allow for extra flexibility,
  particularly when normalizing the data by the climatology.
  - `trainers.py`: This file contains the trainers, which encapsulate most of
  the training logic as specified in the `swirl_dynamics` templates.
  - `utils.py`: This file contains several utility classes that are helpful as
  configuration classes.

## Workflow

The typical workflow consists of two main parts. The first part involves
training various debiasing models with different hyperparameters. The validation
script is then used to check statistical quantities of the debiased samples in
the validation set (the 2010s period) to select one or more models. Finally, the
inference script is used to run inference on the validation set.

The second step requires the super-resolution model. The data generated by the
selected models is then super-resolved and run through an exhaustive battery of
benchmarks. These include spatio-temporal statistics, extreme events, tropical
cyclone detections, etc. A final model is chosen from the pre-screened
candidates based on these benchmark results.

In what follows, we provide instructions on how to run the first part.

### Downloading the Data

First, we download the data necessary for training the models. One can read the
data on-demand from the Google Cloud [bucket](https://console.cloud.google.com/storage/browser/genfocal);
however, for simplicity, we will download the data directly.

Since the data is located in a bucket, we use `gsutil` to download it.

We will download all the data from LENS2, which has already been regridded.
Here, we download the climatologies and the data.

```bash
mkdir -p data/lens2/
gsutil -m -q cp -R gs://genfocal/data/lens2/lens2_240x121_lonlat_1960-2020_10_vars_4_train_members.zarr data/lens2/
gsutil -m -q cp -R gs://genfocal/debiasing/climatology/lens2_240x121_10_vars_4_members_lonlat_clim_daily_1961_to_2000_31_dw.zarr data/lens2/
gsutil -m -q cp -R gs://genfocal/debiasing/climatology/mean_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr data/lens2/
gsutil -m -q cp -R gs://genfocal/debiasing/climatology/std_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr data/lens2/
```

The first dataset contains the data from LENS2, from which we have extracted
only 4 members. The second contains the climatological mean and standard
deviation for these four members, and the last two contain the ensemble mean and
 standard deviation of the climatology.

Now, we will do the same for the ERA5 data, which has been regridded and
daily-averaged.

```bash
mkdir -p data/era5/
gsutil -m -q cp -R gs://genfocal/data/era5/era5_240x121_lonlat_1980-2020_10_vars.zarr data/era5/
gsutil -m -q cp -R gs://genfocal/debiasing/climatology/1p5deg_11vars_windspeed_1961-2000_daily_v2.zarr data/era5/
```

The total storage footprint should be around `80 GB`.

### Training a Model

Training a model requires a configuration file. This file contains the paths to
the data, the neural network configuration, details for the dataloaders, and
several hyperparameters for optimization. We provide an example in
`configs/config_train_lens2_to_era5.py`. Note that this is a smaller model than
the one presented in the paper and is primarily intended to demonstrate the
training process. The config file for the model used in the paper can be found
in the checkpoint folder within the Google bucket (see the Colab demo in
`/colabs/genfocal_debiasing_demo.ipynb`).

In addition, you need to specify a working directory where the checkpoints and
training statistics (readable by TensorBoard) will be stored.

You can then train the model (here, we use the provided config file) by running:

```bash
current_dir=$(pwd)
mkdir -p experiments/001/
absolute_path="$current_dir/experiments/001"

python run_train.py --workdir=${absolute_path} --config=configs/config_train_lens2_to_era5.py
```

This will create a working directory and train the model. You may need to
provide the absolute path in `workdir` options as the checkpointer requires
an absolute path. This was tested on a host with four `A100s` with `40GB` each
and a host with four `H100s` with `80GB` each. The training took around 3 and
2 days respectively. The model in the [paper](https://arxiv.org/abs/2412.08079),
whose checkpoints we provide in the
[bucket](https://console.cloud.google.com/storage/browser/genfocal),
was trained for 3 days in 16 `TPU v5p`.

### Validating a Model

To validate the model, you need a training checkpoint and a config file. We also
provide a config file to run the validation. The configuration file primarily
dictates the validation period, the LENS2 ensemble members to use, and the
model to evaluate. By default, it takes the root folder of several experiments
and runs the validation in bulk. It then stores the results, which can be
opened afterward, in an `hdf5` file within a working folder.

As such, you can validate the models in a root directory (such as the
`experiments` folder used above) by running:

```bash
mkdir -p validation_folder/
python run_validation.py --workdir=validation_folder --config=configs/config_validation_lens2_to_era5.py
```

### Running Inference

This step also requires a configuration file, like the one provided in
`configs/config_inference_lens2_to_era5.py`. Here, you need to provide the
period, the LENS2 members, and the model folder to run inference on.

This step assumes that you have a multi-accelerator instance, and it will
instantly distribute the load across them. Since this step can take several
hours, we save the results of each batch to an `hdf5` file in case your instance
gets preempted or crashes. In general, `zarr` files can get corrupted, so we use
this approach as a backup. Once the script finishes, it will read all the
temporary files and save the final results in a `zarr` file.

You can run inference from a root directory (such as the `experiments` folder
used above) by running:

```bash
mkdir -p inference_folder/
python run_inference.py --workdir=inference_folder --config=configs/config_inference_lens2_to_era5.py
```

If you run several of these scripts in parallel to debias different LENS2
members, you will need to aggregate the results manually. This is because the
super-resolution step requires a single `zarr` file as input.

### Results

If you want to look at the already debiased dataset, you can download it using:

```bash
mkdir -p data/lens_debiased/
gsutil -m -q cp -R gs://genfocal/data/lens_debiased/xm_151818801_1_100_members_1960_2100_gen_154117335.zarr data/lens_debiased/
```

This file contains the 100 LENS2 members, debiased from 1960 to 2099. This is a
relatively large file, so it may take a while to download.

### Tutorial

We use rectified flow for the debiasing (for bias-correction) step. For a
tutorial on this methodology with more low-level details on how to use the
`swirl-dynamics` templates using a (much!) simpler dataset, please see this
[colab](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/debiasing/rectified_flow/colab/demo_reflow.ipynb).

## License

Licensed under the Apache License, Version 2.0.