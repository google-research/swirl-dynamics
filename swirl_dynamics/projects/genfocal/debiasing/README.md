# Genfocal: Debiasing with Rectified Flows

![Diagram of the debiasing step of GenFocal](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/genfocal/debiasing/figures/genfocal_debiasing.png)

This folder contains the code necessary for performing the debiasing step in
[GenFocal](https://arxiv.org/abs/2412.08079).

The code depends heavilty on `swirl-dynamics` template for all the scaffolding.
The dataloaders use extensive use of xarray for lazy reads from disc and of the
modularity of `pygrain` which allows to write a couple of 'DataSources' and then
use `pygrain` transformations to transform the data (such as normalization,
batching, etc.)

The main three scripts are
 - run_train.py: This is the main training script.
 - run_validation.py: This is script to running a light-weight evaluation of
    the model using the latest checkpoint of the model (specified in a config
    file). The script will debias the period (by default 2000-2009) for a small
    number of LENS2 members and it will run a few statistical metrics, which are
    saved in a dictionary and stored in a hdf5 file.
 - run_inference.py: This script runs the debiasing during the target period
    (by default 2010-2019), for the given LENS2 members, and it stores the
    result in a [Zarr](https://zarr.dev/) file.

We also provide configuration files for each step. The models in the file is
significantly smaller than the ones used in [GenFocal](https://arxiv.org/abs/2412.08079).
The original config files can be found in the checkpoints as a config.json file.

The functionality of the rest of the files is summarized below:
 - `dataloaders.py`: This file contains all the necessary scaffolding for
    loading the data using xarray and zarr as a backbone.
 - `models.py`: This files wraps the neural network using the 'swirl-dynamics'
    framework. It interfaces the model with the loss function and evaluation
    protocols.
 - `inferece_utils.py`: it contains several utility function for building the
    different dataloaders from the configuration files, and how to perform the
    sampling using an ODE solver.
 - `pygrain_transforms.py`: This file contains all the transformations necessary
    to build the dataloaders. They allow for extra flexibility, particularly
    when normalizing the data by the climatology.
 - `trainers.py`: It contains the trainers, which encapsulate most of the
    training logic as specified in the `swirl_dynamics` templates.
 - `utils.py`: It contains several utility classes that are helpfull as
    configuration classes.

## Workflow

The typical workflow consists of two main parts. The first part usually involves
training different debiasing models with different hyperparameters, then use the
validation script to check statistical quantities of the debiased samples in the
validation set (period 2010s) in order to choose one (or several) model.
Finally, use the inference script to run inference also in the validation step.

The second steps requires the super-resolution model. The data generated using
a selection of model is super-resolved, and run through an exhaustive battery of
benchmarks. Including, spatio-temporal statistics, extreme events, tropical
cyclones detections, etc. Then the model is chosen among the pre-screened ones
based on the results on these benchmarks.

In what follows we provide instructions on how to run the first part.

### Downloading the Data

First we download the data necessary for training the models. One can read the
data on demand from the google cloud [bucket](https://console.cloud.google.com/storage/browser/genfocal), although, for simplicity we download the data directly.

As the data is locate in a bucket we use `gsutil` to download it.

We download all the data already regrided from LENS2. Here we download the
climatologies and the data.
```
mkdir data/lens2/
gsutil -m -q cp -R gs://genfocal/bc/climatology/lens2_240x121_lonlat_1960-2020_10_vars_4_train_members.zarr data/lens2/
gsutil -m -q cp -R gs://genfocal/bc/climatology/lens2_240x121_10_vars_4_members_lonlat_clim_daily_1961_to_2000_31_dw.zarr data/lens2/
gsutil -m -q cp -R gs://genfocal/bc/climatology/mean_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr data/lens2/
gsutil -m -q cp -R gs://genfocal/bc/climatology/std_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr data/lens2/
```

Here the first is the data from LENS2 form which we only extracted 4 members,
the second is the climatological mean and standard deviation of these four
members, and the last two are the ensemble mean and standard deviation of the
climatology.

Now we perform the same with the ERA5 data, has been regrided and
daily-averaged.
```
mkdir data/era5/
gsutil -m -q cp -R gs://genfocal/staging/era5/era5_240x121_lonlat_1980-2020_10_vars.zarr data/era5/
gsutil -m -q cp -R gs://genfocal/bc/climatology/1p5deg_11vars_windspeed_1961-2000_daily_v2.zarr data/era5/
```

### Training a model

For training a model we need a configuration file. This configuration files
contains the paths to the data, the configuration of the neural network,
details of the dataloaders, and several hyperparameters for the optimization.
We provide an example in the `configs/config_train_lens2_to_era5.py'. We point
out that this is a smaller model than the one presented in the paper and it is
mostly meant to show how to train the model. The config file for the model in
the paper can be found in the checkpoint folder in the Google bucket (see the
colab demo in `/colabs/genfocal_debiasing_demo.ipynb`)

In addition you need to specify a working directory where the checkpoints and
the training statistics (which are readable using tensorboard) will be stored

Then you can train the model (here we use the provided config file) by running:

```
mkdir experiments/001/
python run_train.py --config=configs/config_train_lens2_to_era5.py --workdir=experiments/001
```

This will create a work directory and will train the model.

### Validating a model

For validating the model you need a training checkpoint and a config file.
We also provide a config file to run the validation. The configuration file
mostly dictates the period for the validation, the LENS2 ensemble members to
use, and the model to evaluate. By default it takes the root folder of several
experiments and run the validation in bulk. Then it stores the results in a
`hdf5` file which can be open afterwards, which are stored in a working folder.

As such you can validate the models in a root (such as the experiments folder
used above) by running:
```
mkdir validation_folder/
python run_validation.py --config=configs/config_validation_lens2_to_era5.py --workdir=validation_folder/
```

## License

Licensed under the Apache License, Version 2.0.