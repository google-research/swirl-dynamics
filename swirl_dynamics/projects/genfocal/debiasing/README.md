# Genfocal: Debiasing with Rectified Flows

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

## License

Licensed under the Apache License, Version 2.0.