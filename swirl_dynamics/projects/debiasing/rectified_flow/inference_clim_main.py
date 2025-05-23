# Copyright 2024 The swirl_dynamics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""The entry point for running inference loops with climatology.

Here the inference is performed snapshot by snapshot, in contrast to the
inference_clim_time_main.py where the inference is performed by chunks of
snapshots, as the models are build to perform debiasing per chunks.
"""

import functools
import json
import logging
from os import path as osp
from typing import Any

from absl import app
from absl import flags
import grain.python as pygrain
import jax
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import numpy as np
from swirl_dynamics.lib.solvers import ode as ode_solvers
from swirl_dynamics.projects.debiasing.rectified_flow import dataloaders
from swirl_dynamics.projects.debiasing.rectified_flow import inference_utils
from swirl_dynamics.projects.debiasing.rectified_flow import trainers
import tensorflow as tf
import xarray as xr


# pylint: disable=line-too-long
_ERA5_DATASET_PATH = "/lzepedanunez/data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = "/lzepedanunez/data/era5/climat/1p5deg_11vars_windspeed_1961-2000_daily_v2.zarr"

_LENS2_DATASET_PATH = (
    "/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
)
_LENS2_STATS_PATH = "/lzepedanunez/data/lens2/climat/lens2_240x121_lonlat_clim_daily_1961_to_2000_31_dw.zarr"
_LENS2_MEAN_CLIMATOLOGY_PATH = "/lzepedanunez/data/lens2/climat/mean_lens2_240x121_lonlat_clim_daily_1961_to_2000.zarr"
_LENS2_STD_CLIMATOLOGY_PATH = "/lzepedanunez/data/lens2/climat/std_lens2_240x121_lonlat_clim_daily_1961_to_2000.zarr"
# pylint: enable=line-too-long

_ERA5_VARIABLES = {
    "2m_temperature": None,
    "specific_humidity": {"level": 1000},
    "mean_sea_level_pressure": None,
    "10m_magnitude_of_wind": None,
}

_LENS2_MEMBER_INDEXER = ({"member": "cmip6_1001_001"},)
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV")
_LENS2_VARIABLES = {v: _LENS2_MEMBER_INDEXER[0] for v in _LENS2_VARIABLE_NAMES}

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Directory to store model data.")
flags.DEFINE_string("workdir", None, "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


# Loading the trainers.
def build_data_loaders(
    config: ml_collections.ConfigDict,
    batch_size: int,
    lens2_member_indexer: tuple[dict[str, str], ...] | None = None,
    lens2_variable_names: tuple[str, ...] | None = None,
    era5_variables: dict[str, dict[str, Any]] | None = None,
    date_range: tuple[str, str] | None = None,
    regime="eval",
) -> pygrain.DataLoader:
  """Loads the data loaders.

  Args:
    config: The config file used for the training experiment.
    batch_size: The batch size.
    lens2_member_indexer: The member indexer for the LENS2 dataset, here each
      member indexer is a dictionary with the key "member" and the value is the
      name of the member. In general we will usee only one member indexer at a
      time.
    lens2_variable_names: The names of the variables in the LENS2 dataset.
    era5_variables: The names of the variables in the ERA5 dataset.
    date_range: The date range for the evaluation.
    regime: The regime for the evaluation.

  Returns:
    The dataloader for the inference loop.
  """

  if not date_range:
    logging.info("Using the default date ranges.")
    if regime == "train":
      date_range = config.date_range_train
    elif regime == "eval":
      date_range = config.date_range_eval

  if lens2_member_indexer is None:
    lens2_member_indexer = (
        _LENS2_MEMBER_INDEXER
        if "lens2_member_indexer" not in config
        else config.lens2_member_indexer.to_dict()
    )

  # TODO: Use the paths from the config file.
  dataloader = (
      dataloaders.create_ensemble_lens2_era5_loader_with_climatology(
          date_range=date_range,
          batch_size=batch_size,
          shuffle=False,
          input_dataset_path=_LENS2_DATASET_PATH,
          input_climatology=_LENS2_STATS_PATH,
          input_mean_stats_path=_LENS2_MEAN_CLIMATOLOGY_PATH,
          input_std_stats_path=_LENS2_STD_CLIMATOLOGY_PATH,
          input_variable_names=lens2_variable_names,
          input_member_indexer=lens2_member_indexer,
          output_dataset_path=_ERA5_DATASET_PATH,
          output_climatology=_ERA5_STATS_PATH,
          output_variables=era5_variables,
          time_stamps=True,
          inference_mode=True,  # Using the inference dataset.
          num_epochs=1,  # This is the loops stops automatically.
      )
  )

  return dataloader


def sampling_from_batch(
    batch, model, trained_state, num_sampling_steps: int
) -> jax.Array:
  """Sampling from a batch.

  Args:
    batch: The batch to sample from.
    model: The model that encapsulates the flow model.
    trained_state: The trained state of the model.
    num_sampling_steps: The number of sampling steps for solving the ODE.

  Returns:
    The sampled output.
  """

  # Setting up the conditional ODE.
  cond = {
      "channel:mean": batch["channel:mean"],
      "channel:std": batch["channel:std"],
  }

  latent_dynamics_fn = ode_solvers.nn_module_to_dynamics(
      model.flow_model,
      autonomous=False,
      cond=cond,  # to be added here.
      is_training=False,
  )

  integrator = ode_solvers.RungeKutta4()
  integrate_fn = functools.partial(
      integrator,
      latent_dynamics_fn,
      tspan=jnp.arange(0.0, 1.0, 1.0 / num_sampling_steps),
      params=trained_state.model_variables,
  )

  # Running the integration.
  out = integrate_fn(batch["x_0"])[-1, :]

  # denormalize the output according to ERA5 Climatology.
  out = (out * batch["output_std"]) + batch["output_mean"]

  return out


def inference_pipeline(
    model_dir: str,
    date_range: tuple[str, str] | None = None,
    verbose: bool = False,
    batch_size_eval: int = 16,
    variables: tuple[str, ...] = (
        "temperature",
        "specific_humidity",
        "mean_sea_level_pressure",
        "wind_speed",
    ),
    lens2_member_indexer: tuple[dict[str, str], ...] | None = None,
    lens2_variable_names: dict[str, dict[str, str]] | None = None,
    era5_variables: dict[str, dict[str, int] | None] | None = None,
    num_sampling_steps: int = 100,
) -> dict[str, np.ndarray]:
  """The evaluation pipeline.

  Args:
    model_dir: The directory with the model to infer.
    date_range: The date range for the evaluation.
    verbose: Whether to print the config file.
    batch_size_eval: The batch size for the evaluation.
    variables: Names of physical fields.
    lens2_member_indexer: The member indexer for the LENS2 dataset.
    lens2_variable_names: The names of the variables in the LENS2 dataset.
    era5_variables: The names of the variables in the ERA5 dataset.
    num_sampling_steps: The number of sampling steps for solving the ODE.

  Returns:
    A dictionary with the data both input and output with their corresponding
    time stamps.
  """
  del verbose  # Not used for now.

  # Using the default values.
  if not lens2_member_indexer:
    logging.info("Using the default lens2_member_indexer")
    lens2_member_indexer = _LENS2_MEMBER_INDEXER
  if not era5_variables:
    logging.info("Using the default era5_variables")
    era5_variables = _ERA5_VARIABLES

  # We will leverage parallelization among current devices.
  num_devices = jax.local_device_count()

  logging.info("number of devices: %d", num_devices)

  # load the json file
  with tf.io.gfile.GFile(osp.join(model_dir, "config.json"), "r") as f:
    args = json.load(f)
    if isinstance(args, str):
      args = json.loads(args)

  logging.info("Loading config file")
  config = ml_collections.ConfigDict(args)

  if (
      config.input_shapes[0][-1] != config.input_shapes[1][-1]
      or config.input_shapes[0][-1] != config.out_channels
  ):
    raise ValueError(
        "The number of channels in the input and output must be the same."
    )

  logging.info("Building the model")
  model = inference_utils.build_model_from_config(config)

  try:
    trained_state = trainers.TrainState.restore_from_orbax_ckpt(
        f"{model_dir}/checkpoints", step=None
    )
  except FileNotFoundError:
    print(f"Could not load checkpoint from {model_dir}/checkpoints.")
    return {}
  logging.info("Model loaded")

  sampling_from_batch_partial = functools.partial(
      sampling_from_batch,
      model=model,
      trained_state=trained_state,
      num_sampling_steps=num_sampling_steps,
  )

  logging.info("Pmapping the sampling function.")
  parallel_sampling_fn = jax.pmap(
      sampling_from_batch_partial, in_axes=0, out_axes=0
  )

  logging.info("Building the data loader.")
  eval_dataloader = build_data_loaders(
      config,
      batch_size=batch_size_eval * num_devices,
      lens2_member_indexer=lens2_member_indexer,
      lens2_variable_names=lens2_variable_names,
      era5_variables=era5_variables,
      date_range=date_range,
  )

  output_list = []
  time_stamps = []

  # TODO: These are hardcoded for now. To be changed.
  n_lon = 240
  n_lat = 121
  n_field = len(variables)
  par_keys = ("channel:mean", "channel:std", "output_mean", "output_std", "x_0")

  logging.info("Batch size per device: %d", batch_size_eval)

  for ii, batch in enumerate(eval_dataloader):

    logging.info("Iteration: %d", ii)
    # input_list.append(batch["x_0"])
    time_stamps.append(batch["input_time_stamp"])

    # Running the inference loop.
    batch_par = {key: batch[key] for key in par_keys}
    parallel_batch = jax.tree.map(
        lambda x: x.reshape(
            (
                num_devices,
                -1,
            )
            + x.shape[1:]
        ),
        batch_par,
    )

    # Running the parallel sampling and saving the output.
    samples = np.array(parallel_sampling_fn(parallel_batch))
    samples = samples.reshape((-1, n_lon, n_lat, n_field))
    output_list.append(samples)

  output_array = np.concatenate(output_list, axis=0)
  time_stamps = np.concatenate(time_stamps, axis=0).reshape((-1,))

  data_dict = dict(
      time_stamps=time_stamps,
      output_array=output_array,
  )

  return data_dict


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  # Gets the regular config file here.
  config = FLAGS.config
  model_dir = config.model_dir
  date_range = config.date_range
  batch_size_eval = config.batch_size_eval
  variables = config.variables

  # Dump config as json to workdir.
  workdir = FLAGS.workdir
  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)
  # Only 0-th process should write the json file to disk, in order to avoid
  # race conditions.
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(
        name=osp.join(workdir, "config.json"), mode="w"
    ) as f:
      conf_json = config.to_json_best_effort()
      if isinstance(conf_json, str):  # Sometimes `.to_json()` returns string
        conf_json = json.loads(conf_json)
      json.dump(conf_json, f)

  tf.config.experimental.set_visible_devices([], "GPU")

  ## Use the for loop here.
  logging.info("Evaluating %s", model_dir)

  # This is necessary due the ordering issues inside a ConfigDict.
  # TODO: Change the pipeline to avoid concatenating
  # ConfigDicts.
  if "era5_variables" in config and config.era5_variables:
    logging.info("Using era5_variables from config file.")
    era5_variables = config.era5_variables.to_dict()
    print("ERA5 variables", flush=True)
    print(era5_variables, flush=True)
  else:
    logging.info("Using default era5_variables.")
    era5_variables = _ERA5_VARIABLES

  if "lens2_variable_names" in config and config.lens2_variable_names:
    logging.info("Using lens2_variable_names from config file.")
    lens2_variable_names = config.lens2_variable_names
  else:
    lens2_variable_names = _LENS2_VARIABLE_NAMES

  if "lens2_member_indexer" in config and config.lens2_member_indexer:
    lens2_member_indexer = config.lens2_member_indexer
  else:
    lens2_member_indexer = _LENS2_MEMBER_INDEXER

  if "num_sampling_steps" in config:
    logging.info("Using num_sampling_steps from config file.")
    num_sampling_steps = config.num_sampling_steps
  else:
    logging.info("Using default num_sampling_steps.")
    num_sampling_steps = 100

  logging.info("Number of sampling steps %d", num_sampling_steps)

  for lens2_indexer in lens2_member_indexer:
    print("Evaluating on CMIP dataset indexer: ", lens2_indexer)
    index_member = lens2_indexer

    # Checking that the file wasn't already saved.
    xm, num = model_dir.split("/")[-2:]
    path_zarr = (
        f"{workdir}/debiasing_lens2_to_era5_xm_{xm}_{num}_{index_member}.zarr"
    )

    if tf.io.gfile.exists(path_zarr):
      logging.info("File %s already exists, skipping", path_zarr)
    else:
      # run the inference pipeline here.
      data_dict = inference_pipeline(
          model_dir=model_dir,
          date_range=date_range,
          batch_size_eval=batch_size_eval,
          variables=variables,
          lens2_member_indexer=(
              {"member": lens2_indexer},  # It needs to be a tuple.
          ),
          lens2_variable_names=lens2_variable_names,
          era5_variables=era5_variables,
          num_sampling_steps=num_sampling_steps,
      )

      if jax.process_index() == 0:
        logging.info("Saving data in Zarr format")

        print(f"Shape of time stamps {data_dict['time_stamps'].shape}")

        ds = {}
        ds["reflow"] = xr.DataArray(
            data_dict["output_array"],
            dims=["time", "longitude", "latitude", "variables"],
            coords={"time": data_dict["time_stamps"]},
        )

        ds = xr.Dataset(ds)
        ds = ds.chunk(
            {"time": 128, "longitude": -1, "latitude": -1, "variables": -1}
        )
        ds.to_zarr(path_zarr)
        logging.info("Data saved in Zarr format in %s", path_zarr)


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  handler = app.run
  handler(main)
