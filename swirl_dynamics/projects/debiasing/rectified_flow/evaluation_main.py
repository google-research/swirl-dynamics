# Copyright 2026 The swirl_dynamics Authors.
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

r"""The main entry point for running evaluation loops."""

import functools
import json
import logging
from os import path as osp

from absl import app
from absl import flags
import jax
import ml_collections
from ml_collections import config_flags
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.projects.debiasing.rectified_flow import data_utils
from swirl_dynamics.projects.debiasing.rectified_flow import dataloaders
from swirl_dynamics.projects.debiasing.rectified_flow import evaluation_metrics as metrics
from swirl_dynamics.projects.debiasing.rectified_flow import inference_utils
from swirl_dynamics.projects.debiasing.rectified_flow import trainers
from swirl_dynamics.projects.debiasing.rectified_flow import utils
import tensorflow as tf


# pylint: disable=line-too-long
_ERA5_DATASET_PATH = "/lzepedanunez/data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = "/lzepedanunez/data/era5/stats/daily_mean_240x121_all_variables_and_wind_speed_1961-2000.zarr"
_LENS2_DATASET_PATH = (
    "/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
)
_LENS2_STATS_PATH = "/lzepedanunez/data/lens2/stats/all_variables_240x121_lonlat_1961-2000.zarr/"
_LENS2_MEAN_STATS_PATH = "/lzepedanunez/data/lens2/stats/lens2_mean_stats_all_variables_240x121_lonlat_1961-2000.zarr"
_LENS2_STD_STATS_PATH = "/lzepedanunez/data/lens2/stats/lens2_std_stats_all_variables_240x121_lonlat_1961-2000.zarr"
# pylint: enable=line-too-long

_ERA5_VARIABLES = {
    "2m_temperature": None,
    "specific_humidity": {"level": 1000},
    "mean_sea_level_pressure": None,
    "10m_magnitude_of_wind": None,
}

# This is modified to be a tuple of strings, from a tuple of dictionaries of the
# form : ({"member": "cmip6_1001_001"},)
_LENS2_MEMBER_INDEXER = ("cmip6_1001_001",)
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV")
_LENS2_VARIABLES = {
    v: {"member": _LENS2_MEMBER_INDEXER[0]} for v in _LENS2_VARIABLE_NAMES
}

FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Directory to store model data.")
flags.DEFINE_string("workdir", None, "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def read_stats(
    era5_variables: dict[str, dict[str, int] | None] | None = None,
    lens2_variables: dict[str, dict[str, str] | None] | None = None,
) -> dict[str, np.ndarray]:
  """Read the stats from the Zarr files.

  Args:
    era5_variables: A dictionary containing the ERA5 variables.
    lens2_variables: A dictionary containing the LENS2 variables.

  Returns:
    A dictionary with the variables.
  """

  if not era5_variables:
    era5_variables = _ERA5_VARIABLES
  if not lens2_variables:
    lens2_variables = _LENS2_VARIABLES

  mean_era5_dict = data_utils.read_stats(
      _ERA5_STATS_PATH, era5_variables, "mean"
  )
  std_era5_dict = data_utils.read_stats(_ERA5_STATS_PATH, era5_variables, "std")

  mean_lens2_dict = data_utils.read_stats(
      _LENS2_STATS_PATH, lens2_variables, "mean"
  )
  std_lens2_dict = data_utils.read_stats(
      _LENS2_STATS_PATH, lens2_variables, "std"
  )

  mean_array = []
  for key, variable in mean_era5_dict.items():
    mean_array.append(variable)
    print(key)

  std_array = []
  for _, variable in std_era5_dict.items():
    std_array.append(variable)

  mean_era5 = np.concatenate(mean_array, axis=-1)
  std_era5 = np.concatenate(std_array, axis=-1)

  mean_array_lens2 = []
  for key, variable in mean_lens2_dict.items():
    mean_array_lens2.append(variable)
    print(key)

  std_array_lens2 = []
  for _, variable in std_lens2_dict.items():
    std_array_lens2.append(variable)

  mean_lens2 = np.concatenate(mean_array_lens2, axis=-1)
  std_lens2 = np.concatenate(std_array_lens2, axis=-1)

  return {
      "mean_era5": mean_era5,
      "std_era5": std_era5,
      "mean_lens2": mean_lens2,
      "std_lens2": std_lens2,
  }


def read_normalized_stats(
    lens2_variables: dict[str, dict[str, str] | None] | None = None,
) -> dict[str, np.ndarray]:
  """Reads the normalization statistics for conditining.

  Args:
    lens2_variables:

  Returns:
   The ensemble mean and std of the trajectory mean and std.
  """
  if not lens2_variables:
    lens2_variables = _LENS2_VARIABLES

  print("Reading the normalized statistics of LENS2 across ensemble members.")
  print(f"LENS2 variables {lens2_variables}")

  # We extract the keys, which are the fields of the dataset.
  keys_dict = {len: {} for len in lens2_variables.keys()}
  mean_mean = data_utils.read_stats(_LENS2_MEAN_STATS_PATH, keys_dict, "mean")
  mean_std = data_utils.read_stats(_LENS2_MEAN_STATS_PATH, keys_dict, "std")

  std_mean = data_utils.read_stats(_LENS2_STD_STATS_PATH, keys_dict, "mean")
  std_std = data_utils.read_stats(_LENS2_STD_STATS_PATH, keys_dict, "std")

  mean_mean_array = []
  for _, variable in mean_mean.items():
    mean_mean_array.append(variable)

  mean_std_array = []
  for _, variable in mean_std.items():
    mean_std_array.append(variable)

  std_mean_array = []
  for _, variable in std_mean.items():
    std_mean_array.append(variable)

  std_std_array = []
  for _, variable in std_std.items():
    std_std_array.append(variable)

  return {
      "mean_mean": np.concatenate(mean_mean_array, axis=-1),
      "mean_std": np.concatenate(mean_std_array, axis=-1),
      "std_mean": np.concatenate(std_mean_array, axis=-1),
      "std_std": np.concatenate(std_std_array, axis=-1),
  }


def evaluation_pipeline(
    model_dir: str,
    config_eval: ml_collections.ConfigDict,
    workdir_path: str,
    date_range: tuple[str, str] | None = None,
    verbose: bool = False,
    batch_size_eval: int = 16,
    variables: tuple[str, ...] = (
        "wind_speed",
        "temperature",
        "geopotential_200",
        "geopotential_200",
        "mean_sea_level_pressure",
        "specific_humidity",
        "u_component_of_wind_200",
        "u_component_of_wind_850",
        "v_component_of_wind_200",
        "v_component_of_wind_850",
    ),
    lens2_member_indexer: tuple[dict[str, str], ...] | None = None,
    lens2_variable_names: dict[str, dict[str, str]] | None = None,
    era5_variables: dict[str, dict[str, int] | None] | None = None,
    num_sampling_steps: int = 100,
) -> dict[str, dict[str, dict[str, jax.Array]]]:
  """The evaluation pipeline.

  Args:
    model_dir: The directory to load the model checkpoint from.
    config_eval: The configuration dictionary (ConfigDict) containing the
      parameters for the evaluation.
    workdir_path: The path to the work directory for saving the evaluation
      snapshots, and the metrics.
    date_range: The date range for the evaluation.
    verbose: Whether to print the config file.
    batch_size_eval: The batch size for the evaluation.
    variables: Names of physical fields that are saved in the files.
    lens2_member_indexer: The member indexer for the LENS2 dataset. This is a
      tuple of dictionaries of the form: ({"member": "cmip6_1001_001"},).
    lens2_variable_names: The names of the variables in the LENS2 dataset. This
      is a tuple of strings containing the names of the fields to be extracted
      from the LENS2 dataset. E.g., ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV").
    era5_variables: The names of the variables in the ERA5 dataset. This is a
      dictionary that follows the format of the ERA5 dataset, using Xarray,
      where the keys represent the field and the items represent the levels at
      which the variables are extracted (if None, the variables are surface
      variables). E.g., {"2m_temperature": None, "specific_humidity": {"level":
      1000}, "mean_sea_level_pressure": None, "10m_magnitude_of_wind": None}.
    num_sampling_steps: The number of sampling steps for solving the ODE.

  Returns:
    A dictionary with the data both input and output with their corresponding
    time stamps.
  """
  del verbose  # Not used for now.

  # Using the default values. TODO: Refactor this.
  if lens2_member_indexer is None:
    logging.info("Using the default lens2_member_indexer")
    lens2_member_indexer = _LENS2_MEMBER_INDEXER
  if era5_variables is None:
    logging.info("Using the default era5_variables")
    era5_variables = _ERA5_VARIABLES

  # This is within the lens2_member_indexer tuple of dictionaries.
  # TODO: Refactor this.
  lens2_indexer = lens2_member_indexer[0]["member"]

  # We leverage parallelization among current devices.
  num_devices = jax.local_device_count()

  logging.info("number of devices: %d", num_devices)

  # Loads the json file with the network configuration.
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
      inference_utils.sampling_from_batch,
      model=model,
      trained_state=trained_state,
      num_sampling_steps=num_sampling_steps,
      time_chunk_size=config.time_batch_size,  # This is the time chunk size.
      time_to_channel=config.get("time_to_channel", default=True),
      reverse_flow=False,
  )

  logging.info("Pmapping the sampling function.")
  parallel_sampling_fn = jax.pmap(
      sampling_from_batch_partial, in_axes=0, out_axes=0
  )

  logging.info("Building the data loader.")
  eval_dataloader = dataloaders.build_inference_dataloader(
      config=config,
      config_eval=config_eval,
      batch_size=batch_size_eval * num_devices,
      lens2_member_indexer=lens2_member_indexer,
      lens2_variable_names=lens2_variable_names,
      era5_variables=era5_variables,
      date_range=date_range,
      regime="eval",
  )

  input_list = []
  target_list = []
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

    # Saves the time stamps, regardless of whether the checkpoint exists or not.
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

    input_samples = batch["x_0"] * batch["input_std"] + batch["input_mean"]
    target_samples = batch["x_1"] * batch["output_std"] + batch["output_mean"]
    output_list.append(samples)
    target_list.append(target_samples)
    input_list.append(input_samples)
    target_list.append(batch["x_1"])

  output_array = np.concatenate(output_list, axis=0)
  input_array = np.concatenate(input_list, axis=0)
  target_array = np.concatenate(target_list, axis=0)
  time_stamps_array = np.concatenate(time_stamps, axis=0).reshape((-1,))

  # Computing metrics.
  transport_error = np.mean(
      np.sqrt(np.sum(np.square(output_array - target_array), axis=(1, 2))),
      axis=0,
  ) / np.mean(np.sqrt(np.sum(np.square(target_array), axis=(1, 2))), axis=0)

  wass_err_dict = metrics.wass1_error(
      input_array, output_array, target_array, variables=variables
  )
  log_energy_dict = metrics.log_energy_error(
      input_array, output_array, target_array, variables=variables
  )

  mean_err_dict = metrics.smoothed_average_l1_error(
      input_array,
      output_array,
      target_array,
      variables=variables,
  )

  err_dict = dict(
      transport_error=transport_error,
      wass_err=wass_err_dict,
      energy_err=log_energy_dict,
      mean_err=mean_err_dict,
  )

  # Saving evaluation snapshots in Zarr format.
  utils.save_data_in_zarr(
      time_stamps_array,
      input_array,
      output_array,
      model_dir,
      workdir_path,
      lens2_indexer,
      target_array,
      time_chunk=128,
  )

  return err_dict


def parse_experiment_dirs(exp_dir):
  """Parse directory to load arguments json."""
  dirs_list = []
  if tf.io.gfile.exists(exp_dir):
    dirs = tf.io.gfile.listdir(exp_dir)
  else:
    raise FileNotFoundError(f"Could not list directory: {exp_dir}.")
  for d in dirs:
    try:
      if isinstance(int(d), int):
        dirs_list.append(osp.join(exp_dir, d))
    except ValueError:
      continue
  return dirs_list


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  # Gets the regular config file here.
  config_eval = FLAGS.config
  model_dir = config_eval.model_dir
  date_range = config_eval.date_range
  batch_size_eval = config_eval.batch_size_eval
  variables = config_eval.variables

  # Dump config_eval as json to workdir.
  workdir = FLAGS.workdir
  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)
  # Only 0-th process should write the json file to disk, in order to avoid
  # race conditions.
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(
        name=osp.join(workdir, "config.json"), mode="w"
    ) as f:
      conf_json = config_eval.to_json_best_effort()
      if isinstance(conf_json, str):  # Sometimes `.to_json()` returns string
        conf_json = json.loads(conf_json)
      json.dump(conf_json, f)

  tf.config.experimental.set_visible_devices([], "GPU")

  ## Use the for loop here.
  logging.info("Parsing the experiment directories in %s", model_dir)
  model_dir_array = parse_experiment_dirs(model_dir)
  logging.info("model_dir_array")
  print(model_dir_array)

  err_dict_global = {}
  # We add the xm to keep track of the experiment.
  xm = model_dir.split("/")[-1]
  logging.info("Evaluating xm experiment %s", xm)
  err_dict_global[xm] = {}

  # Loading the variables from the config file.
  variable_dict = utils.parse_vars_from_config_to_dict(config_eval)
  era5_variables = variable_dict["era5_variables"]
  lens2_variable_names = variable_dict["lens2_variable_names"]
  lens2_member_indexer_tuple = variable_dict["lens2_member_indexer_tuple"]

  num_sampling_steps = config_eval.get("num_sampling_steps", default=100)
  logging.info("Number of sampling steps %d", num_sampling_steps)

  print("Indexers")
  print(lens2_member_indexer_tuple, flush=True)

  if tf.io.gfile.exists(workdir + "/err_dict.hdf5"):
    logging.info(
        "File %s already exists, reading from file", workdir + "/err_dict.hdf5"
    )
    err_dict_global = hdf5_utils.read_all_arrays_as_dict(
        workdir + "/err_dict.hdf5", err_dict_global
    )

  for model_dir_idx in model_dir_array:
    logging.info("Evaluating %s", model_dir_idx)
    xm, idx_exp = model_dir_idx.split("/")[-2:]
    # if a given idx_exp is not in the dictionary, we create it as empty.
    if idx_exp not in err_dict_global[xm]:
      err_dict_global[xm][idx_exp] = {}

    # Recall that lens2_member_indexer is a tuple of dictionaries.
    for index_member in lens2_member_indexer_tuple:

      path_zarr = f"{workdir}/debiasing_lens2_to_era5_xm_{xm}_{idx_exp}_{index_member}.zarr"
      if tf.io.gfile.exists(path_zarr):
        logging.info("File %s already exists, skipping", path_zarr)
      else:
        print("Evaluating on CMIP dataset indexer: ", index_member)
        err_dict_global[xm][idx_exp][index_member] = evaluation_pipeline(
            model_dir=model_dir_idx,
            config_eval=config_eval,
            workdir_path=workdir,
            date_range=date_range,
            batch_size_eval=batch_size_eval,
            variables=variables,
            lens2_member_indexer=(
                {"member": index_member},  # It needs to be a tuple.
            ),
            lens2_variable_names=lens2_variable_names,
            era5_variables=era5_variables,
            num_sampling_steps=num_sampling_steps,
        )

    # Save all the error into a file.
    logging.info("Saving the error dictionary")
    if jax.process_index() == 0:
      hdf5_utils.save_array_dict(workdir + "/err_dict.hdf5", err_dict_global)


if __name__ == "__main__":
  app.run(main)
