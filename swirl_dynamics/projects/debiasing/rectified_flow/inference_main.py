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

r"""The main entry point for running inference loops."""

import functools
import json
import logging
from os import path as osp

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import numpy as np
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.lib.solvers import ode as ode_solvers
from swirl_dynamics.projects.debiasing.rectified_flow import data_utils
from swirl_dynamics.projects.debiasing.rectified_flow import models
from swirl_dynamics.projects.debiasing.rectified_flow import trainers
import tensorflow as tf
import xarray as xr

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

_ERA5_WIND_COMPONENTS = {}

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


## Loading the trainers:
def build_data_loaders(
    config: ml_collections.ConfigDict,
    batch_size: int,
    lens2_member_indexer: tuple[dict[str, str], ...] | None = None,
    date_range=None,
    regime="eval",
):
  """Loads the data loaders."""

  if not date_range:
    logging.info("Using the default date ranges.")
    if regime == "train":
      date_range = config.date_range_train
    elif regime == "eval":
      date_range = config.date_range_eval

  lens2_variable_names = (
      _LENS2_VARIABLE_NAMES
      if "lens2_variable_names" not in config
      else config.lens2_variable_names
  )

  if not lens2_member_indexer:
    lens2_member_indexer = (
        _LENS2_MEMBER_INDEXER
        if "lens2_member_indexer" not in config
        else config.lens2_member_indexer.to_dict()
    )

  loader = data_utils.create_lens2_loader_chunked_with_normalized_stats(
      date_range=date_range,
      batch_size=batch_size,
      shuffle=False,
      random_local_shuffle=False,
      batch_ot_shuffle=False,
      dataset_path=_LENS2_DATASET_PATH,
      stats_path=_LENS2_STATS_PATH,
      mean_stats_path=_LENS2_MEAN_STATS_PATH,
      std_stats_path=_LENS2_STD_STATS_PATH,
      variable_names=lens2_variable_names,
      member_indexer=lens2_member_indexer,
      time_stamps=True,
      num_epochs=1,  # This is the loops stops automatically.
  )

  dataloader = data_utils.AlignedChunkedLens2Era5Dataset(loader=loader)
  return dataloader


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
    print(f"mean_era5 variable: {key}", flush=True)
    print(f"Variable shape: {variable.shape}", flush=True)

  std_array = []
  for key, variable in std_era5_dict.items():
    std_array.append(variable)
    print(f"std_era5 variable: {key}", flush=True)
    print(f"Variable shape: {variable.shape}")

  mean_era5 = np.concatenate(mean_array, axis=-1)
  std_era5 = np.concatenate(std_array, axis=-1)

  mean_array_lens2 = []
  for key, variable in mean_lens2_dict.items():
    mean_array_lens2.append(variable)
    print(f"mean_lens2 variable: {key}")
    print(f"Variable shape: {variable.shape}")

  std_array_lens2 = []
  for key, variable in std_lens2_dict.items():
    std_array_lens2.append(variable)
    print(f"std_lens2 variable: {key}")
    print(f"Variable shape: {variable.shape}")

  mean_lens2 = np.concatenate(mean_array_lens2, axis=-1)
  std_lens2 = np.concatenate(std_array_lens2, axis=-1)

  if mean_lens2.shape != mean_era5.shape:
    raise ValueError(
        "The shape of the mean_lens2 and mean_era5 must be the same; ",
        f"instead got {mean_lens2.shape} and {mean_era5.shape}",
    )
  if std_lens2.shape != std_era5.shape:
    raise ValueError(
        "The shape of the std_lens2 and std_era5 must be the same; ",
        f"instead got {std_lens2.shape} and {std_era5.shape}",
    )

  return {
      "mean_era5": mean_era5,
      "std_era5": std_era5,
      "mean_lens2": mean_lens2,
      "std_lens2": std_lens2,
  }


# TODO: This is common to the evaluation pipeline so add it in a
# common file.
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

  print("reading the normalized statistics of LENS2 across ensemble members.")

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


def build_model(config):
  """Builds the model from config file."""

  # Adding the conditional embedding for the FILM layer.
  if "conditional_embedding" in config and config.conditional_embedding:
    logging.info("Using conditional embedding")
    cond_embed_fn = unets.EmbConvMerge
  else:
    cond_embed_fn = None

  flow_model = models.RescaledUnet(
      out_channels=config.out_channels,
      num_channels=config.num_channels,
      downsample_ratio=config.downsample_ratio,
      num_blocks=config.num_blocks,
      noise_embed_dim=config.noise_embed_dim,
      padding=config.padding,
      dropout_rate=config.dropout_rate,
      use_attention=config.use_attention,
      resize_to_shape=config.resize_to_shape,
      use_position_encoding=config.use_position_encoding,
      num_heads=config.num_heads,
      cond_embed_fn=cond_embed_fn,
      normalize_qk=config.normalize_qk,
  )

  model = models.ConditionalReFlowModel(
      # TODO: clean this part.
      input_shape=(
          config.input_shapes[0][1],
          config.input_shapes[0][2],
          config.input_shapes[0][3],
      ),  # This must agree with the expected sample shape.
      cond_shape={
          "channel:mean": (
              config.input_shapes[0][1],
              config.input_shapes[0][2],
              config.input_shapes[0][3],
          ),
          "channel:std": (
              config.input_shapes[0][1],
              config.input_shapes[0][2],
              config.input_shapes[0][3],
          ),
      },
      flow_model=flow_model,
  )

  return model


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
  if not lens2_variable_names:
    logging.info("Using the default lens2_variable_names")
    lens2_variable_names = _LENS2_VARIABLE_NAMES
  if not era5_variables:
    logging.info("Using the default era5_variables")
    era5_variables = _ERA5_VARIABLES

  # Create the lens2 variables.
  lens2_variables = {v: lens2_member_indexer[0] for v in lens2_variable_names}

  # We will leverage parallelization among current devices.
  num_devices = jax.local_device_count()

  logging.info("number of devices: %d", num_devices)

  # load the json file
  with tf.io.gfile.GFile(osp.join(model_dir, "config.json"), "r") as f:
    args = json.load(f)
    if isinstance(args, str):
      args = json.loads(args)

  config = ml_collections.ConfigDict(args)

  assert (
      config.input_shapes[0][-1] == config.input_shapes[1][-1]
      and config.input_shapes[0][-1] == config.out_channels
  )

  model = build_model(config)

  try:
    trained_state = trainers.TrainState.restore_from_orbax_ckpt(
        f"{model_dir}/checkpoints", step=None
    )
  except FileNotFoundError:
    print(f"Could not load checkpoint from {model_dir}/checkpoints.")
    return {}
  logging.info("Model loaded")

  dict_stats = read_stats(
      era5_variables=era5_variables, lens2_variables=lens2_variables
  )
  norm_stats = read_normalized_stats(lens2_variables=lens2_variables)

  print(f"Shape of mean_lens2: {dict_stats['mean_lens2'].shape}")
  print(f"Shape of std_lens2: {dict_stats['std_lens2'].shape}")
  print(f"Shape of mean_mean: {norm_stats['mean_mean'].shape}")
  print(f"Shape of mean_std: {norm_stats['mean_std'].shape}")
  print(f"Shape of std_mean: {norm_stats['std_mean'].shape}")
  print(f"Shape of std_std: {norm_stats['std_std'].shape}")

  norm_mean_lens2 = (
      dict_stats["mean_lens2"] - norm_stats["mean_mean"]
  ) / norm_stats["std_mean"]
  norm_std_lens2 = (
      dict_stats["std_lens2"] - norm_stats["mean_std"]
  ) / norm_stats["std_std"]

  cond = {
      "channel:mean": norm_mean_lens2[None, ...],
      "channel:std": norm_std_lens2[None, ...],
  }

  latent_dynamics_fn = ode_solvers.nn_module_to_dynamics(
      model.flow_model, autonomous=False, is_training=False, cond=cond
  )

  integrator = ode_solvers.RungeKutta4()
  integrate_fn = functools.partial(
      integrator,
      latent_dynamics_fn,
      tspan=jnp.arange(0.0, 1.0, 1.0 / num_sampling_steps),
      params=trained_state.model_variables,
  )
  pmap_integrate_fn = jax.pmap(integrate_fn, in_axes=0, out_axes=0)
  logging.info("Flow model mapped.")

  logging.info("Building the data loader.")
  eval_dataloader = build_data_loaders(
      config,
      batch_size=batch_size_eval * num_devices,
      lens2_member_indexer=lens2_member_indexer,
      date_range=date_range,
  )

  output_list = []
  input_list = []
  time_stamps = []

  # TODO: These are hardcoded for now. To be changed.
  n_lon = 240
  n_lat = 121
  n_field = len(variables)

  logging.info("Batch size per device: %d", batch_size_eval)

  for ii, batch in enumerate(eval_dataloader):

    logging.info("Iteration: %d", ii)
    input_list.append(batch["x_0"])
    time_stamps.append(batch["input_time_stamp"])

    # Running the inference loop.
    input_reshape = jax.tree.map(
        lambda x: x.reshape(
            (
                num_devices,
                -1,
            )
            + x.shape[1:]
        ),
        batch["x_0"],
    )
    output_list.append(
        np.array(
            pmap_integrate_fn(input_reshape)[:, -1, :].reshape(
                (-1, n_lon, n_lat, n_field)
            )
        )
    )

  input_array = np.concatenate(input_list, axis=0)
  output_array = np.concatenate(output_list, axis=0)
  time_stamps = np.concatenate(time_stamps, axis=0).reshape((-1,))

  # Statistics for LENS2 and ERA5.
  mean_era5 = dict_stats["mean_era5"]
  std_era5 = dict_stats["std_era5"]
  mean_lens2 = dict_stats["mean_lens2"]
  std_lens2 = dict_stats["std_lens2"]

  output_array = output_array * std_era5[None, ...] + mean_era5[None, ...]
  input_array = input_array * std_lens2[None, ...] + mean_lens2[None, ...]

  data_dict = dict(
      time_stamps=time_stamps,
      output_array=output_array,
      input_array=input_array,
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
    era5_variables = config.era5_variables.to_dict()
  else:
    era5_variables = _ERA5_VARIABLES

  if "lens2_variable_names" in config and config.lens2_variable_names:
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
        ds["lens2"] = xr.DataArray(
            data_dict["input_array"],
            dims=["time", "longitud", "latitude", "variables"],
            coords={"time": data_dict["time_stamps"]},
        )
        ds["reflow"] = xr.DataArray(
            data_dict["output_array"],
            dims=["time", "longitud", "latitude", "variables"],
            coords={"time": data_dict["time_stamps"]},
        )

        ds = xr.Dataset(ds)
        ds = ds.chunk(
            {"time": 128, "longitud": -1, "latitude": -1, "variables": -1}
        )
        ds.to_zarr(path_zarr)
        logging.info("Data saved in Zarr format in %s", path_zarr)


if __name__ == "__main__":
  app.run(main)
