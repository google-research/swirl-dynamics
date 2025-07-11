# Copyright 2025 The swirl_dynamics Authors.
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

In this case we suppose that we have a model trained with climatology and
contiguous time chunks, which are embedded either in an additional dimension or
in the channel dimension.

In addition, we add a checkpointing mechanism to save the intermediate results
of the inference loop. This is useful for long running inference loops, which
might be interrupted for various reasons (e.g. preemption).
"""

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
from swirl_dynamics.projects.genfocal.debiasing import dataloaders
from swirl_dynamics.projects.genfocal.debiasing import inference_utils
from swirl_dynamics.projects.genfocal.debiasing import trainers
from swirl_dynamics.projects.genfocal.debiasing import utils
import tensorflow as tf
import xarray as xr


FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", None, "Directory to store model data.")
flags.DEFINE_string("workdir", None, "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def inference_pipeline(
    model_dir: str,
    config_eval: ml_collections.ConfigDict,
    workdir_path: str,
    lens2_member_indexer: tuple[dict[str, str], ...],
    lens2_variable_names: tuple[str, ...],
    era5_variables: dict[str, dict[str, int] | None],
    date_range: tuple[str, str] | None = None,
    verbose: bool = False,
    batch_size_eval: int = 16,
    variables: tuple[str, ...] = (
        "temperature",
        "specific_humidity",
        "mean_sea_level_pressure",
        "wind_speed",
    ),
    num_sampling_steps: int = 100,
    reverse_flow: bool = False,
) -> dict[str, np.ndarray]:
  """The evaluation pipeline.

  This function is used to run the inference pipeline. It loads the model,
  builds the dataloader, runs the inference loop and saves the results to disk.

  Args:
    model_dir: The directory with the model to infer.
    config_eval: The config file for the evaluation.
    workdir_path: The path to the work directory for saving the checkpoints.
    lens2_member_indexer: The member indexer for the LENS2 dataset.
    lens2_variable_names: The names of the variables in the LENS2 dataset.
    era5_variables: The names of the variables in the ERA5 dataset.
    date_range: The date range for the evaluation.
    verbose: Whether to print the config file.
    batch_size_eval: The batch size for the evaluation.
    variables: Names of physical fields.
    num_sampling_steps: The number of sampling steps for solving the ODE.
    reverse_flow: Whether to reverse the flow. If True, it debiases ERA5 to
      LENS2. If False, it debiases LENS2 to ERA5.

  Returns:
    A dictionary with the data both input and output with their corresponding
    time stamps.
  """
  del verbose  # Not used for now.

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

  # Checks the shapes of the inputs and the channels.
  utils.checks_input_shapes(config.input_shapes, config.out_channels)

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
      reverse_flow=reverse_flow,
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
      regime="test" if not reverse_flow else "eval",
  )

  output_list = []
  time_stamps = []

  # TODO: These are hardcoded for now. To be changed.
  n_lon = 240
  n_lat = 121
  n_field = len(variables)
  # We can debias either LENS2 to ERA5 or ERA5 to LENS2, by reversing the flow.
  if reverse_flow:
    par_keys = {
        "channel:mean": "channel:mean",
        "channel:std": "channel:std",
        "output_mean": "input_mean",
        "output_std": "input_std",
        "x_0": "x_1",
    }
  else:
    par_keys = {
        "channel:mean": "channel:mean",
        "channel:std": "channel:std",
        "output_mean": "output_mean",
        "output_std": "output_std",
        "x_0": "x_0",
    }

  logging.info("Batch size per device: %d", batch_size_eval)

  # Running the inference loop.
  for ii, batch in enumerate(eval_dataloader):

    logging.info("Iteration: %d", ii)

    # Saves the time stamps, regardless of whether the checkpoint exists or not.
    time_stamps.append(batch["input_time_stamp"])

    # Check if the sample already exists in a checkpoint file. If so, skip.
    path_checkpoint = f"{workdir_path}/checkpoint_{ii}.hdf5"
    if tf.io.gfile.exists(path_checkpoint):
      logging.info("Checkpoint %s already exists, skipping", path_checkpoint)
      saved_dict = hdf5_utils.read_all_arrays_as_dict(path_checkpoint)
      output_list.append(saved_dict["samples"])
      continue

    # Extractin only the fiekds that are needed for the parallel sampling.
    batch_par = {key1: batch[key2] for key1, key2 in par_keys.items()}
    # Reshaping the batch to be compatible with the parallel sampling function.
    parallel_batch = jax.tree.map(
        lambda x: x.reshape((num_devices, -1,) + x.shape[1:]), batch_par)

    # Running the parallel sampling and saving the output.
    samples = np.array(parallel_sampling_fn(parallel_batch))
    samples = samples.reshape((-1, n_lon, n_lat, n_field))

    # Saving the current output to disk as a checkpoint.
    dict_to_save = dict(
        samples=samples,
    )
    hdf5_utils.save_array_dict(path_checkpoint, dict_to_save)

    output_list.append(samples)

  # Concatenating the output and the time stamps.
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
  reverse_flow = config.get("reverse_flow", default=False)

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

  lens2_variable_names = config.get("lens2_variable_names")
  lens2_member_indexer = config.get("lens2_member_indexer")
  era5_variables = config.get("era5_variables")
  if None in (era5_variables, lens2_variable_names, lens2_member_indexer):
    raise ValueError(
        "ERA5 variables, LENS2 variable names and LENS2 member",
        "indexer should all be specified in the config file.",
    )

  # This is necessary due to ConfigDict issues with nested containers.
  if isinstance(era5_variables, ml_collections.ConfigDict):
    era5_variables = era5_variables.to_dict()

  num_sampling_steps = config.get("num_sampling_steps", default=100)
  logging.info("Using %d discretization steps.", num_sampling_steps)

  for lens2_indexer in lens2_member_indexer:
    print("Evaluating on CMIP dataset indexer: ", lens2_indexer)
    index_member = lens2_indexer

    # Changing the prefix path depending on whether we are debiasing LENS2 to
    # ERA5 or vice versa.
    prefix_path = "era5_to_lens2" if reverse_flow else "lens2_to_era5"

    # Checking that the file wasn't already saved.
    path_zarr = f"{workdir}/debiasing_{prefix_path}_{index_member}.zarr"

    if tf.io.gfile.exists(path_zarr):
      logging.info("File %s already exists, skipping", path_zarr)
    else:
      # Runs the inference pipeline on a single LENS2 member.
      data_dict = inference_pipeline(
          model_dir=model_dir,
          config_eval=config,
          workdir_path=workdir,
          date_range=date_range,
          batch_size_eval=batch_size_eval,
          variables=variables,
          lens2_member_indexer=(
              {"member": lens2_indexer},  # It needs to be a tuple.
          ),
          lens2_variable_names=lens2_variable_names,
          era5_variables=era5_variables,
          num_sampling_steps=num_sampling_steps,
          reverse_flow=reverse_flow,
      )

      if jax.process_index() == 0:
        logging.info("Saving data in Zarr format")

        print(f"Shape of time stamps {data_dict['time_stamps'].shape}")

        # TODO: Add the variable names to the zarr file here.
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
