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

"""Utility functions for inference."""

import functools
import logging
from typing import Any, Literal

import grain.python as pygrain
import jax
import jax.numpy as jnp
import ml_collections
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.lib.solvers import ode as ode_solvers
from swirl_dynamics.projects.debiasing.rectified_flow import dataloaders
from swirl_dynamics.projects.debiasing.rectified_flow import models as reflow_models
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states


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

_LENS2_MEMBER_INDEXER = ("cmip6_1001_001",)
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV")
_LENS2_VARIABLES = {v: _LENS2_MEMBER_INDEXER[0] for v in _LENS2_VARIABLE_NAMES}


def move_time_to_channel(
    input_array: jax.Array, time_chunk_size: int, time_to_channel: bool = True
) -> jax.Array:
  """Moves the time dimension to a new or the channel dimension.

  Args:
    input_array: The input array to move the time dimension.
    time_chunk_size: The size of the time chunks in the batch.
    time_to_channel: Whether to move the time dimension to the channel dimension
      or add it to another dimension between the batch and the spatial
      dimensions.

  Returns:
    The input array with the time dimension moved to the channel dimension or a
    new dimension between the batch and the spatial dimensions.
  """
  if input_array.shape[0] % time_chunk_size != 0:
    raise ValueError("Batch dimension should be multiple a time chunk size.")

  new_chunk = input_array.shape[0] // time_chunk_size
  new_channel = input_array.shape[-1] * time_chunk_size

  # Shape: (new_chunk, time_chunk_size, lon, lat, channel)
  input_new = jnp.reshape(
      input_array, (new_chunk, time_chunk_size, *input_array.shape[1:])
  )

  if time_to_channel:
    # Shape: (new_chunk, lon, lat, channel, time_chunk_size)
    input_new = jnp.moveaxis(input_new, 1, -1)
    # Shape: (new_chunk, lon, lat, new_channel)
    input_new = jnp.reshape(input_new, (*input_new.shape[:-2], new_channel))

  return input_new


def move_channel_to_time(
    input_array: jax.Array, time_chunk_size: int, time_to_channel: bool = True
) -> jax.Array:
  """Moves the time dimension back to the batch dimension.

  Args:
    input_array: The input array to move the time dimension.
    time_chunk_size: The size of the time chunks in the batch.
    time_to_channel: Whether to move the time dimension back from the channel
      dimension or to squeeze the first two dimensions together.

  Returns:
    The input array with the time dimension moved back to the batch dimension.
    The output of this function is the inverse of the move_time_to_channel
    function.
  """
  new_chunk = input_array.shape[0] * time_chunk_size
  new_channel = input_array.shape[-1] // time_chunk_size

  if time_to_channel:
    # Shape: (batch, lon, lat, new_channel, time_chunk_size)
    input_new = jnp.reshape(
        input_array, (*input_array.shape[:-1], new_channel, time_chunk_size)
    )
    # Shape: (batch, time_chunk_size, lon, lat, new_channel)
    input_new = jnp.moveaxis(input_new, -1, 1)
    # Shape: (new_batch, lon, lat, new_channel)
    input_new = jnp.reshape(input_new, (new_chunk, *input_new.shape[2:]))
  else:
    input_new = jnp.reshape(input_array, (new_chunk, *input_array.shape[2:]))
  return input_new


def sampling_from_batch(
    batch: models.BatchType,
    model: reflow_models.ReFlowModel | reflow_models.ConditionalReFlowModel,
    trained_state: train_states.BasicTrainState,
    num_sampling_steps: int,
    time_chunk_size: int,
    time_to_channel: bool = True,
    reverse_flow: bool = False,
) -> jax.Array:
  """Sampling from a batch using a reflow model.

  Here the batch is expected to have the following keys:
    - x_0: The initial state of the ODE.
    - channel:mean: The normalized mean of the LENS2 input, which is fed to the
      model as a conditioning field.
    - channel:std:  The normalized standard deviation of the LENS2 input, which
      is fed to the model as a conditioning field.
    - output_mean: The mean of the output distribution.
    - output_std: The standard deviation of the output distribution.

  The output is expected to have the same shape as the input given by
  batch["x_0"].

  Args:
    batch: The batch to sample from.
    model: The model that encapsulates the flow model.
    trained_state: The trained state of the model.
    num_sampling_steps: The number of sampling steps for solving the ODE.
    time_chunk_size: The size of the time chunks in the batch.
    time_to_channel: Whether to move the time dimension to the channel
      dimension. Or just another dimension between the batch and the spatial
      dimensions.
    reverse_flow: Whether to use the reverse flow, i.e., integrate from 1 to 0,
      instead of 0 to 1.

  Returns:
    The sampled output.
  """
  # Checks the keys in the batch.
  if "channel:mean" not in batch or "channel:std" not in batch:
    raise ValueError("Batch does not contain the channel:mean or channel:std.")
  if "output_mean" not in batch or "output_std" not in batch:
    raise ValueError("Batch does not contain the output_mean or output_std.")
  if "x_0" not in batch:
    raise ValueError("Batch does not contain the x_0 field.")

  # Setting up the conditional ODE for the sampling.
  cond = {
      "channel:mean": move_time_to_channel(
          batch["channel:mean"], time_chunk_size, time_to_channel
      ),
      "channel:std": move_time_to_channel(
          batch["channel:std"], time_chunk_size, time_to_channel
      ),
  }

  # Sets up the vector field for the sampling. This is a temporal one in order
  # to easily compute the reverse flow.
  latent_dynamics_fn_tmp = ode_solvers.nn_module_to_dynamics(
      model.flow_model,
      autonomous=False,
      cond=cond,
      is_training=False,
  )

  # If we want to use the reverse flow, we need to change the sign of the
  # dynamics function and the evaluation time.
  if reverse_flow:
    latent_dynamics_fn = (
        lambda x, t, params: -latent_dynamics_fn_tmp(x, 1.0 - t, params)
    )
  else:
    latent_dynamics_fn = latent_dynamics_fn_tmp

  integrator = ode_solvers.RungeKutta4()
  integrate_fn = functools.partial(
      integrator,
      latent_dynamics_fn,
      tspan=jnp.arange(0.0, 1.0, 1.0 / num_sampling_steps),
      params=trained_state.model_variables,
  )

  # Running the integration. Then take the last state.
  out = integrate_fn(
      move_time_to_channel(batch["x_0"], time_chunk_size, time_to_channel)
  )[-1, :]

  # Denormalize the output according to output (ERA5) climatology.
  # In the case we use the reverse flow, the output is the LENS3 climatology.
  # This needs to be changed outside this function by modifying the statistics.
  era5_std = move_time_to_channel(
      batch["output_std"], time_chunk_size, time_to_channel
  )
  era5_mean = move_time_to_channel(
      batch["output_mean"], time_chunk_size, time_to_channel
  )
  out = out * era5_std + era5_mean

  # Move the channel dimension to the time dimension.
  out = move_channel_to_time(out, time_chunk_size, time_to_channel)

  return out


def sampling_era5_to_era5_from_batch(
    batch: models.BatchType,
    model: reflow_models.ReFlowModel | reflow_models.ConditionalReFlowModel,
    trained_state: train_states.BasicTrainState,
    num_sampling_steps: int,
    time_chunk_size: int,
    dims_geopotential: tuple[int, ...],
    time_to_channel: bool = True,
) -> jax.Array:
  """Sampling from a batch using a reflow model.

  Here the batch is expected to have the following keys:
    - x_1: The initial state of the ODE corresponding to a normalized ERA5
      sample.
    - channel:mean: The normalized mean of the LENS2 input, which is fed to the
      model as a conditioning field.
    - channel:std:  The normalized standard deviation of the LENS2 input, which
      is fed to the model as a conditioning field.
    - output_mean: The mean of the output distribution.
    - output_std: The standard deviation of the output distribution.
    - input_mean: The mean (unnormalized) of the input distribution.
    - input_std: The standard deviation  (unnormalized) of the input
      distribution.

  The output is expected to have the same shape as the input given by
  batch["x_1"].

  Args:
    batch: The batch to sample from.
    model: The model that encapsulates the flow model.
    trained_state: The trained state of the model.
    num_sampling_steps: The number of sampling steps for solving the ODE.
    time_chunk_size: The size of the time chunks in the batch.
    dims_geopotential: The indices of the fields corresponding to the
      geopotentials. They needs to be normalized by the universal gravitational
      constant. This is necessary as ERA5 and LENS2 use different units for the
      geopotentials.
    time_to_channel: Whether to move the time dimension to the channel
      dimension. Or just another dimension between the batch and the spatial
      dimensions.

  Returns:
    The sampled output.
  """
  # Checks the keys in the batch.
  if "channel:mean" not in batch or "channel:std" not in batch:
    raise ValueError("Batch does not contain the channel:mean or channel:std.")
  if "input_mean" not in batch or "input_std" not in batch:
    raise ValueError("Batch does not contain the input_mean or input_std.")
  if "output_mean" not in batch or "output_std" not in batch:
    raise ValueError("Batch does not contain the output_mean or output_std.")
  if "x_1" not in batch:
    raise ValueError("Batch does not contain the x_1 field.")

  # Setting up the conditional ODE for the sampling.
  cond = {
      "channel:mean": move_time_to_channel(
          batch["channel:mean"], time_chunk_size, time_to_channel
      ),
      "channel:std": move_time_to_channel(
          batch["channel:std"], time_chunk_size, time_to_channel
      ),
  }

  # Extracts the (unnormalized) climatology for the input and output.
  lens2_std = move_time_to_channel(
      batch["input_std"], time_chunk_size, time_to_channel
  )
  lens2_mean = move_time_to_channel(
      batch["input_mean"], time_chunk_size, time_to_channel
  )
  era5_std = move_time_to_channel(
      batch["output_std"], time_chunk_size, time_to_channel
  )
  era5_mean = move_time_to_channel(
      batch["output_mean"], time_chunk_size, time_to_channel
  )

  latent_dynamics_fn = ode_solvers.nn_module_to_dynamics(
      model.flow_model,
      autonomous=False,
      cond=cond,
      is_training=False,
  )

  integrator = ode_solvers.RungeKutta4()
  integrate_fn = functools.partial(
      integrator,
      latent_dynamics_fn,
      tspan=jnp.arange(0.0, 1.0, 1.0 / num_sampling_steps),
      params=trained_state.model_variables,
  )

  # Denormalizes the input according to the output (ERA5) climatology and it
  # normalizes according to the input (LENS2) climatology.
  era5_norm_era5 = move_time_to_channel(
      batch["x_1"], time_chunk_size, time_to_channel
  )
  denorm = era5_norm_era5 * era5_std + era5_mean
  # Normalize by the universal gravitational constant the geopotentials.
  norm_factor_geo_pot = jnp.ones(
      (lens2_std.ndim - 1) * (1,) + (lens2_std.shape[-1],)
  )
  norm_factor_geo_pot = norm_factor_geo_pot.at[..., dims_geopotential].set(9.8)
  denorm = denorm / norm_factor_geo_pot
  # Normalize using the lens2 statistics.
  era5_norm_lens2 = (denorm - lens2_mean) / lens2_std

  # Running the integration. Then take the last state.
  out = integrate_fn(era5_norm_lens2)[-1, :]

  # Denormalize the output according to output (ERA5) Climatology.
  out = out * era5_std + era5_mean

  # Move the channel dimension to the time dimension.
  out = move_channel_to_time(out, time_chunk_size, time_to_channel)

  return out


def build_model_from_config(
    config: ml_collections.ConfigDict,
) -> reflow_models.ReFlowModel:
  """Builds the model from config file.

  This function is used to build the model from the config file as saved in the
  training step. This function will become obsolete once the code is migrated to
  fiddle, but we will keep it for now to be able to load the models saved in
  previous experiments.

  Args:
    config: The config file for the model as saved in the training step.

  Returns:
    The model as a ReflowModel or ConditionalReFlowModel.
  """

  # Adding the conditional embedding for the FILM layer.
  if config.get("conditional_embedding", default=False):
    print("Using conditional embedding")
    cond_embed_fn = unets.EmbConvMerge
  else:
    cond_embed_fn = None

  if config.get("use_3d_model", default=False):
    print("Using 3D U-ViT model")
    flow_model = reflow_models.RescaledUnet3d(
        out_channels=config.out_channels,
        num_channels=config.num_channels,
        downsample_ratio=config.downsample_ratio,
        num_blocks=config.num_blocks,
        noise_embed_dim=config.noise_embed_dim,
        padding=config.padding,
        dropout_rate=config.dropout_rate,
        use_spatial_attention=config.use_spatial_attention,
        use_temporal_attention=config.use_temporal_attention,
        resize_to_shape=config.resize_to_shape,
        use_position_encoding=config.use_position_encoding,
        # ffn_type=config.get("ffn_type", default="dense"),
        num_heads=config.num_heads,
        normalize_qk=config.normalize_qk,
    )
  else:
    print("Using 2D U-ViT model")
    flow_model = reflow_models.RescaledUnet(
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

  # Building the model. By default the input shape doesn't include the batch
  # dimension, wheres the input_shapes in the config file includes a dummy batch
  # dimension of size 1.
  model = reflow_models.ConditionalReFlowModel(
      input_shape=config.input_shapes[0][1:],
      cond_shape={
          "channel:mean": config.input_shapes[0][1:],
          "channel:std": config.input_shapes[0][1:],
      },
      flow_model=flow_model,
  )

  return model


def build_inference_dataloader(
    config: ml_collections.ConfigDict,
    config_eval: ml_collections.ConfigDict,
    batch_size: int,
    lens2_member_indexer: tuple[dict[str, str], ...] | None = None,
    lens2_variable_names: tuple[str, ...] | None = None,
    era5_variables: dict[str, dict[str, Any]] | None = None,
    date_range: tuple[str, str] | None = None,
    regime: Literal["train", "eval"] = "eval",
) -> pygrain.DataLoader:
  """Loads the data loaders.

  Args:
    config: The config file used for the training experiment.
    config_eval: The config file used for the evaluation experiment.
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

  if date_range is None:
    logging.info("Using the default date ranges.")
    if regime == "train":
      date_range = config.date_range_train
    elif regime == "eval":
      date_range = config.date_range_eval

  if lens2_member_indexer is None:
    lens2_member_indexer = config.get(
        "lens2_member_indexer", _LENS2_MEMBER_INDEXER
    )
    if isinstance(lens2_member_indexer, ml_collections.ConfigDict):
      lens2_member_indexer = lens2_member_indexer.to_dict()

  # Extract the paths from the config file or use the default values.
  input_dataset_path = config_eval.get(
      "input_dataset_path", default=_LENS2_DATASET_PATH
  )
  input_climatology = config_eval.get(
      "input_climatology", default=_LENS2_STATS_PATH
  )
  input_mean_stats_path = config_eval.get(
      "input_mean_stats_path", default=_LENS2_MEAN_CLIMATOLOGY_PATH
  )
  input_std_stats_path = config_eval.get(
      "input_std_stats_path", default=_LENS2_STD_CLIMATOLOGY_PATH
  )
  output_dataset_path = config_eval.get(
      "output_dataset_path", default=_ERA5_DATASET_PATH
  )
  output_climatology = config_eval.get(
      "output_climatology", default=_ERA5_STATS_PATH
  )

  logging.info("input_dataset_path: %s", input_dataset_path)
  logging.info("input_climatology: %s", input_climatology)
  logging.info("input_mean_stats_path: %s", input_mean_stats_path)
  logging.info("input_std_stats_path: %s", input_std_stats_path)
  logging.info("output_dataset_path: %s", output_dataset_path)
  logging.info("output_climatology: %s", output_climatology)

  dataloader = dataloaders.create_ensemble_lens2_era5_loader_with_climatology(
      date_range=date_range,
      batch_size=batch_size,
      shuffle=False,
      input_dataset_path=input_dataset_path,
      input_climatology=input_climatology,
      input_mean_stats_path=input_mean_stats_path,
      input_std_stats_path=input_std_stats_path,
      input_variable_names=lens2_variable_names,
      input_member_indexer=lens2_member_indexer,
      output_dataset_path=output_dataset_path,
      output_climatology=output_climatology,
      output_variables=era5_variables,
      time_stamps=True,
      inference_mode=True,  # Using the inference dataset.
      num_epochs=1,  # This is so the loop stops automatically.
  )

  return dataloader
