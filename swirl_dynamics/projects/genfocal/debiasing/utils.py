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

"""Utility functions for the rectified flow models.

Here we include the different configuration classes and the functions to load
them from either a json or a yaml file.
"""

import dataclasses
import functools
import logging
from typing import Any, Literal

import grain.python as pygrain
import jax
import ml_collections
import numpy as np
import optax
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.projects.genfocal.debiasing import dataloaders
from swirl_dynamics.projects.genfocal.debiasing import models as reflow_models
import xarray as xr


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataLoaderConfig:
  """Configuration class for the Dataloaders.

  Attributes:
    date_range: Date range for the data loader, in the format of ("start_date",
      "end_date").
    batch_size: Batch size for the data loader.
    time_batch_size: Lenght of the debiasing sequence, it needs to divide the
      the batch size exactly.
    chunk_size: The size of the chunk for the contiguous data (for each process)
      in number of snapshots. This is a multiple of the time_batch_size, and
      accounts for the local batch size.
    shuffle: Whether to shuffle the data.
    worker_count: Number of workers for the data loader.
    input_dataset_path: Path to the input dataset.
    input_climatology: Path to the input climatology.
    input_mean_stats_path: Path to the input mean stats of the climatology.
    input_std_stats_path: Path to the input std stats of the climatology.
    input_variable_names: Names of the variables to be used in the input.
    input_member_indexer: Indexer for the input dataset.
    output_variables: Variables to be used in the output.
    output_dataset_path: Path to the output dataset.
    output_climatology: Path to the output climatology.
    time_to_channel: Whether to use time as the channel dimension. See the the
      definition of the `move_time_to_channel` function in the
      inference_utils.py file. False for model with three-dimensional topology,
      i.e., the input is a 4-tensor, and True for model with two-dimensional
      topology, i.e., the input is a 3-tensor.
    yearly_offset: The yearly offset for the data loader.
    daily_offset: The daily offset for the data loader, this is used to sample
      data in which the coupling have a small offset between the input and
      output.
    use_tensor_coupling: Whether to use the tensor coupling in the data loader.
      This is used to sample data in which the coupling is the tensor product of
      the target marginals.
  """

  date_range: tuple[str, str]
  batch_size: int
  time_batch_size: int
  chunk_size: int
  shuffle: bool
  worker_count: int | None
  input_dataset_path: str
  input_climatology: str
  input_mean_stats_path: str
  input_std_stats_path: str
  input_variable_names: tuple[str, ...]
  input_member_indexer: tuple[dict[str, str], ...]
  output_variables: dict[str, dict[str, Any] | None]
  output_dataset_path: str
  output_climatology: str
  time_to_channel: bool
  yearly_offset: int
  daily_offset: int
  use_tensor_coupling: bool


def get_dataloader_config(
    config: ml_collections.ConfigDict,
    regime: Literal["train", "eval"] = "train",
    verbose: bool = False,
) -> DataLoaderConfig:
  """Returns the data loader config from a ConfigDict.

  Args:
    config: The configuration from a ConfigDict, this is usually loaded from a
      json or yaml file.
    regime: In which regime the data loader is used. This is used to select the
      date range and the batch size from the config file.
    verbose: Whether to print the ERA5 variables and the LENS2 variable names
      and indexer. This is useful for debugging.

  Returns:
    The data loader config as a DataLoaderConfig object.
  """
  if regime == "train":
    date_range = config.get("date_range_train")
    batch_size = config.get("batch_size")
    yearly_offset = config.get("yearly_offset", default=0)
    daily_offset = config.get("daily_offset", default=0)
    use_tensor_coupling = config.get("use_tensor_coupling", default=False)
  elif regime == "eval":
    date_range = config.get("date_range_eval")
    batch_size = config.get("batch_size_eval")
    yearly_offset = 0
    daily_offset = 0
    use_tensor_coupling = False
  else:
    raise ValueError(f"Unknown regime: {regime}")

  # This is to avoid the default behavior of the ConfigDict, which converts to
  # ConfigDict any nested dictionaries.
  era5_variables = config.get("era5_variables")
  if isinstance(era5_variables, ml_collections.ConfigDict):
    era5_variables = era5_variables.to_dict()

  lens2_member_indexer = config.get("lens2_member_indexer")
  if isinstance(lens2_member_indexer, ml_collections.ConfigDict):
    lens2_member_indexer = lens2_member_indexer.to_dict()

  if verbose:
    print(f"Date range for {regime}: {date_range}", flush=True)
    print(f"Batch size for {regime}: {batch_size}", flush=True)
    print("ERA5 variables", flush=True)
    print(era5_variables, flush=True)
    print("LENS2 variable names", flush=True)
    print(config.get("lens2_variable_names"), flush=True)
    print("LENS2 member indexer", flush=True)
    print(lens2_member_indexer, flush=True)
    print(f"Number of years offset {yearly_offset}", flush=True)

  return DataLoaderConfig(
      date_range=date_range,
      batch_size=batch_size,
      time_batch_size=config.get("time_batch_size"),
      chunk_size=config.get("chunk_size"),
      shuffle=config.get("shuffle", default=True),
      worker_count=config.get("num_workers", default=0),
      input_dataset_path=config.get("lens2_dataset_path"),
      input_climatology=config.get("lens2_stats_path"),
      input_mean_stats_path=config.get("lens2_mean_stats_path"),
      input_std_stats_path=config.get("lens2_std_stats_path"),
      input_variable_names=config.get("lens2_variable_names"),
      input_member_indexer=lens2_member_indexer,
      output_variables=era5_variables,
      output_dataset_path=config.get("era5_dataset_path"),
      output_climatology=config.get("era5_stats_path"),
      time_to_channel=config.get("time_to_channel", default=False),
      yearly_offset=yearly_offset,
      daily_offset=daily_offset,
      use_tensor_coupling=use_tensor_coupling,
  )


def build_dataloader_from_config(
    config: DataLoaderConfig,
) -> pygrain.DataLoader:
  """Builds the dataloader from the configuration class.

  Args:
    config: The configuration class for the dataloaders.

  Returns:
    The dataloader as a pygrain.DataLoader object.
  """
  train_dataloader = dataloaders.create_ensemble_lens2_era5_time_chunked_loader_with_climatology(
      date_range=config.date_range,
      batch_size=config.batch_size,
      time_batch_size=config.time_batch_size,
      chunk_size=config.chunk_size,
      shuffle=config.shuffle,
      worker_count=config.worker_count,
      input_dataset_path=config.input_dataset_path,
      input_climatology=config.input_climatology,
      input_mean_stats_path=config.input_mean_stats_path,
      input_std_stats_path=config.input_std_stats_path,
      input_variable_names=config.input_variable_names,
      input_member_indexer=config.input_member_indexer,
      output_variables=config.output_variables,
      output_dataset_path=config.output_dataset_path,
      output_climatology=config.output_climatology,
      time_to_channel=config.time_to_channel,
      yearly_offset=config.yearly_offset,
      daily_offset=config.daily_offset,
  )
  return train_dataloader


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReFlowModelConfig:
  """Configuration class for the rectified flow model.

  Attributes:
    out_channels: Number of output channels. This is usually the number of
      output variables.
    num_channels: Number of channels at each downsample level.
    downsample_ratio: tuple[int, ...]
    bfloat16: Whether to explicitly use bfloat16 for representing the model
      weights (Experimental).
    resize_to_shape: The (optional) shape to resize the samples to, such that
      they can be conveniently downsampled at integer-valued factors.
    num_blocks: Number of processing blocks in the UNet backbone.
    dropout_rate: Rate of dropout to apply to the model.
    noise_embed_dim: The dimension of the noise (or time) embedding.
    padding: The padding method for the convolution layers.
    use_attention: Whether to use attention at each level in the UNet backbone.
      (This is only used for the 2D model).
    use_spatial_attention: Whether to use spatial attention at each level in the
      UNet backbone. (This is only used for the 3D model).
    use_temporal_attention: Whether to use temporal attention at each level in
      the UNet backbone. (This is only used for the 3D model).
    use_position_encoding: Whether to use positional encoding.
    num_heads: Number of attention heads in the multi-head attention layers.
    ema_decay: The decay rate for the exponential moving average (EMA) of the
      model weights.
    final_act_fun: The activation function to use at the final convolution
      layer.
    use_skips: Whether to use skip connections in the UNet backbone.
    use_weight_global_skip: Whether to use weight-shared skip connections in the
      UNet backbone.
    use_local: Whether to use local convolution layers (i.e. no weight sharing)
    input_shapes: The input shapes for the model. The first tuple is for the
      input samples, and the second tuple is output.
    same_dimension: Whether the input and output have the same spatial
      dimensions.
    time_sampler: The method to sample the time used in the model.
    min_time: The minimum time value for the model while training
    max_time: The maximum time value for the model while training
    normalize_qk: Whether to normalize query and key vectors in the attention
      layers.
    conditional_embedding: Whether to use conditional embedding in the model.
    ffn_type: The type of the feed-forward network to use in the model in the
      Attention layers.
    use_3d_model: Whether to use the 3D UViT or a regular 2D UViT architecture.
  """

  out_channels: int
  num_channels: tuple[int, ...]
  downsample_ratio: tuple[int, ...]
  bfloat16: bool
  resize_to_shape: tuple[int, int]
  num_blocks: int
  dropout_rate: float
  noise_embed_dim: int
  padding: str
  use_attention: bool
  use_spatial_attention: tuple[bool, ...]
  use_temporal_attention: tuple[bool, ...]
  use_position_encoding: bool
  num_heads: int
  ema_decay: bool
  use_skips: bool
  use_weight_global_skip: bool
  use_local: bool
  input_shapes: tuple[tuple[int, ...], tuple[int, ...]]
  same_dimension: bool
  time_sampler: Literal["uniform", "lognorm"]
  min_time: float
  max_time: float
  normalize_qk: bool
  conditional_embedding: bool
  ffn_type: Literal["dense", "conv"]
  use_3d_model: bool


def get_model_config(config: ml_collections.ConfigDict) -> ReFlowModelConfig:
  """Returns the model config from the config file."""

  # This is to avoid issues with nested tuples being silently converted to
  # nested lists when reading from a json file.
  input_shapes_list = config.get("input_shapes")
  input_shapes = (tuple(input_shapes_list[0]), tuple(input_shapes_list[1]))

  return ReFlowModelConfig(
      out_channels=config.get("out_channels"),
      num_channels=tuple(config.get("num_channels")),
      downsample_ratio=tuple(config.get("downsample_ratio")),
      bfloat16=config.get("bfloat16", False),
      resize_to_shape=tuple(config.get("resize_to_shape")),
      use_3d_model=config.get("use_3d_model", True),
      num_blocks=config.get("num_blocks"),
      dropout_rate=config.get("dropout_rate"),
      noise_embed_dim=config.get("noise_embed_dim"),
      padding=config.get("padding", "LONLAT"),
      use_attention=config.get("use_attention", False),  # Only for 2D model.
      use_spatial_attention=tuple(
          config.get("use_spatial_attention", (False,))
      ),
      use_temporal_attention=tuple(
          config.get("use_temporal_attention", (False,))
      ),
      use_position_encoding=config.get("use_position_encoding", True),
      num_heads=config.get("num_heads"),
      ema_decay=config.get("ema_decay", 0.99),
      use_skips=config.get("use_skips", True),
      use_weight_global_skip=config.get("use_weight_global_skip", False),
      use_local=config.get("use_local", False),
      input_shapes=input_shapes,
      same_dimension=config.get("same_dimension", True),
      time_sampler=config.get("time_sampler", "uniform"),
      min_time=config.get("min_time", 1e-4),
      max_time=config.get("max_time", 1 - 1e-4),
      normalize_qk=config.get("normalize_qk", True),
      conditional_embedding=config.get("conditional_embedding", False),
      ffn_type=config.get("ffn_type", default="dense"),
  )


def build_model_from_config(
    config: ReFlowModelConfig,
) -> reflow_models.ReFlowModel:
  """Builds the model from config file.

  This function is used to build the model from a configuration class. This
  function may become obsolete once the code is migrated to
  fiddle, but we will keep it for now to be able to load the models saved in
  previous experiments.

  Args:
    config: The config file for the model as saved in the training step.

  Returns:
    The model as a ReflowModel or ConditionalReFlowModel.
  """

  # Adding the conditional embedding for the FILM layer.
  if config.conditional_embedding:
    print("Using conditional embedding")
    cond_embed_fn = unets.EmbConvMerge
  else:
    cond_embed_fn = None

  dtype = jax.numpy.bfloat16 if config.bfloat16 else jax.numpy.float32

  if config.use_3d_model:
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
        num_heads=config.num_heads,
        normalize_qk=config.normalize_qk,
        ffn_type=config.ffn_type,
        dtype=dtype,
        param_dtype=dtype,
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
        dtype=dtype,
        param_dtype=dtype,
    )

  # Setting up the time sampler.
  if config.time_sampler == "lognorm":
    time_sampler = reflow_models.lognormal_sampler()
  elif config.time_sampler == "uniform":
    time_sampler = functools.partial(
        jax.random.uniform, dtype=jax.numpy.float32
    )
  else:
    raise ValueError(f"Unknown time sampler: {config.time_sampler}")

  # Checks the shapes of the input and conditioning.
  input_shape = config.input_shapes[0][1:]
  cond_shape = {
      "channel:mean": config.input_shapes[0][1:],
      "channel:std": config.input_shapes[0][1:],
  }
  # TODO: Change the input type.
  # check_shapes(config, input_shape, cond_shape)

  # Building the model. By default the input shape doesn't include the batch
  # dimension, wheres the input_shapes in the config file includes a dummy batch
  # dimension of size 1.
  model = reflow_models.ConditionalReFlowModel(
      input_shape=input_shape,  # This must agree with the sample shape.
      cond_shape=cond_shape,
      time_sampling=time_sampler,
      min_train_time=config.min_time,
      max_train_time=config.max_time,
      flow_model=flow_model,
  )

  return model


@dataclasses.dataclass(frozen=True, kw_only=True)
class OptimizerConfig:
  """Configuration class for the optimizer.

  Attributes:
    init_value: Initial value of the learning rate.
    peak_value: Peak value of the learning rate for the schedule.
    warmup_steps: Number of steps to linearly warm up the learning rate.
    decay_steps: Number of steps to decay the learning rate.
    end_value: The final value of the learning rate.
    max_norm: Maximum norm of the gradient before clipping.
    beta1: The momentum for the Adam optimizer.
  """

  init_value: float
  peak_value: float
  warmup_steps: int
  decay_steps: int
  end_value: float
  max_norm: float
  beta1: float


def get_optimizer_config(config: ml_collections.ConfigDict) -> OptimizerConfig:
  """Returns the optimizer config from the config dict."""
  return OptimizerConfig(
      init_value=config.get("initial_lr"),
      peak_value=config.get("peak_lr"),
      warmup_steps=config.get("warmup_steps"),
      decay_steps=config.get("num_train_steps"),
      end_value=config.get("end_lr"),
      max_norm=config.get("max_norm"),
      beta1=config.get("beta1"),
  )


def build_optimizer_from_config(
    config: OptimizerConfig,
) -> optax.GradientTransformation:
  """Builds the optimizer from the config object.

  Args:
    config: The configuration object.

  Returns:
    The optimizer as an optax.GradientTransformation.
  """
  schedule = optax.warmup_cosine_decay_schedule(
      init_value=config.init_value,
      peak_value=config.peak_value,
      warmup_steps=config.warmup_steps,
      decay_steps=config.decay_steps,
      end_value=config.end_value,
  )

  optimizer = optax.chain(
      optax.clip_by_global_norm(config.max_norm),
      optax.adam(
          learning_rate=schedule,
          b1=config.beta1,
      ),
  )
  return optimizer


def parse_vars_from_config_to_dict(
    config: ml_collections.ConfigDict,
) -> dict[str, Any]:
  """Function to parse the variables names from the configdict to a dictionary.

  Args:
    config: The instance of the configuration dictionary.

  Returns:
    A dictionary with the variables, if they are not in the config file,
    the default values are used as defined at the top of the file.
  """
  # The era5 variables are a dictionary so they get transformed to a ConfigDict.
  era5_variables = config.get("era5_variables", None)
  if era5_variables is None:
    raise ValueError("era5_variables should be defined in the config file.")

  if isinstance(era5_variables, ml_collections.ConfigDict):
    era5_variables = era5_variables.to_dict()

  # These are a tuple of strings.
  lens2_variable_names = config.get("lens2_variable_names", None)
  if lens2_variable_names is None:
    raise ValueError(
        "lens2_variable_names should be defined in the config file."
    )

  # These are a tuple of dictionaries. They elements of the tuple are not
  # converted to ConfigDict.
  lens2_member_indexer_tuple = config.get("lens2_member_indexer", None)
  if lens2_member_indexer_tuple is None:
    raise ValueError(
        "lens2_member_indexer should be defined in the config file."
    )

  return dict(
      era5_variables=era5_variables,
      lens2_variable_names=lens2_variable_names,
      lens2_member_indexer_tuple=lens2_member_indexer_tuple,
  )


def checks_input_shapes(
    input_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    out_channels: int,
) -> None:
  """Checks the shapes of the inputs and the channels.

  Args:
    input_shapes: The shapes of both input and output of the network.
    out_channels: The number of the channels of the output of the network.
  """
  if (
      input_shapes[0][-1] != input_shapes[1][-1]
      or input_shapes[0][-1] != out_channels
  ):
    raise ValueError(
        "The number of channels in the input and output must be the same."
    )


def check_shapes(
    config: ReFlowModelConfig,
    input_shape: tuple[int, ...],
    cond_shape: dict[str, tuple[int, ...]],
) -> None:
  """Checks the shapes of the input and conditioning.

  Args:
    config: The instance of the configuration class.
    input_shape: The shape of the input samples.
    cond_shape: The shape of the conditioning signal.

  Raises:
    ValueError: If the shapes are not compatible with the model.
  """
  if config.use_3d_model:
    if len(input_shape) != 4:
      raise ValueError("Input shape must be 4D for 3D model.")
    if len(cond_shape["channel:mean"]) != 4:
      raise ValueError("Conditional shape of the mean must be 4D for 3D model.")
    if len(cond_shape["channel:std"]) != 4:
      raise ValueError("Conditional shape of the std must be 4D for 3D model.")
  else:
    if len(input_shape) != 3:
      raise ValueError("Input shape must be 3D for 2D model.")
    if len(cond_shape["channel:mean"]) != 3:
      raise ValueError("Conditional shape of the mean must be 3D for 2D model.")
    if len(cond_shape["channel:std"]) != 3:
      raise ValueError("Conditional shape of the std must be 3D for 2D model.")


def save_data_in_zarr(
    time_stamps_array: np.ndarray,
    input_array: np.ndarray,
    output_array: np.ndarray,
    model_dir: str,
    workdir: str,
    lens2_member_name: str,
    target_array: np.ndarray | None = None,
    time_chunk: int = 256,
) -> None:
  """Saves the snapshots in Zarr format after debiasing.

  This function is used to save the snapshots in Zarr format after debiasing.
  The files are names using the xm experiment id and the index of the LENS2
  member. So they can be easily identified and traced back to the experiment
  that trained the model used for debiasing.

  Args:
    time_stamps_array: Time stamps in np.datetime64 format.
    input_array: Array with the input data from a member of LENS2.
    output_array: Array with the reflow data from a member of LENS2.
    model_dir: The directory containing the model checkpoint and the model
      configuration file.
    workdir: The current working directory where the Zarr file will be saved.
    lens2_member_name: The name of the index of the LENS2 member.
    target_array: Array with the target data from a member of ERA5. In the case
      of no target data, we don't save it to the Zarr file.
    time_chunk: The chunk size of the time dimension in the Zarr file.
  """
  # Saving evaluation snapshots in Zarr format.
  logging.info("Saving data in Zarr format")
  print(f"Shape of time stamps {time_stamps_array.shape}")

  path_zarr = f"{workdir}/debiasing_lens2_to_era5_{lens2_member_name}.zarr"  # pylint: disable=line-too-long


  ds = {}

  # For evaluation we save the target data.
  if target_array:
    ds["era5"] = xr.DataArray(
        target_array,
        dims=["time", "longitude", "latitude", "variables"],
        coords={"time": time_stamps_array},
    )

  ds["lens2"] = xr.DataArray(
      input_array,
      dims=["time", "longitude", "latitude", "variables"],
      coords={"time": time_stamps_array},
  )
  ds["reflow"] = xr.DataArray(
      output_array,
      dims=["time", "longitude", "latitude", "variables"],
      coords={"time": time_stamps_array},
  )

  ds = xr.Dataset(ds)
  ds = ds.chunk(
      {"time": time_chunk, "longitude": -1, "latitude": -1, "variables": -1}
  )
  ds.to_zarr(path_zarr)
  logging.info("Data saved in Zarr format in %s", path_zarr)
