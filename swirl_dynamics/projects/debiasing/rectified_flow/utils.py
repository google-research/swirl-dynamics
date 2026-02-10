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
import logging
from typing import Any

import ml_collections
import numpy as np
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.projects.debiasing.rectified_flow import models as reflow_models
import xarray as xr


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
    min_time: The minimum time value for the model while training
    max_time: The maximum time value for the model while training
    normalize_qk: Whether to normalize query and key vectors in the attention
      layers.
    conditional_embedding: Whether to use conditional embedding in the model.
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
  min_time: float
  max_time: float
  normalize_qk: bool
  conditional_embedding: bool
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
      num_blocks=config.get("num_blocks"),
      dropout_rate=config.get("dropout_rate"),
      noise_embed_dim=config.get("noise_embed_dim"),
      padding=config.get("padding", "LONLAT"),
      use_attention=config.get("use_attention", False),
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
      min_time=config.get("min_time", 1e-4),
      max_time=config.get("max_time", 1 - 1e-4),
      normalize_qk=config.get("normalize_qk", True),
      conditional_embedding=config.get("conditional_embedding", False),
      use_3d_model=config.get("use_3d_model", False),
  )


# TODO: This seems to have some issues with the model.
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


