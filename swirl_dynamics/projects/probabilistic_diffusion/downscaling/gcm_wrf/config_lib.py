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

"""Gin model configuration classes specifying the input and output datasets."""

import dataclasses
from typing import Literal
import gin


@gin.configurable
@dataclasses.dataclass
class DatasetConfig(object):
  """A dataset configuration.

  Attributes:
    input_dataset: Path to Zarr dataset from which conditioning inputs are
      sampled.
    input_stats: Path to Zarr dataset containing statistics of the input data,
      used to normalize conditioning inputs.
    input_variables: The names of the variables to be used as conditioning
      inputs.
    input_shape: The shape of the conditioning inputs, excluding the batch
      dimension. The expected shape is (*spatial_dims, num_variables).
    output_dataset: Path to Zarr dataset from which targets/observations are
      sampled. It must be aligned in time with the `input_dataset`.
    output_stats: Path to Zarr dataset containing statistics of the output data,
      used to normalize targets/observations.
    output_variables: The names of the variables to be used as targets/
      observations.
    output_shape: The shape of the observations, excluding the batch dimension.
      The expected shape is (*spatial_dims, num_variables).
    static_input_dataset: Path to Zarr dataset containing static conditioning
      inputs.
    static_input_variables: The names of the variables to be used as static
      conditioning inputs.
    static_input_shape: The shape of the static conditioning inputs, excluding
      the batch dimension. The expected shape is (*spatial_dims, num_variables).
    crop_input: Whether to crop the input data to the shape of the output data.
    normalization: Whether to normalize the input data using local or global
      statistics.
    forcing_dataset: Path to a dataset containing forcing scalar inputs, such as
      the atmospheric CO2 concentration. Defaults to None.
    use_temporal_inputs: Whether to use temporal information as input to models.
      Defaults to False.
  """

  input_dataset: str
  input_stats: str
  input_variables: list[str]
  input_shape: tuple[int, ...]
  output_dataset: str
  output_stats: str
  output_variables: list[str]
  output_shape: tuple[int, ...]
  static_input_dataset: str
  static_input_variables: list[str]
  static_input_shape: tuple[int, ...]
  crop_input: bool = False
  normalization: Literal["local", "global"] = "local"
  forcing_dataset: str | None = None
  use_temporal_inputs: bool = False
