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

"""Utils for beam pipelines."""

import typing

import xarray as xr


def get_climatology_mean(
    climatology: xr.Dataset, variables: list[str], **sel_kwargs
) -> xr.Dataset:
  """Returns the climatological mean of the given variables.

  The climatology dataset is assumed to have been produced through
  the weatherbench2 compute_climatology.py script,
  (https://github.com/google-research/weatherbench2/blob/main/scripts/compute_climatology.py)
  and statistics `mean`, and `std`. The convention is that the climatological
  means do not have a suffix, and standard deviations have a `_std` suffix.

  Args:
    climatology: The climatology dataset.
    variables: The variables to extract from the climatology.
    **sel_kwargs: Additional selection criteria for the variables.

  Returns:
    The climatological mean of the given variables.
  """
  climatology_mean = climatology[variables]
  return typing.cast(xr.Dataset, climatology_mean.sel(**sel_kwargs).compute())


def get_climatology_std(
    climatology: xr.Dataset, variables: list[str], **sel_kwargs
) -> xr.Dataset:
  """Returns the climatological standard deviation of the given variables.

  The climatology dataset is assumed to have been produced through
  the weatherbench2 compute_climatology.py script, and statistics
  `mean`, and `std`. The convention is that the climatological means do not
  have a suffix, and standard deviations have a `_std` suffix.

  Args:
    climatology: The climatology dataset.
    variables: The variables to extract from the climatology.
    **sel_kwargs: Additional selection criteria for the variables.

  Returns:
    The climatological standard deviation of the given variables.
  """
  clim_std_dict = {key + '_std': key for key in variables}  # pytype: disable=unsupported-operands
  climatology_std = climatology[list(clim_std_dict.keys())].rename(
      clim_std_dict
  )
  return typing.cast(xr.Dataset, climatology_std.sel(**sel_kwargs).compute())
