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

"""Modules for sampling and packaging inference results."""

import os
import jax
import numpy as np
import pandas as pd
import xarray as xr


def samples_to_dataset(
    samples: jax.Array,
    *,
    field_names: list[str],
    times: pd.DatetimeIndex,
    spatial_dims: tuple[str, str] = ('south_north', 'west_east'),
    spatial_coords: dict[str, xr.DataArray] | None = None,
    sample_dim: str = 'sample',
) -> xr.Dataset:
  """Packages inference output samples as an xarray.Dataset.

  Args:
    samples: An inference array with dimensions [`time`, `sample`,
      *spatial_dims, `field`].
    field_names: A list of variable names mapping to the `field` dimension of
      the `samples` array.
    times: The time stamps of the samples.
    spatial_dims: The name of the spatial dimensions of the samples.
    spatial_coords: The spatial coordinates of the samples.
    sample_dim: The name of the sample dimension of the samples.

  Returns:
    The inference results as an xarray.Dataset with dimensions [`time`,
    `sample`, *spatial_dims], and data variables for each field.
  """
  dims = ['time', sample_dim, *spatial_dims]
  num_samples, n_south_north, n_west_east = samples.shape[1:4]
  if spatial_coords is None:
    spatial_coords = {
        spatial_dims[0]: ([spatial_dims[0]], range(n_south_north)),
        spatial_dims[1]: ([spatial_dims[1]], range(n_west_east)),
    }
  data_vars = {}
  for i, field_name in enumerate(field_names):
    data_vars[field_name] = (dims, samples[..., i])

  ds = xr.Dataset(
      data_vars=data_vars,
      coords={
          'time': (['time'], times),
          sample_dim: ([sample_dim], range(num_samples)),
          **spatial_coords,
      },
      attrs=dict(description=f'Downscaled {num_samples}-member ensembles.'),
  )
  return ds


def batch_to_dataset(
    batch: jax.Array,
    *,
    field_names: list[str],
    times: pd.DatetimeIndex,
    spatial_dims: tuple[str, str] = ('south_north', 'west_east'),
    spatial_coords: dict[str, xr.DataArray] | None = None,
) -> xr.Dataset:
  """Packages batch inputs or targets as an xarray.Dataset.

  Args:
    batch: An array of weather data with dimensions [`time`, *spatial_dims,
      `field`].
    field_names: A list of variable names mapping to the `field` dimension of
      the `inference` array.
    times: The time stamps of the data snapshots.
    spatial_dims: The name of the spatial dimensions of the data.
    spatial_coords: The spatial coordinates of the data.

  Returns:
    The results as an xarray.Dataset with dimensions [`time`, *spatial_dims],
    and data variables for each field.
  """
  dims = ['time', *spatial_dims]
  n_south_north, n_west_east = batch.shape[1:3]
  if spatial_coords is None:
    spatial_coords = {
        spatial_dims[0]: ([spatial_dims[0]], range(n_south_north)),
        spatial_dims[1]: ([spatial_dims[1]], range(n_west_east)),
    }

  data_vars = {}
  for i, field_name in enumerate(field_names):
    data_vars[field_name] = (dims, batch[..., i])

  ds = xr.Dataset(
      data_vars=data_vars,
      coords={
          'time': (['time'], times),
          **spatial_coords,
      },
  )
  return ds


def concat_to_zarr(
    ds_list: list[xr.Dataset],
    out_path: str,
    basename: str,
    append: bool = True,
    append_dim: str | None = None,
):
  """Concatenates a list of xarray.Dataset and writes to zarr.

  Args:
    ds_list: List of datasets to concatenate.
    out_path: The directory where the output is written.
    basename: The name of the Zarr file to be written to.
    append: Whether to try appending to an existing dataset. Write otherwise.
    append_dim: Appending dimension.
  """
  kwargs = {'mode': 'a', 'append_dim': append_dim} if append else {'mode': 'w'}
  ds = xr.concat(ds_list, dim=append_dim)
  ds = xr.apply_ufunc(np.asarray, ds.load())
  ds.to_zarr(os.path.join(out_path, basename + '.zarr'), **kwargs)
