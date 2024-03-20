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

"""Utility functions for Zarr file reading and writing."""

from collections.abc import Mapping
import os
from typing import Any

from etils import epath
import xarray as xr

filesys = epath.backend.tf_backend


def collected_metrics_to_ds(
    data: Mapping[str, Any],
    append_dim: str,
    append_slice: slice,
    coords: xr.core.coordinates.DatasetCoordinates | None = None,
) -> xr.Dataset:
  """Packages collected metrics as an xarray.Dataset.

  Args:
    data: A mapping of metric names to their collected values.
    append_dim: The name of the axis dimension of metric collection, enforced to
      allow downstream dataset appending.
    append_slice: Current index slice in the `append_dim` axis.
    coords: xarray coordinates of the label dataset used to compute the metrics.

  Returns:
    A dataset containing all collected metrics as variables, with coordinate
    metadata.
  """
  fixed_shape = next(iter(data.values())).shape[1:]
  coord_dict = None
  dims = [append_dim]
  if coords is not None:
    for cur_size in fixed_shape:
      dims.append(
          list(coords.dims.keys())[list(coords.dims.values()).index(cur_size)]
      )

    coord_dict = {
        elem: coords[elem].data for elem in dims if elem != append_dim
    }
    coord_dict[append_dim] = coords[append_dim].data[append_slice]

  data_vars = {}
  for key, value in data.items():
    if coords is None:
      dims.extend([f'dim_{i}' for i in range(value.ndim - 1)])
    data_vars[key] = (dims, value)

  return xr.Dataset(
      data_vars=data_vars,
      coords=coord_dict,
      attrs=dict(description='Collected local metrics.'),
  )


def collected_metrics_to_zarr(
    data: Mapping[str, Any],
    *,
    out_dir: epath.PathLike,
    basename: str,
    append_dim: str,
    coords: xr.core.coordinates.DatasetCoordinates | None = None,
    append_slice: slice,
) -> None:
  """Writes collected metrics to zarr."""
  ds = collected_metrics_to_ds(
      data,
      append_dim,
      append_slice,
      coords,
  )
  write_to_file(ds, out_dir, basename, append_dim)


def write_to_file(
    ds, out_dir: epath.PathLike, basename: str, append_dim: str | None = None
) -> None:
  """Writes an xarray.Dataset to zarr or appends to an existing zarr file."""
  out_path = os.path.join(out_dir, basename + '.zarr')
  if filesys.exists(out_path) and append_dim is not None:
    kwargs = {'mode': 'a', 'append_dim': append_dim}
  else:
    kwargs = {'mode': 'w'}
  ds.to_zarr(out_path, **kwargs)
