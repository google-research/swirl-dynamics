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

"""Pipeline utils."""

from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

from etils import epath
import numpy as np
import xarray


class DatasetVariable(NamedTuple):
  """Metadata for a dataset variable.

  Used for selecting a variable (possibly sliced) from a raw Xarray dataset and
  rename it in a new dataset.

  Attributes:
    name: The name of the variable as it appears in the raw dataset.
    indexers: The indexers for the variable (e.g. selecting a pressure level
      and/or a local region), which will be used through Xarray's `.sel`
      interface.
    rename: The new name of the selected variable in the output dataset.
  """

  name: str
  indexers: dict[str, Any] | None
  rename: str


DatasetVariables = Sequence[DatasetVariable]


def read_stats(
    dataset: epath.PathLike,
    variables: DatasetVariables,
    field: Literal["mean", "std"],
) -> dict[str, np.ndarray]:
  """Reads variables from a zarr dataset and returns as a dict of ndarrays.

  Args:
    dataset: The dataset path to read stats from.
    variables: The variables to read stats for.
    field: The stat field to read, one of ['mean', 'std'].

  Returns:
    A dictionary of numpy arrays representing name -> corresponding stat field.
  """
  ds = xarray.open_zarr(dataset)
  out = {}
  for variable in variables:
    indexers = (
        variable.indexers | {"stats": field}
        if variable.indexers
        else {"stats": field}
    )
    stats = ds[variable.name].sel(indexers).to_numpy()
    assert (
        stats.ndim == 2 or stats.ndim == 3
    ), f"Expected 2 or 3 dimensions for {variable}, got {stats.ndim}."
    stats = np.expand_dims(stats, axis=-1) if stats.ndim == 2 else stats
    out[variable.rename] = stats
  return out


def read_stats_and_concat(
    dataset: epath.PathLike,
    variables: DatasetVariables,
    field: Literal["mean", "std"],
) -> np.ndarray:
  """Reads stats and concat along the channel dimension."""
  stats = read_stats(dataset, variables, field)
  return np.concatenate(list(stats.values()), axis=-1)


def select_variables(
    dataset: epath.PathLike | xarray.Dataset,
    variables: DatasetVariables,
) -> xarray.Dataset:
  """Selects variables from a zarr dataset."""
  if not isinstance(dataset, xarray.Dataset):
    dataset = xarray.open_zarr(dataset)
  out = {}
  for v in variables:
    var = dataset[v.name].sel(v.indexers)
    # Assume that selected variables are always surface ones or on a single
    # vertical level.
    if "level" in var.dims:
      var = var.squeeze(dim="level", drop=True)
    var = var.drop_vars("level", errors="ignore")
    out[v.rename] = var
  return xarray.Dataset(out)


def add_indexers(
    indexers: dict[str, Any], variables: DatasetVariables
) -> list[DatasetVariable]:
  """Adds a set of common indexers to all variables."""
  new_variables = []
  for v in variables:
    new_indexers = indexers if v.indexers is None else indexers | v.indexers
    new_variables.append(
        DatasetVariable(name=v.name, indexers=new_indexers, rename=v.rename)
    )
  return new_variables


def add_member_indexer(
    members: str | Sequence[str], variables: DatasetVariables
) -> list[DatasetVariable]:
  """Selects a subset of members from a list of variables."""
  return add_indexers({"member": members}, variables)


# **********************
# Region configs
# **********************


class Region(NamedTuple):
  """Region configs."""

  lon_range: tuple[float, float]
  lat_range: tuple[float, float]
  cond_shape: tuple[int, int]
  sample_shape: tuple[int, int]
  sample_resize: tuple[int, int] | None
  time_downsample: int
  num_days_per_example: int
  spatial_downsample: tuple[int, ...]
  channels: tuple[int, ...]
  spatial_attn: tuple[bool, ...]
  temporal_attn: tuple[bool, ...]


def regionalize(
    region: Region, variables: DatasetVariables
) -> list[DatasetVariable]:
  """Adds regional indexers to variables."""
  lonlat_indexers = {
      "longitude": slice(*region.lon_range),
      "latitude": slice(*region.lat_range),
  }
  return add_indexers(lonlat_indexers, variables)


def get_coord_as_nparray(
    dataset: epath.PathLike, dim: str, region: Region | None = None
) -> np.ndarray:
  dataset = xarray.open_zarr(dataset)
  if region is not None:
    dataset = dataset.sel(
        longitude=slice(*region.lon_range),
        latitude=slice(*region.lat_range),
        time=slice(None, None, region.time_downsample),
    )
  return dataset[dim].to_numpy()
