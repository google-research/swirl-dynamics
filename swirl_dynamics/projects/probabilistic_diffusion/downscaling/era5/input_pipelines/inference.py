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

"""Inference data pipeline."""

from collections.abc import Sequence
from typing import Protocol

from etils import epath
import jax
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils
import xarray as xr

DatasetVariables = utils.DatasetVariables


class CondLoader(Protocol):
  """Loads daily mean data as condition for inference."""

  def get_days(self, days: np.ndarray) -> dict[str, np.ndarray]:
    """Retrieves the data record for a given starting day."""
    ...


class DefaultCondLoader:
  """Loads daily mean data as condition for inference.

  Selects variables from the dataset and normalizes them accordingly.
  """

  def __init__(
      self,
      date_range: tuple[str, str] | None,
      dataset: epath.PathLike,
      dataset_variables: DatasetVariables,
      stats: epath.PathLike,
      stats_variables: DatasetVariables | None = None,
      dims_order: Sequence[str] = ("time", "longitude", "latitude", "level"),
  ):
    if stats_variables is None:
      stats_variables = dataset_variables

    ds = xr.open_zarr(dataset)
    if date_range is not None:
      date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)
      ds = ds.sel(time=slice(*date_range))

    ds = ds.reindex(
        latitude=np.sort(ds.latitude), longitude=np.sort(ds.longitude)
    )
    ds = ds.transpose(*dims_order)

    self._data_arrays = {}
    for v in dataset_variables:
      self._data_arrays[v.rename] = ds[v.name].sel(v.indexers, drop=True)

    self.mean = utils.read_stats(stats, stats_variables, "mean")
    self.std = utils.read_stats(stats, stats_variables, "std")

  def get_days(self, days: np.ndarray) -> dict[str, np.ndarray]:
    """Retrieves the data record for a given starting day."""
    item = []
    for v, da in self._data_arrays.items():
      array = da.sel(time=days).to_numpy()
      assert array.ndim == 3 or array.ndim == 4
      array = np.expand_dims(array, axis=-1) if array.ndim == 3 else array

      # transforms
      array = (array - self.mean[v]) / self.std[v]
      array = np.rot90(array, axes=(1, 2))
      item.append(array)

    # add prefix
    all_cond = np.concatenate(item, axis=-1)

    return {"channel:daily_mean": all_cond}


class CondLoaderFromDebiasedOutput:
  """Loads daily mean data as condition for inference from debiased output.

  The source is expected to be in Zarr with time coordinates in numpy datetime
  format. Variables (temperature, wind speed etc.) are concatenated in the
  channel dimension. Zarr data variables denote the data type (i.e. era5, lens2,
  debiased) - only one of them will be loaded for inference.
  """

  def __init__(
      self,
      date_range: tuple[str, str],
      zarr_path: epath.PathLike,
      field: str,
      mean: np.ndarray,  # Global - (lon, lat, variables)
      std: np.ndarray,  # Global - (lon, lat, variables)
  ):
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)
    self.ds = xr.open_zarr(zarr_path)[field].sel(time=slice(*date_range))
    self.mean = mean
    self.std = std

  def get_days(self, days: np.ndarray) -> dict[str, np.ndarray]:
    """Retrieves the data record for a given starting day."""
    all_cond = (self.ds.sel(time=days).to_numpy() - self.mean) / self.std
    all_cond = np.rot90(all_cond, axes=(1, 2))
    return {"channel:daily_mean": all_cond}
