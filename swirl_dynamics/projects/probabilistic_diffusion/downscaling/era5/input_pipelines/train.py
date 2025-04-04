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

"""Training data pipeline."""

from typing import SupportsIndex

from absl import logging
from etils import epath
import grain.python as pygrain
import jax
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import transforms
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils
import xarray_tensorstore as xrts

DatasetVariables = utils.DatasetVariables

DELTA_1D = np.timedelta64(1, "D")
DIMS_ORDER = ("time", "longitude", "latitude", "level")


class HourlyDailyPair:
  """Loads paired (in time) hourly and daily ERA5 data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      hourly_dataset: epath.PathLike,
      hourly_variables: DatasetVariables,
      hourly_downsample: int,
      daily_dataset: epath.PathLike,
      daily_variables: DatasetVariables,
      replicate_daily: bool = True,
      num_days_per_example: int = 1,
      max_retries: int = 3,
      retry_seed: int = 9999,
  ):
    """Data source constructor.

    Args:
      date_range: The date range applied. Data not within this range will be
        ignored.
      hourly_dataset: The path of a zarr dataset containing the hourly samples.
      hourly_variables: A dictionary where keys are variable names and values
        are indexers used for the hourly dataset's `.sel()` method. Example:
        `{'geopotential': {'level': [1000]}}`.
      hourly_downsample: The downsampling factor in time (applied uniformly).
        For example, setting this value to 24 for an hourly dataset results in
        effectively loading daily data.
      daily_dataset: (optional) The path of the zarr dataset containing the
        daily aggregated information.
      daily_variables: Same as `hourly_variables` but for the daily dataset.
      replicate_daily: If `True`, replicates daily data along the time axis to
        match the hourly data length.
      num_days_per_example: The number of consecutive days each loaded example
        contains.
      max_retries: The maximum number of attempts made to retrieve an item with
        a randomly selected key if the original key fails.
      retry_seed: The random seed used for retrying.
    """
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    hourly_ds = xrts.open_zarr(hourly_dataset).sel(
        time=slice(*date_range, hourly_downsample)
    )
    # Reindex and transpose the coordinates to accommodate zarr datasets written
    # in different ways. Examples yielded from this data source always have
    # increasing lat/lon coordinates and axes following the same order.
    hourly_ds = hourly_ds.reindex(
        latitude=np.sort(hourly_ds.latitude),
        longitude=np.sort(hourly_ds.longitude),
    )
    hourly_ds = hourly_ds.transpose(*DIMS_ORDER)

    self._hourly_arrays = {}
    self._daily_arrays = {}

    for v in hourly_variables:
      self._hourly_arrays[v.rename] = hourly_ds[v.name].sel(v.indexers)

    daily_ds = xrts.open_zarr(daily_dataset).sel(time=slice(*date_range))
    daily_ds = daily_ds.reindex(
        latitude=np.sort(daily_ds.latitude),
        longitude=np.sort(daily_ds.longitude),
    )
    daily_ds = daily_ds.transpose(*DIMS_ORDER)

    daily_variables = daily_variables or {}
    for v in daily_variables:
      self._daily_arrays[v.rename] = daily_ds[v.name].sel(v.indexers)

    self._date_range = date_range
    self._num_days_per_example = num_days_per_example
    self._len = (date_range[1] - date_range[0]) // DELTA_1D - (
        self._num_days_per_example - 1
    )
    self._replicate_daily = 24 // hourly_downsample if replicate_daily else 1
    self._max_retries = max_retries
    self._retry_seed = retry_seed

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, np.ndarray]:
    """Returns the data record for a given key."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    day = self._date_range[0] + idx * DELTA_1D
    for _ in range(self._max_retries):
      # Fetching may fail for specific days (a very tiny fraction). When this
      # happens, we retry fetching a different one pseudo-randomly chosen based
      # on the failed index (up to a maximum number of retries).
      try:
        return self.get(day)

      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Error loading record for day %s. Error details: %s", str(day), e
        )
        rng = np.random.default_rng(self._retry_seed + idx)
        idx = rng.integers(0, self._len)
        day = self._date_range[0] + idx * DELTA_1D

    raise KeyError(
        f"Failed to retrieve a record after {self._max_retries} retries."
    )

  def get(self, day: np.datetime64) -> dict[str, np.ndarray]:
    """Retrieves the data record for a given starting day."""
    day_slice = slice(
        np.datetime64(day, "m"),
        np.datetime64(day, "m") + DELTA_1D * self._num_days_per_example,
    )
    item = {}
    for v, da in self._hourly_arrays.items():
      array = xrts.read(da.sel(time=day_slice)).data
      assert array.ndim == 3 or array.ndim == 4
      array = np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      item[v] = array[:-1]

    for v, da in self._daily_arrays.items():
      array = xrts.read(da.sel(time=day_slice)).data
      assert array.ndim == 3 or array.ndim == 4
      array = np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      item[v] = np.repeat(array[:-1], self._replicate_daily, axis=0)

    return item


def create_dataloader(
    date_range: tuple[str, str],
    hourly_dataset_path: epath.PathLike,
    hourly_variables: DatasetVariables,
    hourly_downsample: int,
    daily_dataset_path: epath.PathLike,
    daily_variables: DatasetVariables,
    num_days_per_example: int = 2,
    hourly_stats_path: epath.PathLike | None = None,
    daily_stats_path: epath.PathLike | None = None,
    sample_field_rename: str = "x",
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    cond_maskout_prob: float = 0.0,
    read_options: pygrain.ReadOptions | None = None,
) -> pygrain.DataLoader:
  """Creates a grain dataloader with paired hourly and daily data."""
  source = HourlyDailyPair(
      date_range=date_range,
      hourly_dataset=hourly_dataset_path,
      hourly_variables=hourly_variables,
      hourly_downsample=hourly_downsample,
      daily_dataset=daily_dataset_path,
      daily_variables=daily_variables,
      num_days_per_example=num_days_per_example,
      replicate_daily=True,
  )

  standardizations = []
  hourly_var_renames = [v.rename for v in hourly_variables]
  if hourly_stats_path:
    standardizations += [
        transforms.Standardize(
            input_fields=hourly_var_renames,
            mean=utils.read_stats(hourly_stats_path, hourly_variables, "mean"),
            std=utils.read_stats(hourly_stats_path, hourly_variables, "std"),
        ),
    ]
  # Standardization for condition.
  daily_var_renames = [v.rename for v in daily_variables]
  if daily_stats_path:
    standardizations += [
        transforms.Standardize(
            input_fields=daily_var_renames,
            mean=utils.read_stats(daily_stats_path, daily_variables, "mean"),
            std=utils.read_stats(daily_stats_path, daily_variables, "std"),
        ),
    ]

  all_variables = hourly_var_renames + daily_var_renames
  transformations = standardizations + [
      # Data is lon-lat and model takes lat-lon.
      transforms.Rot90(input_fields=all_variables, k=1, axes=(1, 2)),
      transforms.Concatenate(
          input_fields=hourly_var_renames,
          output_field=sample_field_rename,
          axis=-1,
      ),
      transforms.Concatenate(
          input_fields=daily_var_renames, output_field="daily_mean", axis=-1
      ),
      transforms.RandomMaskout(
          input_fields=("daily_mean",), probability=cond_maskout_prob
      ),
      transforms.AssembleCondDict(
          cond_fields=("daily_mean",), prefix="channel:"
      ),
  ]
  data_loader = pygrain.load(
      source=source,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
      read_options=read_options,
  )
  return data_loader
