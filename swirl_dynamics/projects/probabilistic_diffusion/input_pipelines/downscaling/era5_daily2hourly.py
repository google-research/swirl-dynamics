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

"""Input pipeline that yields paired high-res hourly/low-res daily ERA5 data."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Literal, SupportsIndex

from absl import logging
from etils import epath
import grain.python as pygrain
import jax
import numpy as np
import xarray_tensorstore as xrts

DatasetVariables = Mapping[str, Mapping[str, Any] | None]
FlatFeatures = dict[str, Any]

DELTA_1D = np.timedelta64(1, "D")
DIMS_ORDER = ("time", "longitude", "latitude", "level")


@dataclasses.dataclass(frozen=True)
class Standardize(pygrain.MapTransform):
  """Standardize variables pixel-wise using pre-computed mean and std."""

  input_fields: Sequence[str]
  mean: Mapping[str, np.ndarray]
  std: Mapping[str, np.ndarray]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = (features[field] - self.mean[field]) / self.std[field]
    return features


@dataclasses.dataclass(frozen=True)
class Concatenate(pygrain.MapTransform):
  """Creates a new field by concatenating selected fields."""

  input_fields: Sequence[str]
  output_field: str
  axis: int
  remove_inputs: bool = True

  def map(self, features: FlatFeatures) -> FlatFeatures:
    arrays = [features[f] for f in self.input_fields]
    features[self.output_field] = np.concatenate(arrays, axis=self.axis)
    if self.remove_inputs:
      for f in self.input_fields:
        del features[f]
    return features


@dataclasses.dataclass(frozen=True)
class Rot90(pygrain.MapTransform):
  """Rotate selected field 90 degrees counterclockwise."""

  input_fields: Sequence[str]
  k: int = 1
  axes: tuple[int, int] = (0, 1)

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = np.rot90(features[field], k=self.k, axes=self.axes)
    return features


@dataclasses.dataclass(frozen=True)
class RandomMaskout(pygrain.RandomMapTransform):
  """Randomly mask out selected fields with a given value."""

  input_fields: Sequence[str]
  probability: float = 0.1
  fill_value: float = 0.0

  def random_map(
      self, features: FlatFeatures, rng: np.random.Generator
  ) -> FlatFeatures:
    if rng.choice([True, False], p=[self.probability, 1 - self.probability]):
      for field in self.input_fields:
        features[field] = self.fill_value * np.ones_like(features[field])
    return features


@dataclasses.dataclass(frozen=True)
class AssembleCondDict(pygrain.MapTransform):
  """Assemble fields into a conditional dictionary."""

  cond_fields: Sequence[str]
  prefix: str = "channel:"
  remove_original: bool = True

  def map(self, features: FlatFeatures) -> FlatFeatures:
    if "cond" not in features:
      features["cond"] = {}
    for field in self.cond_fields:
      features["cond"][f"{self.prefix}{field}"] = features[field]
      if self.remove_original:
        del features[field]
    return features


class HourlyDailyPair:
  """Loads paired (in time) hourly and daily ERA5 data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      hourly_dataset: epath.PathLike,
      hourly_variables: DatasetVariables,
      hourly_downsample: int = 1,
      daily_dataset: epath.PathLike | None = None,
      daily_variables: DatasetVariables | None = None,
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
    date_range = jax.tree_map(lambda x: np.datetime64(x, "D"), date_range)

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

    for v, indexers in hourly_variables.items():
      self._hourly_arrays[v] = hourly_ds[v].sel(indexers)

    if daily_dataset:
      daily_ds = xrts.open_zarr(daily_dataset).sel(time=slice(*date_range))
      daily_ds = daily_ds.reindex(
          latitude=np.sort(daily_ds.latitude),
          longitude=np.sort(daily_ds.longitude),
      )
      daily_ds = daily_ds.transpose(*DIMS_ORDER)

      daily_variables = daily_variables or {}
      for v, indexers in daily_variables.items():
        self._daily_arrays[f"{v}"] = daily_ds[v].sel(indexers)

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
      item[f"hourly_{v}"] = array[:-1]

    for v, da in self._daily_arrays.items():
      array = xrts.read(da.sel(time=day_slice)).data
      assert array.ndim == 3 or array.ndim == 4
      array = np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      item[f"daily_{v}"] = np.repeat(array[:-1], self._replicate_daily, axis=0)

    return item


def read_stats(
    dataset: epath.PathLike,
    variables: DatasetVariables,
    field: Literal["mean", "std"],
    add_prefix: str = "",
) -> dict[str, np.ndarray]:
  """Reads variables from a zarr dataset and returns as a dict of ndarrays."""
  ds = xrts.open_zarr(dataset)
  out = {}
  for var, indexers in variables.items():
    indexers = indexers | {"stats": field} if indexers else {"stats": field}
    stats = ds[var].sel(indexers).to_numpy()
    assert stats.ndim == 2 or stats.ndim == 3
    stats = np.expand_dims(stats, axis=-1) if stats.ndim == 2 else stats
    out[f"{add_prefix}{var}"] = stats
  return out


def create_dataset(
    date_range: tuple[str, str],
    hourly_dataset: epath.PathLike,
    hourly_variables: DatasetVariables,
    hourly_downsample: int,
    daily_dataset: epath.PathLike,
    daily_variables: DatasetVariables,
    num_days_per_example: int,
    hourly_stats: epath.PathLike | None = None,
    daily_stats: epath.PathLike | None = None,
    sample_field_rename: str = "x",
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    cond_maskout_prob: float = 0.0,
) -> pygrain.DataLoader:
  """Creates a grain dataloader with paired hourly and daily data."""
  source = HourlyDailyPair(
      date_range=date_range,
      hourly_dataset=hourly_dataset,
      hourly_variables=hourly_variables,
      hourly_downsample=hourly_downsample,
      daily_dataset=daily_dataset,
      daily_variables=daily_variables,
      num_days_per_example=num_days_per_example,
      replicate_daily=True,
  )

  standardizations = []
  hourly_vars_list = ["hourly_" + v for v in hourly_variables.keys()]
  if hourly_stats:
    standardizations += [
        Standardize(
            input_fields=hourly_vars_list,
            mean=read_stats(
                hourly_stats, hourly_variables, "mean", add_prefix="hourly_"
            ),
            std=read_stats(
                hourly_stats, hourly_variables, "std", add_prefix="hourly_"
            ),
        ),
    ]
  # Standardization for condition.
  daily_vars_list = ["daily_" + v for v in daily_variables.keys()]
  if daily_stats:
    standardizations += [
        Standardize(
            input_fields=daily_vars_list,
            mean=read_stats(
                daily_stats, daily_variables, "mean", add_prefix="daily_"
            ),
            std=read_stats(
                daily_stats, daily_variables, "std", add_prefix="daily_"
            ),
        ),
    ]

  all_variables = hourly_vars_list + daily_vars_list
  transformations = standardizations + [
      Rot90(input_fields=all_variables, k=1, axes=(1, 2)),
      Concatenate(
          input_fields=hourly_vars_list,
          output_field=sample_field_rename,
          axis=-1,
      ),
      Concatenate(
          input_fields=daily_vars_list, output_field="daily_mean", axis=-1
      ),
      RandomMaskout(
          input_fields=("daily_mean",), probability=cond_maskout_prob
      ),
      AssembleCondDict(cond_fields=("daily_mean",), prefix="channel:"),
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
  )
  return data_loader
