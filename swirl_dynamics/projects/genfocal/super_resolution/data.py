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

"""Data modules."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Literal, NamedTuple, SupportsIndex, TypeAlias

from absl import logging
from etils import epath
import grain.python as pygrain
import jax
import numpy as np
import xarray as xr
import xarray_tensorstore as xrts

FlatFeatures: TypeAlias = dict[str, Any]

DELTA_1D = np.timedelta64(1, "D")
DIMS_ORDER = ("time", "longitude", "latitude", "level")


class DatasetVariable(NamedTuple):
  """Metadata for a modeled variable in an Xarray dataset.

  This metadata is used for selecting a variable, optionally indexed, from a raw
  Xarray dataset and rename it in a new dataset.

  Attributes:
    name: The name of the variable as it appears in the raw dataset.
    indexers: The optional indexers for the variable (e.g. selecting a pressure
      level and/or a local region), used through Xarray's `.sel` interface.
    rename: The new name of the selected variable in the output dataset.
  """

  name: str
  indexers: dict[str, Any] | None
  rename: str


DatasetVariables = Sequence[DatasetVariable]


def select_variables(
    dataset: epath.PathLike | xr.Dataset, variables: DatasetVariables
) -> xr.Dataset:
  """Selects a subset of variables from a zarr dataset."""
  if not isinstance(dataset, xr.Dataset):
    dataset = xr.open_zarr(dataset)
  out = {}
  for v in variables:
    var = dataset[v.name].sel(v.indexers)
    # Assume that selected variables are always surface ones or on a single
    # vertical level.
    if "level" in var.dims:
      var = var.squeeze(dim="level", drop=True)
    var = var.drop_vars("level", errors="ignore")
    out[v.rename] = var
  return xr.Dataset(out)


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


class DataSource:
  """Loads time-paired hourly and daily ERA5 data.

  This class defines indexed access to a super-resolution dataset, consisting of
  an hourly- and a daily-resolution source. All loaded examples are paired in
  time, i.e. the hourly and daily fields are aligned to cover the same dates.
  The daily data is repeated to match the temporal dimension of the hourly data.
  """

  def __init__(
      self,
      date_range: tuple[str, str],
      hourly_dataset_path: epath.PathLike,
      hourly_variables: DatasetVariables,
      hourly_downsample: int,
      daily_dataset_path: epath.PathLike,
      daily_variables: DatasetVariables,
      num_days_per_example: int = 1,
      max_retries: int = 3,
      retry_seed: int = 9999,
  ):
    """Data source constructor.

    Args:
      date_range: A tuple of strings `(start_date, end_date)` defining the
        inclusive date range for data loading (e.g., '2020-01-01',
        '2020-01-31'). Data outside this range will not be loaded.
      hourly_dataset_path: The path of a Zarr dataset containing the hourly
        samples. The dataset must have the following dimensions: ['time',
        'longitude', 'latitude', 'level'].
      hourly_variables: A collection of `DatasetVariable` objects specifying
        which variables to load from the hourly dataset and how to index select
        and rename them.
      hourly_downsample: The temporal downsampling ratio applied to the hourly
        data. For example, a value of 24 will load data at daily resolution.
        Must be a divisor of 24.
      daily_dataset_path: The path of the Zarr dataset containing the daily
        data. The dataset must have the following dimensions: ['time',
        'longitude', 'latitude', 'level'].
      daily_variables: A collection of `DatasetVariable` objects specifying
        which variables to load from the daily dataset and how to index select
        and rename them.
      num_days_per_example: The number of consecutive days that each yielded
        example will cover.
      max_retries: The maximum number of attempts to retrieve an item if the
        initial attempt fails due to an error. A different, randomly selected
        key will be used for each retry.
      retry_seed: The random seed used for generating new keys during retry
        attempts.
    """
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    hourly_ds = xrts.open_zarr(hourly_dataset_path).sel(
        time=slice(*date_range, hourly_downsample)
    )
    # Reindex and transpose the coordinates to accommodate Zarr datasets written
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

    daily_ds = xrts.open_zarr(daily_dataset_path).sel(time=slice(*date_range))
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
    self._snapshots_per_day = 24 // hourly_downsample
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
      item[v] = np.repeat(array[:-1], self._snapshots_per_day, axis=0)

    return item


@dataclasses.dataclass(frozen=True)
class Standardize(pygrain.MapTransform):
  """Standardize fields using pre-computed mean and standard deviation.

  This transform makes no assumption about the field and stats shapes, as long
  as they are compatible with numpy broadcasting rules.
  """

  input_fields: Sequence[str]
  mean: Mapping[str, np.ndarray]
  std: Mapping[str, np.ndarray]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = (features[field] - self.mean[field]) / self.std[field]
    return features


@dataclasses.dataclass(frozen=True)
class Concatenate(pygrain.MapTransform):
  """Concatenates selected fields into a new one."""

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
  """Rotate fields counterclockwise by 90 degrees."""

  input_fields: Sequence[str]
  k: int = 1
  axes: tuple[int, int] = (0, 1)

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = np.rot90(features[field], k=self.k, axes=self.axes)
    return features


@dataclasses.dataclass(frozen=True)
class RandomMaskout(pygrain.RandomMapTransform):
  """Randomly replace all values of selected fields with a given value."""

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
class NestCondDict(pygrain.MapTransform):
  """Nests selected fields under a 'cond' key."""

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


def read_stats(
    dataset_path: epath.PathLike,
    variables: DatasetVariables,
    field: Literal["mean", "std"],
) -> dict[str, np.ndarray]:
  """Reads variable statistics stored in a Zarr dataset.

  Args:
    dataset_path: The dataset path (.zarr) to read statistics from.
    variables: The variables to read statistics for.
    field: The statistic field to read, one of ['mean', 'std'].

  Returns:
    A dictionary {variable: statistic}.
  """
  ds = xr.open_zarr(dataset_path)
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
    worker_count: int | None = 0,
    cond_maskout_prob: float = 0.0,
    read_options: pygrain.ReadOptions | None = None,
) -> pygrain.DataLoader:
  """Creates a dataloader with paired hourly and daily data.

  Optionally standardizes the input and output, and applies a sequence of
  transformations including
    - Rotate the data counterclockwise by 90 degrees.
    - Stack the variables along a trailing channel dimension.
    - Nest the inputs inside a `cond` dictionary.
    - Randomly mask out the input values with zeros (for training
    classifier-free guidance).

  Args:
    date_range: See `DataSource`.
    hourly_dataset_path: See `DataSource`.
    hourly_variables: See `DataSource`.
    hourly_downsample: See `DataSource`.
    daily_dataset_path: See `DataSource`.
    daily_variables: See `DataSource`.
    num_days_per_example: See `DataSource`.
    hourly_stats_path: The dataset path (.zarr) containing the hourly variable
      statistics.
    daily_stats_path: The dataset path (.zarr) containing the daily variable
      statistics.
    sample_field_rename: The name of field aggregating all modeled variables
      (concatenated along a channel dimension).
    num_epochs: The number of epochs to cycle through the dataset. If `None`,
      the dataset will be cycled through indefinitely.
    shuffle: If `True`, the dataset will be reshuffled at every epoch.
    seed: The random seed for shuffling loading order.
    batch_size: The batch size for the yielded examples.
    worker_count: Number of child processes launched to parallelize the
      transformations among. Zero means processing runs in the same process.
    cond_maskout_prob: The probability of randomly masking out condition data.
      Required for classifier-free guidance.
    read_options: Pygrain read options.

  Returns:
    A pygrain DataLoader object.
  """
  source = DataSource(
      date_range=date_range,
      hourly_dataset_path=hourly_dataset_path,
      hourly_variables=hourly_variables,
      hourly_downsample=hourly_downsample,
      daily_dataset_path=daily_dataset_path,
      daily_variables=daily_variables,
      num_days_per_example=num_days_per_example,
  )

  standardizations = []
  hourly_var_renames = [v.rename for v in hourly_variables]
  if hourly_stats_path:
    standardizations += [
        Standardize(
            input_fields=hourly_var_renames,
            mean=read_stats(hourly_stats_path, hourly_variables, "mean"),
            std=read_stats(hourly_stats_path, hourly_variables, "std"),
        ),
    ]
  # Standardization for condition.
  daily_var_renames = [v.rename for v in daily_variables]
  if daily_stats_path:
    standardizations += [
        Standardize(
            input_fields=daily_var_renames,
            mean=read_stats(daily_stats_path, daily_variables, "mean"),
            std=read_stats(daily_stats_path, daily_variables, "std"),
        ),
    ]

  all_variables = hourly_var_renames + daily_var_renames
  transformations = standardizations + [
      # Data is lon-lat and model takes lat-lon.
      Rot90(input_fields=all_variables, k=1, axes=(1, 2)),
      Concatenate(
          input_fields=hourly_var_renames,
          output_field=sample_field_rename,
          axis=-1,
      ),
      Concatenate(
          input_fields=daily_var_renames, output_field="daily_mean", axis=-1
      ),
      RandomMaskout(
          input_fields=("daily_mean",), probability=cond_maskout_prob
      ),
      NestCondDict(cond_fields=("daily_mean",), prefix="channel:"),
  ]
  data_loader = pygrain.load(
      source=source,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=True,
      worker_count=worker_count,
      read_options=read_options,
  )
  return data_loader


class InputLoader:
  """Loads daily data as input for sampling."""

  def __init__(
      self,
      date_range: tuple[str, str] | None,
      daily_dataset_path: epath.PathLike,
      daily_variables: DatasetVariables,
      daily_stats_path: epath.PathLike,
      daily_stats_variables: DatasetVariables | None = None,
      dims_order: Sequence[str] = ("time", "longitude", "latitude", "level"),
  ):
    """Input dataloader constructor.

    Args:
      date_range: Same as in `DataSource`.
      daily_dataset_path: Same as in `DataSource`.
      daily_variables: Same as in `DataSource`.
      daily_stats_path: Same as in `create_dataloader`.
      daily_stats_variables: A separate set of metadata for which to read
        statistics. If `None`, will copy from `daily_variables`.
      dims_order: The order of the dimensions in the dataset.
    """
    if daily_stats_variables is None:
      daily_stats_variables = daily_variables

    ds = xr.open_zarr(daily_dataset_path)
    if date_range is not None:
      date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)
      ds = ds.sel(time=slice(*date_range))
    ds = ds.reindex(
        latitude=np.sort(ds.latitude), longitude=np.sort(ds.longitude)
    )
    ds = ds.transpose(*dims_order)

    self._data_arrays = {}
    for v in daily_variables:
      self._data_arrays[v.rename] = ds[v.name].sel(v.indexers, drop=True)

    self.mean = read_stats(daily_stats_path, daily_stats_variables, "mean")
    self.std = read_stats(daily_stats_path, daily_stats_variables, "std")

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


class VariableStackedInputLoader:
  """Loads variable-stacked daily data as input for sampling."""

  def __init__(
      self,
      date_range: tuple[str, str],
      daily_dataset_path: epath.PathLike,
      dataset_variable: str,
      mean: np.ndarray,
      std: np.ndarray,
  ):
    """Input dataloader constructor.

    Args:
      date_range: Same as in `DataSource`.
      daily_dataset_path: Assumes a different format than `InputLoader`. All
        physical variables (temperature, wind speed etc.) are stacked in a
        trailing channel dimension, into a single dataset variable. The dataset
        (.zarr) may contain multiple such variables, each of which may denote a
        different data source (e.g. era5, lens2, debiased), but only one of them
        will be loaded to be the input.
      dataset_variable: The dataset variable (containing variable-stacked data)
        to use for input.
      mean: Variable-stacked mean. Has shape (longitude, latitude, variables).
      std: Variable-stacked standard deviation. Has shape (longitude, latitude,
        variables).
    """
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)
    ds = xr.open_zarr(daily_dataset_path)[dataset_variable]
    self.ds = ds.sel(time=slice(*date_range))
    self.mean = mean
    self.std = std

  def get_days(self, days: np.ndarray) -> dict[str, np.ndarray]:
    """Retrieves the data record for a given starting day."""
    all_cond = (self.ds.sel(time=days).to_numpy() - self.mean) / self.std
    all_cond = np.rot90(all_cond, axes=(1, 2))
    return {"channel:daily_mean": all_cond}
