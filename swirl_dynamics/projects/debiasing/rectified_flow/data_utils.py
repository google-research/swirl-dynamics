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

"""Utilities for loading samples from initial/target sets from hdf5 files."""

from collections.abc import Callable, Mapping, Sequence
import types
from typing import Any, Literal, SupportsIndex

from absl import logging
from etils import epath
import grain.python as pygrain
import grain.tensorflow as tfgrain
import jax
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.projects.debiasing.rectified_flow import pygrain_transforms as transforms
import tensorflow as tf
import xarray_tensorstore as xrts

Array = jax.Array
PyTree = Any
DynamicsFn = Callable[[Array, Array, PyTree], Array]

_ERA5_VARIABLES = types.MappingProxyType({
    "2m_temperature": None,
    "specific_humidity": {"level": 1000},
    "geopotential": {"level": [200, 500]},
    "mean_sea_level_pressure": None,
})

_ERA5_WIND_COMPONENTS = types.MappingProxyType({
    "10m_u_component_of_wind": None,
    "10m_v_component_of_wind": None,
})

# pylint: disable=line-too-long
_ERA5_DATASET_PATH = "/lzepedanunez/data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = "/lzepedanunez/data/era5/stats/daily_mean_240x121_all_variables_and_wind_speed_1961-2000.zarr"

_LENS2_MEMBER_INDEXER = types.MappingProxyType({"member": "cmip6_1001_001"})
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "Z200", "Z500", "PSL", "WSPDSRFAV")

# Data set with original resolution.
_LENS2_DATASET_PATH_288_192 = (
    "/wanzy/data/lens2/unnormalized_288x192.zarr"
)
_LENS2_STATS_PATH_288_192 = "/wanzy/data/lens2/stats/all_variables_288x192_1961-2000.zarr/"

# Interpolated dataset to match the resolution of the ERA5 data set.
_LENS2_DATASET_PATH = (
    "/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
)
_LENS2_STATS_PATH = "/lzepedanunez/data/lens2/stats/all_variables_240x121_lonlat_1961-2000.zarr"
# pylint: enable=line-too-long

_LENS2_MEAN_STATS_PATH = "/lzepedanunez/data/lens2/stats/lens2_mean_stats_all_variables_240x121_lonlat_1961-2000.zarr"
_LENS2_STD_STATS_PATH = "/lzepedanunez/data/lens2/stats/lens2_std_stats_all_variables_240x121_lonlat_1961-2000.zarr"


class UnpairedDataLoader:
  """Unpaired dataloader for loading samples from two distributions."""

  def __init__(
      self,
      batch_size: int,
      dataset_path_a: str,
      dataset_path_b: str,
      seed: int,
      split: str | None = None,
      spatial_downsample_factor_a: int = 1,
      spatial_downsample_factor_b: int = 1,
      normalize: bool = False,
      normalize_stats_a: dict[str, Array] | None = None,
      normalize_stats_b: dict[str, Array] | None = None,
      tf_lookup_batch_size: int = 1024,
      tf_lookup_num_parallel_calls: int = -1,
      tf_interleaved_shuffle: bool = False,
  ):

    loader, normalize_stats_a = create_loader_from_hdf5(
        batch_size=batch_size,
        dataset_path=dataset_path_a,
        seed=seed,
        split=split,
        spatial_downsample_factor=spatial_downsample_factor_a,
        normalize=normalize,
        normalize_stats=normalize_stats_a,
        tf_lookup_batch_size=tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=tf_interleaved_shuffle,
    )

    self.loader_a = iter(loader)

    loader, normalize_stats_b = create_loader_from_hdf5(
        batch_size=batch_size,
        dataset_path=dataset_path_b,
        seed=seed,
        split=split,
        spatial_downsample_factor=spatial_downsample_factor_b,
        normalize=normalize,
        normalize_stats=normalize_stats_b,
        tf_lookup_batch_size=tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=tf_interleaved_shuffle,
    )
    self.loader_b = iter(loader)

    self.normalize_stats_a = normalize_stats_a
    self.normalize_stats_b = normalize_stats_b

  def __iter__(self):
    return self

  def __next__(self):

    b = next(self.loader_b)
    a = next(self.loader_a)

    # Return dictionary with a tuple, following the cycleGAN convention.
    return {"x_0": a["u"], "x_1": b["u"]}


def create_loader_from_hdf5(
    batch_size: int,
    dataset_path: str,
    seed: int,
    split: str | None = None,
    spatial_downsample_factor: int = 1,
    normalize: bool = False,
    normalize_stats: dict[str, Array] | None = None,
    tf_lookup_batch_size: int = 1024,
    tf_lookup_num_parallel_calls: int = -1,
    tf_interleaved_shuffle: bool = False,
) -> tuple[tfgrain.TfDataLoader, dict[str, Array] | None]:
  """Load pre-computed trajectories dumped to hdf5 file.

  Args:
    batch_size: Batch size returned by dataloader. If set to -1, use entire
      dataset size as batch_size.
    dataset_path: Absolute path to dataset file.
    seed: Random seed to be used in data sampling.
    split: Data split - train, eval, test, or None.
    spatial_downsample_factor: reduce spatial resolution by factor of x.
    normalize: Flag for adding data normalization (subtact mean divide by std.).
    normalize_stats: Dictionary with mean and std stats to avoid recomputing.
    tf_lookup_batch_size: Number of lookup batches (in cache) for grain.
    tf_lookup_num_parallel_calls: Number of parallel call for lookups in the
      dataset. -1 is set to let grain optimize tha number of calls.
    tf_interleaved_shuffle: Using a more localized shuffle instead of a global
      suffle of the data.

  Returns:
    loader, stats (optional): Tuple of dataloader and dictionary containing
                              mean and std stats (if normalize=True, else dict
                              contains NoneType values).
  """
  # TODO: create the data arrays following a similar convention.
  snapshots = hdf5_utils.read_single_array(
      dataset_path,
      f"{split}/u",
  )

  # If the data is given aggregated by trajectory, we scramble the time stamps.
  if snapshots.ndim == 5:
    # We assume that the data is 2-dimensional + channels.
    num_trajs, num_time, nx, ny, dim = snapshots.shape
    snapshots = snapshots.reshape((num_trajs * num_time, nx, ny, dim))
  elif snapshots.ndim != 4:
    raise ValueError(
        "The dimension of the data should be either a 5- or 4-",
        "dimensional tensor: two spatial dimension, one chanel ",
        "dimension and either number of samples, or number of ",
        "trajectories plus number of time-steps per trajectories.",
        f" Instead the data is a {snapshots.ndim}-tensor.",
    )

  # Downsample the data spatially, the data is two-dimensional.
  snapshots = snapshots[
      :, ::spatial_downsample_factor, ::spatial_downsample_factor, :
  ]

  return_stats = None

  if normalize:
    if normalize_stats is not None:
      mean = normalize_stats["mean"]
      std = normalize_stats["std"]
    else:
      if split != "train":
        data_for_stats = hdf5_utils.read_single_array(
            dataset_path,
            "train/u",
        )
        if data_for_stats.ndim == 5:
          num_trajs, num_time, nx, ny, dim = data_for_stats.shape
          data_for_stats = data_for_stats.reshape(
              (num_trajs * num_time, nx, ny, dim)
          )
        # Also perform the downsampling.
        data_for_stats = data_for_stats[
            :, ::spatial_downsample_factor, ::spatial_downsample_factor, :
        ]
      else:
        data_for_stats = snapshots

      # This need to be run in CPU. This needs to be done only once.
      mean = np.mean(data_for_stats, axis=0)
      std = np.std(data_for_stats, axis=0)

    # Normalize snapshot so they are distributed appropiately.
    snapshots -= mean
    snapshots /= std

    return_stats = {"mean": mean, "std": std}

  source = tfgrain.TfInMemoryDataSource.from_dataset(
      tf.data.Dataset.from_tensor_slices({
          "u": snapshots,
      })
  )

  # Grain fine-tuning.
  tfgrain.config.update(
      "tf_lookup_num_parallel_calls", tf_lookup_num_parallel_calls
  )
  tfgrain.config.update("tf_interleaved_shuffle", tf_interleaved_shuffle)
  tfgrain.config.update("tf_lookup_batch_size", tf_lookup_batch_size)

  if batch_size == -1:  # Use full dataset as batch
    batch_size = len(source)

  loader = tfgrain.TfDataLoader(
      source=source,
      sampler=tfgrain.TfDefaultIndexSampler(
          num_records=len(source),
          seed=seed,
          num_epochs=None,  # loads indefinitely.
          shuffle=True,
          shard_options=tfgrain.ShardByJaxProcess(drop_remainder=True),
      ),
      transformations=[],
      batch_fn=tfgrain.TfBatch(batch_size=batch_size, drop_remainder=False),
  )
  return loader, return_stats


def read_stats(
    dataset: epath.PathLike,
    variables: Mapping[str, Mapping[str, Any] | None],
    field: Literal["mean", "std"],
    use_batched: bool = False,
) -> dict[str, np.ndarray]:
  """Reads variables from a zarr dataset and returns as a dict of ndarrays."""
  ds = xrts.open_zarr(dataset)
  out = {}
  for var, indexers in variables.items():
    indexers = indexers | {"stats": field} if indexers else {"stats": field}
    stats = ds[var].sel(indexers).to_numpy()
    assert stats.ndim == 2 or stats.ndim == 3
    stats = np.expand_dims(stats, axis=-1) if stats.ndim == 2 else stats
    if use_batched:
      # If we are using a batched or chuncked dataloader add an extra dim.
      stats = np.expand_dims(stats, axis=0)
    out[var] = stats
  return out


class SingleSource:
  """A data source that loads samples from a single Zarr dataset."""

  def __init__(
      self,
      date_range: tuple[str, str],
      dataset_path: epath.PathLike,
      variables: Mapping[str, Mapping[str, Any] | None],
      dims_order: Sequence[str] | None = None,
      resample_at_nan: bool = False,
      resample_seed: int = 666,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in years) applied.
      dataset_path: Path for the Zarr dataset.
      variables: The variables to yield from the input dataset.
      dims_order: The order of the dimensions.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Whether to add the time stamps to the samples.
    """
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)
    ds = xrts.open_zarr(dataset_path).sel(time=slice(*date_range))
    if dims_order:
      ds = ds.transpose(*dims_order)
    self._data_arrays = {}
    for v, indexers in variables.items():
      self._data_arrays[v] = ds[v].sel(indexers)

    self._date_range = date_range
    self._time_array = xrts.read(ds["time"]).data
    self._len = len(self._time_array)
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, np.ndarray]:
    """Returns the data record for a given key."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    item = self.get_item(idx)

    if self._resample_at_nan:
      while np.array([np.isnan(val).any() for val in item.values()]).any():
        logging.info("NaN detected")

        rng = np.random.default_rng(self._resample_seed + idx)
        resample_idx = rng.integers(0, len(self))
        item = self.get_item(resample_idx)

    return item

  def get_item(self, idx: int) -> dict[str, np.ndarray]:
    item = {}
    for v, da in self._data_arrays.items():
      array = xrts.read(da.isel(time=idx)).data
      assert array.ndim == 2 or array.ndim == 3
      item[v] = np.expand_dims(array, axis=-1) if array.ndim == 2 else array

    if self._time_stamps:
      item["time_stamp"] = self._time_array[idx]
    return item


class ContiguousSource:
  """A data source that loads a single dataset by contiguous chunks."""

  def __init__(
      self,
      date_range: tuple[str, str],
      batch_size: int,
      dataset_path: epath.PathLike,
      variables: Mapping[str, Mapping[str, Any] | None],
      dims_order: Sequence[str] | None = None,
      time_stamp: bool = False,
  ):
    """Data source constructor."""
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)
    ds = xrts.open_zarr(dataset_path).sel(time=slice(*date_range))

    if dims_order:
      ds = ds.transpose(*dims_order)
    self._data_arrays = {}

    for v, indexers in variables.items():
      self._data_arrays[v] = ds[v].sel(indexers)

    self._batch_size = batch_size
    self._time_stamp = time_stamp
    self._date_range = date_range
    # TODO: remove this a bit.
    self._time_array = xrts.read(ds["time"]).data
    self._len = len(self._time_array) - self._batch_size

  def __len__(self):
    """Returns the length of the data source."""
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, np.ndarray]:
    """Returns the data record for a given key."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    example = {}
    for v, da in self._data_arrays.items():
      array = xrts.read(da.isel(time=slice(idx, idx + self._batch_size))).data
      assert array.ndim == 3 or array.ndim == 4
      example[v] = np.expand_dims(array, axis=-1) if array.ndim == 3 else array

      if self._time_stamp:
        example["time_stamp"] = self._time_array[idx]
    return example


class DataSource:
  """A data source that loads aligned (by date) daily mean ERA5-LENS2 data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      input_dataset: epath.PathLike,
      input_variable_names: Sequence[str],
      input_member_indexer: Mapping[str, Any],
      output_dataset: epath.PathLike,
      output_variables: Mapping[str, Any],
      dims_order_input: Sequence[str] | None = None,
      dims_order_output: Sequence[str] | None = None,
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variable_names: The names of the variables to yield from the input
        dataset.
      input_member_indexer: The name of the ensemble member to sample from.
      output_dataset: The path of a zarr dataset containing the output data.
      output_variables: The variables to yield from the output dataset.
      dims_order_input: Order of the variables for the input.
      dims_order_output: Order of the variables for the output.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Whether to add the time stamps to the samples.
    """

    # Using lens as input, they need to be modified.
    input_variables = {v: input_member_indexer for v in input_variable_names}

    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    output_ds = xrts.open_zarr(output_dataset).sel(time=slice(*date_range))

    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_output:
      output_ds = output_ds.transpose(*dims_order_output)

    self._input_arrays = {}
    for v, indexers in input_variables.items():
      self._input_arrays[v] = input_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    self._output_arrays = {}
    for v, indexers in output_variables.items():
      self._output_arrays[v] = output_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    self._output_coords = output_ds.coords

    # self._dates = get_common_times(input_ds, date_range)
    # The times can be slightly off due to the leap years.
    self._len = np.min([input_ds.dims["time"], output_ds.dims["time"]])
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps

  def __len__(self):
    """Returns the length of the data source."""
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, Any]:
    """Retrieves record and retry if NaN found for a given key."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    item = self.get_item(idx)

    if self._resample_at_nan:
      while np.isnan(item["input"]).any() or np.isnan(item["output"]).any():
        logging.info("NaN detected for day %s", str(item["input_time_stamp"]))

        rng = np.random.default_rng(self._resample_seed + idx)
        resample_idx = rng.integers(0, len(self))
        item = self.get_item(resample_idx)

    return item

  def get_item(self, idx: int) -> Mapping[str, Any]:
    """Returns the data record for a given index."""
    item = {}

    sample_input = {}
    for v, da in self._input_arrays.items():
      array = xrts.read(da.isel(time=idx)).data
      assert array.ndim == 2 or array.ndim == 3
      sample_input[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 2 else array
      )

    sample_input["time_stamp"] = self._input_time_array[idx]

    item["input"] = sample_input

    sample_output = {}
    # TODO encapsulate these functions.
    for v, da in self._output_arrays.items():
      array = xrts.read(da.isel(time=idx)).data
      assert array.ndim == 2 or array.ndim == 3
      sample_output[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 2 else array
      )

    item["output"] = sample_output

    if self._time_stamps:
      item["input_time_stamp"] = self._input_time_array[idx]
      item["output_time_stamp"] = self._output_time_array[idx]

    return item

  def get_output_coords(self):
    return self._output_coords


# TODO Merge this class with the one below.
class DataSourceContiguous:
  """A data source that loads date aligned daily ERA5-LENS2 data by chunks."""

  def __init__(
      self,
      date_range: tuple[str, str],
      batch_size: int,
      input_dataset: epath.PathLike,
      input_variable_names: Sequence[str],
      input_member_indexer: Mapping[str, Any],
      output_dataset: epath.PathLike,
      output_variables: Mapping[str, Any],
      dims_order_input: Sequence[str] | None = None,
      dims_order_output: Sequence[str] | None = None,
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      batch_size: The size of the batch/chunk.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variable_names: The names of the variables to yield from the input
        dataset.
      input_member_indexer: The name of the ensemble member to sample from.
      output_dataset: The path of a zarr dataset containing the output data.
      output_variables: The variables to yield from the output dataset.
      dims_order_input: Order of the variables for the input.
      dims_order_output: Order of the variables for the output.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Whether to add the time stamps to the samples.
    """

    # Using lens as input, they need to be modified.
    input_variables = {v: input_member_indexer for v in input_variable_names}

    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    output_ds = xrts.open_zarr(output_dataset).sel(time=slice(*date_range))

    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_output:
      output_ds = output_ds.transpose(*dims_order_output)

    self._input_arrays = {}
    for v, indexers in input_variables.items():
      self._input_arrays[v] = input_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    self._output_arrays = {}
    for v, indexers in output_variables.items():
      self._output_arrays[v] = output_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    self._output_coords = output_ds.coords

    # The times can be slightly off due to the leap years.
    self._len = (
        np.min([input_ds.dims["time"], output_ds.dims["time"]]) - batch_size
    )
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps
    self._batch_size = batch_size

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, Any]:
    """Retrieves record and retry if NaN found."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    item = self.get_item(idx)

    if self._resample_at_nan:
      while np.isnan(item["input"]).any() or np.isnan(item["output"]).any():
        logging.info("NaN detected for day %s", str(item["input_time_stamp"]))

        rng = np.random.default_rng(self._resample_seed + idx)
        resample_idx = rng.integers(0, len(self))
        item = self.get_item(resample_idx)

    return item

  def get_item(self, idx: int) -> Mapping[str, Any]:
    """Returns the data record for a given index."""
    item = {}

    sample_input = {}
    for v, da in self._input_arrays.items():
      array = xrts.read(da.isel(time=slice(idx, idx + self._batch_size))).data
      assert array.ndim == 3 or array.ndim == 4
      sample_input[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    sample_input["time_stamp"] = self._input_time_array[idx]

    item["input"] = sample_input

    sample_output = {}
    # TODO encapsulate these functions.
    for v, da in self._output_arrays.items():
      array = xrts.read(da.isel(time=slice(idx, idx + self._batch_size))).data
      assert array.ndim == 3 or array.ndim == 4
      sample_output[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    item["output"] = sample_output

    if self._time_stamps:
      item["input_time_stamp"] = self._input_time_array[idx]
      item["output_time_stamp"] = self._output_time_array[idx]

    return item

  def get_output_coords(self):
    return self._output_coords


class DataSourceContiguousEnsemble:
  """A data source that loads paired daily ERA5- with ensemble LENS2 data.

  This loader is used to load data snapshots from different ensemble members,
  which are then aligned using their date-stamps with snapshot from ERA5 data.

  This is contrast to the function above, which only loads data from a single
  ensemble member.
  """

  def __init__(
      self,
      date_range: tuple[str, str],
      batch_size: int,
      input_dataset: epath.PathLike,
      input_variable_names: Sequence[str],
      input_member_indexer: tuple[Mapping[str, Any], ...],
      output_dataset: epath.PathLike,
      output_variables: Mapping[str, Any],
      dims_order_input: Sequence[str] | None = None,
      dims_order_output: Sequence[str] | None = None,
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      batch_size: Size of the batch.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variable_names: The names of the variables to yield from the input
        dataset.
      input_member_indexer: The name of the ensemble member to sample from.
      output_dataset: The path of a zarr dataset containing the output data.
      output_variables: The variables to yield from the output dataset.
      dims_order_input: Order of the variables for the input.
      dims_order_output: Order of the variables for the output.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Whether to add the time stamps to the samples.
    """

    # Using lens as input, they need to be modified.
    input_variables = {v: input_member_indexer for v in input_variable_names}

    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    output_ds = xrts.open_zarr(output_dataset).sel(time=slice(*date_range))

    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_output:
      output_ds = output_ds.transpose(*dims_order_output)

    self._input_arrays = {}
    for v, indexers in input_variables.items():
      self._input_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_arrays[v][idx] = input_ds[v].sel(index)

    self._output_arrays = {}
    for v, indexers in output_variables.items():
      self._output_arrays[v] = output_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    self._output_coords = output_ds.coords

    # self._dates = get_common_times(input_ds, date_range)
    # The times can be slightly off due to the leap years.
    self._indexes = [ind["member"] for ind in input_member_indexer]
    self._len_time = (
        np.min([input_ds.dims["time"], output_ds.dims["time"]]) - batch_size
    )
    self._len = len(self._indexes) * self._len_time
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps
    self._batch_size = batch_size

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, Any]:
    """Retrieves record and retry if NaN found."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    item = self.get_item(idx)

    if self._resample_at_nan:
      while np.isnan(item["input"]).any() or np.isnan(item["output"]).any():
        logging.info("NaN detected for day %s", str(item["input_time_stamp"]))

        rng = np.random.default_rng(self._resample_seed + idx)
        resample_idx = rng.integers(0, len(self))
        item = self.get_item(resample_idx)

    return item

  def get_item(self, idx: int) -> Mapping[str, Any]:
    """Returns the data record for a given index."""
    item = {}

    idx_member = idx // self._len_time
    idx_time = idx % self._len_time
    member = self._indexes[idx_member]
    sample_input = {}
    for v, da in self._input_arrays.items():
      array = xrts.read(
          da[member].isel(time=slice(idx_time, idx_time + self._batch_size))
      ).data

      assert array.ndim == 3 or array.ndim == 4
      sample_input[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    item["input"] = sample_input

    sample_output = {}
    # TODO encapsulate these functions.
    for v, da in self._output_arrays.items():
      array = xrts.read(
          da.isel(time=slice(idx_time, idx_time + self._batch_size))
      ).data
      assert array.ndim == 3 or array.ndim == 4
      sample_output[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    item["output"] = sample_output

    if self._time_stamps:
      item["input_time_stamp"] = self._input_time_array[idx_time]
      item["output_time_stamp"] = self._output_time_array[idx_time]

    return item

  def get_output_coords(self):
    return self._output_coords


class DataSourceContiguousEnsembleWithStats:
  """A data source that loads paired daily ERA5- with ensemble LENS2 data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      batch_size: int,
      input_dataset: epath.PathLike,
      input_variable_names: Sequence[str],
      input_member_indexer: tuple[Mapping[str, Any], ...],
      input_stats_dataset: epath.PathLike,
      output_dataset: epath.PathLike,
      output_variables: Mapping[str, Any],
      dims_order_input: Sequence[str] | None = None,
      dims_order_output: Sequence[str] | None = None,
      dims_order_input_stats: Sequence[str] | None = (
          "member",
          "longitude",
          "latitude",
          "stats",
      ),
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      batch_size: Size of the batch.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variable_names: The variables to yield from the input dataset.
      input_member_indexer: The name of the ensemble member to sample from.
      input_stats_dataset: The path of a zarr dataset containing the input
        statistics.
      output_dataset: The path of a zarr dataset containing the output data.
      output_variables: The variables to yield from the output dataset.
      dims_order_input: Order of the variables for the input.
      dims_order_output: Order of the variables for the output.
      dims_order_input_stats: Order of the variables for the input statistics.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Wheter to add the time stamps to the samples.
    """

    # Using lens as input, they need to be modified.
    input_variables = {v: input_member_indexer for v in input_variable_names}

    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    output_ds = xrts.open_zarr(output_dataset).sel(time=slice(*date_range))
    input_stats_ds = xrts.open_zarr(input_stats_dataset)

    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_output:
      output_ds = output_ds.transpose(*dims_order_output)
    if dims_order_input_stats:
      input_stats_ds = input_stats_ds.transpose(*dims_order_input_stats)

    self._input_arrays = {}
    for v, indexers in input_variables.items():
      self._input_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_arrays[v][idx] = input_ds[v].sel(index)

    self._input_stats_arrays = {}
    for v, indexers in input_variables.items():
      self._input_stats_arrays[v] = {}
      for index in indexers:
        # print(f"index = {index}")
        idx = tuple(index.values())[0]
        self._input_stats_arrays[v][idx] = input_stats_ds[v].sel(index)

    self._output_arrays = {}
    for v, indexers in output_variables.items():
      self._output_arrays[v] = output_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    self._output_coords = output_ds.coords

    # self._dates = get_common_times(input_ds, date_range)
    # The times can be slightly off due to the leap years.
    self._indexes = [ind["member"] for ind in input_member_indexer]
    self._len_time = (
        np.min([input_ds.dims["time"], output_ds.dims["time"]]) - batch_size
    )
    self._len = len(self._indexes) * self._len_time
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps
    self._batch_size = batch_size

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, Any]:
    """Retrieves record and retry if NaN found."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    item = self.get_item(idx)

    if self._resample_at_nan:
      while np.isnan(item["input"]).any() or np.isnan(item["output"]).any():

        rng = np.random.default_rng(self._resample_seed + idx)
        resample_idx = rng.integers(0, len(self))
        item = self.get_item(resample_idx)

    return item

  def get_item(self, idx: int) -> Mapping[str, Any]:
    """Returns the data record for a given index."""
    item = {}

    idx_member = idx // self._len_time
    idx_time = idx % self._len_time

    member = self._indexes[idx_member]
    sample_input = {}
    mean_input = {}
    std_input = {}

    for v, da in self._input_arrays.items():
      array = xrts.read(
          da[member].isel(time=slice(idx_time, idx_time + self._batch_size))
      ).data

      assert array.ndim == 3 or array.ndim == 4
      sample_input[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    for v, da in self._input_stats_arrays.items():
      mean_array = xrts.read(da[member].sel(stats="mean")).data

      # As there is no time dimension yet the mean and std are either
      # 2- or 3-tensors.
      assert mean_array.ndim == 2 or mean_array.ndim == 3
      mean_array = (
          np.expand_dims(mean_array, axis=-1)
          if mean_array.ndim == 2
          else mean_array
      )
      mean_input[v] = np.tile(
          mean_array, (self._batch_size,) + (1,) * mean_array.ndim
      )

      std_array = xrts.read(da[member].sel(stats="std")).data

      assert std_array.ndim == 2 or std_array.ndim == 3
      std_array = (
          np.expand_dims(std_array, axis=-1)
          if std_array.ndim == 2
          else std_array
      )
      std_input[v] = np.tile(
          std_array, (self._batch_size,) + (1,) * std_array.ndim
      )

    item["input"] = sample_input
    item["input_mean"] = mean_input
    item["input_std"] = std_input

    sample_output = {}
    # TODO encapsulate these functions.
    for v, da in self._output_arrays.items():
      array = xrts.read(
          da.isel(time=slice(idx_time, idx_time + self._batch_size))
      ).data
      assert array.ndim == 3 or array.ndim == 4
      sample_output[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    item["output"] = sample_output

    if self._time_stamps:
      item["input_time_stamp"] = self._input_time_array[
          idx_time : idx_time + self._batch_size
      ]
      item["output_time_stamp"] = self._output_time_array[
          idx_time : idx_time + self._batch_size
      ]

    return item

  def get_output_coords(self):
    return self._output_coords


class DataSourceContiguousNonOverlappingEnsembleWithStats:
  """A data source that loads paired daily ERA5- with ensemble LENS2 data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      batch_size: int,
      input_dataset: epath.PathLike,
      input_variable_names: Sequence[str],
      input_member_indexer: tuple[Mapping[str, Any], ...],
      input_stats_dataset: epath.PathLike,
      output_dataset: epath.PathLike,
      output_variables: Mapping[str, Any],
      dims_order_input: Sequence[str] | None = None,
      dims_order_output: Sequence[str] | None = None,
      dims_order_input_stats: Sequence[str] | None = (
          "member",
          "longitude",
          "latitude",
          "stats",
      ),
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      batch_size: Size of the batch.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variable_names: The names of variables (in a tuple) to yield from
        the input dataset.
      input_member_indexer: The name of the ensemble member to sample from.
      input_stats_dataset: The path of a zarr dataset containing the input
        statistics.
      output_dataset: The path of a zarr dataset containing the output data.
      output_variables: The variables to yield from the output dataset.
      dims_order_input: Order of the variables for the input.
      dims_order_output: Order of the variables for the output.
      dims_order_input_stats: Order of the variables for the input statistics.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Wheter to add the time stamps to the samples.
    """

    # Using lens as input, they need to be modified.
    input_variables = {v: input_member_indexer for v in input_variable_names}

    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    output_ds = xrts.open_zarr(output_dataset).sel(time=slice(*date_range))
    input_stats_ds = xrts.open_zarr(input_stats_dataset)

    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_output:
      output_ds = output_ds.transpose(*dims_order_output)
    if dims_order_input_stats:
      input_stats_ds = input_stats_ds.transpose(*dims_order_input_stats)

    self._input_arrays = {}
    for v, indexers in input_variables.items():
      self._input_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_arrays[v][idx] = input_ds[v].sel(index)

    self._input_stats_arrays = {}
    for v, indexers in input_variables.items():
      self._input_stats_arrays[v] = {}
      for index in indexers:
        # print(f"index = {index}")
        idx = tuple(index.values())[0]
        self._input_stats_arrays[v][idx] = input_stats_ds[v].sel(index)

    self._output_arrays = {}
    for v, indexers in output_variables.items():
      self._output_arrays[v] = output_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    self._output_coords = output_ds.coords

    # self._dates = get_common_times(input_ds, date_range)
    # The times can be slightly off due to the leap years.
    self._indexes = [ind["member"] for ind in input_member_indexer]
    self._len_time = (
        np.min([input_ds.dims["time"], output_ds.dims["time"]]) // batch_size
    )
    self._len = len(self._indexes) * self._len_time
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps
    self._batch_size = batch_size

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, Any]:
    """Retrieves record and retry if NaN found."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    item = self.get_item(idx)

    if self._resample_at_nan:
      while np.isnan(item["input"]).any() or np.isnan(item["output"]).any():

        rng = np.random.default_rng(self._resample_seed + idx)
        resample_idx = rng.integers(0, len(self))
        item = self.get_item(resample_idx)

    return item

  def get_item(self, idx: int) -> Mapping[str, Any]:
    """Returns the data record for a given index."""
    item = {}

    idx_member = idx // self._len_time
    idx_time = idx % self._len_time

    member = self._indexes[idx_member]
    sample_input = {}
    mean_input = {}
    std_input = {}

    for v, da in self._input_arrays.items():
      array = xrts.read(
          da[member].isel(
              time=slice(
                  idx_time * self._batch_size, (idx_time + 1) * self._batch_size
              )
          )
      ).data

      assert array.ndim == 3 or array.ndim == 4
      sample_input[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    for v, da in self._input_stats_arrays.items():
      mean_array = xrts.read(da[member].sel(stats="mean")).data

      # there is no time dimension yet,
      # so the mean and std are either 2- or 3-tensors
      assert mean_array.ndim == 2 or mean_array.ndim == 3
      mean_array = (
          np.expand_dims(mean_array, axis=-1)
          if mean_array.ndim == 2
          else mean_array
      )
      mean_input[v] = np.tile(
          mean_array, (self._batch_size,) + (1,) * mean_array.ndim
      )

      std_array = xrts.read(da[member].sel(stats="std")).data

      assert std_array.ndim == 2 or std_array.ndim == 3
      std_array = (
          np.expand_dims(std_array, axis=-1)
          if std_array.ndim == 2
          else std_array
      )
      std_input[v] = np.tile(
          std_array, (self._batch_size,) + (1,) * std_array.ndim
      )

    item["input"] = sample_input
    item["input_mean"] = mean_input
    item["input_std"] = std_input

    sample_output = {}
    # TODO encapsulate these functions.
    for v, da in self._output_arrays.items():
      # We change the slicing to be contiguous and non-overlapping.
      array = xrts.read(
          da.isel(
              time=slice(
                  idx_time * self._batch_size, (idx_time + 1) * self._batch_size
              )
          )
      ).data
      assert array.ndim == 3 or array.ndim == 4
      sample_output[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    item["output"] = sample_output

    if self._time_stamps:
      item["input_time_stamp"] = self._input_time_array[
          idx_time * self._batch_size : (idx_time + 1) * self._batch_size
      ]
      item["output_time_stamp"] = self._output_time_array[
          idx_time * self._batch_size : (idx_time + 1) * self._batch_size
      ]

    return item

  def get_output_coords(self):
    return self._output_coords


class DataSourceContiguousEnsembleNonOverlappingWithStatsLENS2:
  """A data source that loads ensemble LENS2 data with stats."""

  def __init__(
      self,
      date_range: tuple[str, str],
      batch_size: int,
      input_dataset: epath.PathLike,
      input_variable_names: Sequence[str],
      input_member_indexer: tuple[Mapping[str, Any], ...],
      input_stats_dataset: epath.PathLike,
      dims_order_input: Sequence[str] | None = None,
      dims_order_input_stats: Sequence[str] | None = (
          "member",
          "longitude",
          "latitude",
          "stats",
      ),
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      batch_size: Size of the batch.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variable_names: The variables to yield from the input dataset.
      input_member_indexer: The name of the ensemble member to sample from.
      input_stats_dataset: The path of zarr dataset containing the input stats.
      dims_order_input: Order of the variables for the input.
      dims_order_input_stats: Order of the variables for the input stats.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Wheter to add the time stamps to the samples.
    """

    # Using lens as input, they need to be modified.
    # This only accepts a sequence
    input_variables = {v: input_member_indexer for v in input_variable_names}

    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    input_stats_ds = xrts.open_zarr(input_stats_dataset)

    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_input_stats:
      input_stats_ds = input_stats_ds.transpose(*dims_order_input_stats)

    self._input_arrays = {}
    for v, indexers in input_variables.items():
      self._input_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_arrays[v][idx] = input_ds[v].sel(index)

    self._input_stats_arrays = {}
    for v, indexers in input_variables.items():
      self._input_stats_arrays[v] = {}
      for index in indexers:
        # print(f"index = {index}")
        idx = tuple(index.values())[0]
        self._input_stats_arrays[v][idx] = input_stats_ds[v].sel(index)

    # self._dates = get_common_times(input_ds, date_range)
    # The times can be slightly off due to the leap years.
    self._indexes = [ind["member"] for ind in input_member_indexer]
    # We will use non-overlapping time steps.
    self._len_time = input_ds.dims["time"] // batch_size
    self._len = len(self._indexes) * self._len_time
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps
    self._batch_size = batch_size

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, Any]:
    """Retrieves record and retry if NaN found."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    item = self.get_item(idx)

    if self._resample_at_nan:
      while np.isnan(item["input"]).any():

        rng = np.random.default_rng(self._resample_seed + idx)
        resample_idx = rng.integers(0, len(self))
        item = self.get_item(resample_idx)

    return item

  def get_item(self, idx: int) -> Mapping[str, Any]:
    """Returns the data record for a given index."""
    item = {}

    idx_member = idx // self._len_time
    idx_time = idx % self._len_time

    member = self._indexes[idx_member]
    sample_input = {}
    mean_input = {}
    std_input = {}

    for v, da in self._input_arrays.items():
      array = xrts.read(
          da[member].isel(
              time=slice(
                  idx_time * self._batch_size, (idx_time + 1) * self._batch_size
              )
          )
      ).data

      assert array.ndim == 3 or array.ndim == 4
      sample_input[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 3 else array
      )

    for v, da in self._input_stats_arrays.items():
      mean_array = xrts.read(da[member].sel(stats="mean")).data

      # there is no time dimension yet,
      # so the mean and std are either 2- or 3-tensors
      assert mean_array.ndim == 2 or mean_array.ndim == 3
      mean_array = (
          np.expand_dims(mean_array, axis=-1)
          if mean_array.ndim == 2
          else mean_array
      )
      mean_input[v] = np.tile(
          mean_array, (self._batch_size,) + (1,) * mean_array.ndim
      )

      std_array = xrts.read(da[member].sel(stats="std")).data

      assert std_array.ndim == 2 or std_array.ndim == 3
      std_array = (
          np.expand_dims(std_array, axis=-1)
          if std_array.ndim == 2
          else std_array
      )
      std_input[v] = np.tile(
          std_array, (self._batch_size,) + (1,) * std_array.ndim
      )

    item["input"] = sample_input
    item["input_mean"] = mean_input
    item["input_std"] = std_input

    if self._time_stamps:
      item["input_time_stamp"] = self._input_time_array[
          idx_time * self._batch_size : (idx_time + 1) * self._batch_size
      ]

    return item


def create_era5_loader(
    date_range: tuple[str, str],
    variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,  # pylint: disable=dangerous-default-value
    wind_components: (
        dict[str, dict[str, Any]] | types.MappingProxyType
    ) = _ERA5_WIND_COMPONENTS,  # pylint: disable=dangerous-default-value
    dataset_path: epath.PathLike = _ERA5_DATASET_PATH,
    stats_path: epath.PathLike = _ERA5_STATS_PATH,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    time_stamps: bool = False,
):
  """Creates a loader for ERA5 dataset.

  Args:
    date_range: Range in years of the data considered.
    variables: The names of the variables in the Zarr file to be included.
    wind_components: Dictionary with the wind components to be considered.
    dataset_path: The path to the dataset.
    stats_path: The path to the files with the precomputed statistics, which
      included the mean and standard deviation.
    shuffle: Whether to randomly shuffle the data.
    seed: Random seed for the random number generator.
    batch_size: The size of batch.
    drop_remainder: Whether drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to add the time stamps to the samples.

  Returns:
    A grain dataloader.
  """

  source = SingleSource(
      date_range=date_range,
      dataset_path=dataset_path,
      variables=variables | wind_components,
      dims_order=["time", "longitude", "latitude", "level"],
      time_stamps=time_stamps,
  )

  transformations = []

  transformations.append(
      transforms.Standardize(
          input_fields=variables.keys(),
          mean=read_stats(stats_path, variables, "mean"),
          std=read_stats(stats_path, variables, "std"),
      )
  )
  if wind_components:
    transformations.append(
        transforms.ComputeWindSpeedExact(
            u_field="10m_u_component_of_wind",
            v_field="10m_v_component_of_wind",
            speed_field="10m_magnitude_of_wind",
            mean=read_stats(
                stats_path,
                {
                    "10m_magnitude_of_wind": wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "mean",
            ),
            std=read_stats(
                stats_path,
                {
                    "10m_magnitude_of_wind": wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "std",
            ),
            output_field="wind_speed",
            remove_inputs=True,
        ),
    )
  input_fields = (
      (*variables.keys(), "wind_speed")
      if wind_components
      else (*variables.keys(),)
  )
  transformations.append(
      transforms.Concatenate(
          input_fields=input_fields,
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      )
  )

  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_lens2_loader(
    date_range: tuple[str, str],
    dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    stats_path: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    member_indexer: (
        dict[str, str] | types.MappingProxyType
    ) = _LENS2_MEMBER_INDEXER,
    variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    time_stamps: bool = False,
):
  """Creates a loader for LENS2 dataset.

  Args:
    date_range: Range for the data used in the dataloader.
    dataset_path:
    stats_path:
    member_indexer: The index for the ensemble member from which the data is
      extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    variable_names: The names of the variables in the Zarr file to be included.
    shuffle: Do we shuffle the data or the samples are done sequentially.
    seed: Random seed for the random number generator.
    batch_size: The size of batch.
    drop_remainder: Whether drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to add the time stamps to the samples.

  Returns:
    A pygrain dataloader with the LENS2 data set.
  """

  variables = {v: member_indexer for v in variable_names}

  source = SingleSource(
      date_range=date_range,
      dataset_path=dataset_path,
      variables=variables,
      dims_order=["member", "time", "longitude", "latitude"],
      time_stamps=time_stamps,
  )

  # Adding the different transformations.
  transformations = []
  transformations.append(
      transforms.Standardize(
          input_fields=variable_names,
          mean=read_stats(stats_path, variables, "mean"),
          std=read_stats(stats_path, variables, "std"),
      )
  )
  transformations.append(
      transforms.Concatenate(
          input_fields=variable_names,
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      )
  )

  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_default_era5_loader(
    date_range: tuple[str, str],
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
):
  """Creates the default loader for ERA5 dataset.

  This is the default loader for the ERA5 dataset. It includes the variables
  and wind components that are used in the debaising rectified flow project.

  Args:
    date_range: Range for the data used in the dataloader.
    shuffle: Do we shuffle the data or the samples are done sequentially.
    seed: Random seed for the random number generator.
    batch_size: The size of batch.
    drop_remainder: Whether drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.

  Returns:
    A pygrain dataloader with the LENS2 data set.
  """

  dataset_path = "/weatherbench/datasets/era5_daily_mean/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
  stats_path = "/wanzy/data/era5/stats/daily_mean_240x121_all_variables_and_wind_speed_1961-2000.zarr"
  variables = {
      "2m_temperature": None,
      "specific_humidity": {"level": 1000},
      "geopotential": {"level": [200, 500]},
      "mean_sea_level_pressure": None,
  }
  wind_components = {
      "10m_u_component_of_wind": None,
      "10m_v_component_of_wind": None,
  }

  source = SingleSource(
      date_range=date_range,
      dataset_path=dataset_path,
      variables=variables | wind_components,
      dims_order=["time", "longitude", "latitude", "level"],
  )
  transformations = [
      transforms.Standardize(
          input_fields=variables.keys(),
          mean=read_stats(stats_path, variables, "mean"),
          std=read_stats(stats_path, variables, "std"),
      ),
      transforms.ComputeWindSpeedExact(
          u_field="10m_u_component_of_wind",
          v_field="10m_v_component_of_wind",
          speed_field="10m_magnitude_of_wind",
          mean=read_stats(
              stats_path,
              {
                  "10m_magnitude_of_wind": wind_components[
                      "10m_u_component_of_wind"
                  ]
              },
              "mean",
          ),
          std=read_stats(
              stats_path,
              {
                  "10m_magnitude_of_wind": wind_components[
                      "10m_u_component_of_wind"
                  ]
              },
              "std",
          ),
          output_field="wind_speed",
          remove_inputs=True,
      ),
      transforms.Concatenate(
          input_fields=(*variables.keys(), "wind_speed"),
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      ),
  ]
  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_default_lens2_loader(
    date_range: tuple[str, str],
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
):
  """Creates the default loader for Lens2 dataset.

  Args:
    date_range: Range for the data used in the dataloader.
    shuffle: Do we shuffle the data or the samples are done sequentially.
    seed: Random seed for the random number generator.
    batch_size: The size of batch.
    drop_remainder: Whether drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.

  Returns:
    A pygrain dataloader with the LENS2 data set.
  """

  dataset_path = "/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
  stats_path = "/lzepedanunez/data/lens2/stats/all_variables_240x121_lonlat_1961-2000.zarr/"
  member_indexer = {"member": "cmip6_1001_001"}
  variable_names = ["TREFHT", "QREFHT", "Z200", "Z500", "PSL", "WSPDSRFAV"]
  variables = {v: member_indexer for v in variable_names}

  source = SingleSource(
      date_range=date_range,
      dataset_path=dataset_path,
      variables=variables,
      dims_order=["member", "time", "longitude", "latitude"],
  )
  transformations = [
      transforms.Standardize(
          input_fields=variable_names,
          mean=read_stats(stats_path, variables, "mean"),
          std=read_stats(stats_path, variables, "std"),
      ),
      transforms.Concatenate(
          input_fields=variable_names,
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  ]

  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_chunked_era5_loader(
    date_range: tuple[str, str],
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    num_chunks: int = 1,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
):
  """Creates the chunked loader for ERA5 dataset.

  Args:
    date_range: Range for the data used in the dataloader.
    shuffle: Do we shuffle the data or the samples are done sequentially.
    seed: Random seed for the random number generator.
    batch_size: The size of each contiguous chunks.
    num_chunks: Number of contiguous chunks to sample from the dataset.
    drop_remainder: Whether drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.

  Returns:
    A pygrain dataloader with the ERA5 data set.
  """
  dataset_path = "/weatherbench/datasets/era5_daily_mean/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
  stats_path = "/wanzy/data/era5/stats/daily_mean_240x121_all_variables_and_wind_speed_1961-2000.zarr"
  variables = {
      "2m_temperature": None,
      "specific_humidity": {"level": 1000},
      "geopotential": {"level": [200, 500]},
      "mean_sea_level_pressure": None,
  }
  wind_components = {
      "10m_u_component_of_wind": None,
      "10m_v_component_of_wind": None,
  }

  # TODO: add a if else here to use either single or contiguous
  # data sets.
  source = ContiguousSource(
      date_range=date_range,
      dataset_path=dataset_path,
      batch_size=batch_size,
      variables=variables | wind_components,
      dims_order=["time", "longitude", "latitude", "level"],
  )
  transformations = [
      transforms.Standardize(
          input_fields=variables.keys(),
          mean=read_stats(stats_path, variables, "mean", use_batched=True),
          std=read_stats(stats_path, variables, "std", use_batched=True),
      ),
      transforms.ComputeWindSpeedExact(
          u_field="10m_u_component_of_wind",
          v_field="10m_v_component_of_wind",
          speed_field="10m_magnitude_of_wind",
          mean=read_stats(
              stats_path,
              {
                  "10m_magnitude_of_wind": wind_components[
                      "10m_u_component_of_wind"
                  ]
              },
              "mean",
          ),
          std=read_stats(
              stats_path,
              {
                  "10m_magnitude_of_wind": wind_components[
                      "10m_u_component_of_wind"
                  ]
              },
              "std",
          ),
          output_field="wind_speed",
          remove_inputs=True,
      ),
      transforms.Concatenate(
          input_fields=(*variables.keys(), "wind_speed"),
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      ),
  ]
  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=num_chunks,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_chunked_lens2_loader(
    date_range: tuple[str, str],
    dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    stats_path: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    member_indexer: (
        dict[str, str] | types.MappingProxyType
    ) = _LENS2_MEMBER_INDEXER,
    variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    num_chunks: int = 1,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
):
  """Creates the chunked loader for LENS2 dataset.

  Args:
    date_range: Range for the data used in the dataloader.
    dataset_path: Path for the dataset.
    stats_path: Path of the Zarr file containing the statistics.
    member_indexer: The index for the ensemble member from which the data is
      extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    variable_names: The names of the variables in the Zarr file to be included.
    shuffle: Do we shuffle the data or the samples are done sequentially.
    seed: Random seed for the random number generator.
    batch_size: The size of each contiguos chunks.
    num_chunks: Number of contiguous chunks to sample from the dataset.
    drop_remainder: Whether drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.

  Returns:
    A pygrain dataloader with the LENS2 data set.
  """

  variables = {v: member_indexer for v in variable_names}

  source = ContiguousSource(
      date_range=date_range,
      batch_size=batch_size,
      dataset_path=dataset_path,
      variables=variables,
      dims_order=["member", "time", "longitude", "latitude"],
  )
  transformations = [
      transforms.Standardize(
          input_fields=variable_names,
          mean=read_stats(stats_path, variables, "mean", use_batched=True),
          std=read_stats(stats_path, variables, "std", use_batched=True),
      ),
      transforms.Concatenate(
          input_fields=variable_names,
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  ]

  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=num_chunks,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_lens2_era5_loader(
    date_range: tuple[str, str],
    input_dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    input_stats_path: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    input_member_indexer: (
        dict[str, str] | types.MappingProxyType
    ) = _LENS2_MEMBER_INDEXER,
    input_variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    output_dataset_path: epath.PathLike = _ERA5_DATASET_PATH,
    output_stats_path: epath.PathLike = _ERA5_STATS_PATH,
    output_variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,  # pylint: disable=dangerous-default-value
    output_wind_components: (
        dict[str, dict[str, Any]] | types.MappingProxyType
    ) = _ERA5_WIND_COMPONENTS,  # pylint: disable=dangerous-default-value
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
):
  """Creates a loader for ERA5 and LENS2 aligned by date.

  Args:
    date_range: Range for the data used in the dataloader.
    input_dataset_path: Path for the input (lens2) dataset.
    input_stats_path: Path of the Zarr file containing the statistics.
    input_member_indexer: The index for the ensemble member from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    input_variable_names: The names of the variables in the Zarr file to be
      included.
    output_dataset_path: Path for the output (era5) dataset.
    output_stats_path: Path of the Zarr file containing the statistics.
    output_variables: Variables in the output dataset to be chosen.
    output_wind_components: Components of the wind speed.
    shuffle: Whether to randomly pick the data points.
    seed: Random seed for the random number generator.
    batch_size: Size of the batch.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.

  Returns:
    A data loader with the LENS2 and ERA5 data set.
  """
  input_variables = {v: input_member_indexer for v in input_variable_names}

  source = DataSource(
      date_range=date_range,
      input_dataset=input_dataset_path,
      input_variable_names=input_variable_names,
      input_member_indexer=input_member_indexer,
      output_dataset=output_dataset_path,
      output_variables=output_variables | output_wind_components,
      resample_at_nan=False,
      dims_order_input=["member", "time", "longitude", "latitude"],
      dims_order_output=["time", "longitude", "latitude", "level"],
      resample_seed=9999,
  )

  transformations = [
      transforms.StandardizeNested(
          main_field="output",
          input_fields=(*output_variables.keys(),),
          mean=read_stats(output_stats_path, output_variables, "mean"),
          std=read_stats(output_stats_path, output_variables, "std"),
      ),
      transforms.StandardizeNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          mean=read_stats(input_stats_path, input_variables, "mean"),
          std=read_stats(input_stats_path, input_variables, "std"),
      ),
  ]
  if output_wind_components:
    transformations.append(
        transforms.ComputeWindSpeedExactNested(
            main_field="output",
            u_field="10m_u_component_of_wind",
            v_field="10m_v_component_of_wind",
            speed_field="10m_magnitude_of_wind",
            mean=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "mean",
            ),
            std=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "std",
            ),
            output_field="wind_speed",
            remove_inputs=True,
        )
    )

  output_fields = (
      (*output_variables.keys(), "wind_speed")
      if output_wind_components
      else (*output_variables.keys(),)
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="output",
          input_fields=output_fields,
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  )
  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_lens2_era5_loader_chunked(
    date_range: tuple[str, str],
    input_dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    input_stats_path: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    input_member_indexer: (
        dict[str, str] | types.MappingProxyType
    ) = _LENS2_MEMBER_INDEXER,
    input_variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    output_dataset_path: epath.PathLike = _ERA5_DATASET_PATH,
    output_stats_path: epath.PathLike = _ERA5_STATS_PATH,
    output_variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,  # pylint: disable=dangerous-default-value
    output_wind_components: (
        dict[str, dict[str, Any]] | types.MappingProxyType
    ) = _ERA5_WIND_COMPONENTS,  # pylint: disable=dangerous-default-value
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
):
  """Creates a loader for ERA5 and LENS2 aligned by date.

  Args:
    date_range: Range for the data used in the dataloader.
    input_dataset_path: Path for the input (lens2) dataset.
    input_stats_path: Path of the Zarr file containing the statistics.
    input_member_indexer: The index for the ensemble member from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    input_variable_names: The names of the variables in the Zarr file to be
      included.
    output_dataset_path: Path for the output (era5) dataset.
    output_stats_path: Path of the Zarr file containing the statistics.
    output_variables: Variables in the output dataset to be chosen.
    output_wind_components: Components of the wind speed.
    shuffle: Whether to randomly pick the data points.
    seed: Random seed for the random number generator.
    batch_size: Size of the batch.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.

  Returns:
    A data loader with the LENS2 and ERA5 data set.
  """
  input_variables = {v: input_member_indexer for v in input_variable_names}

  source = DataSourceContiguous(
      date_range=date_range,
      batch_size=batch_size,
      input_dataset=input_dataset_path,
      input_variable_names=input_variable_names,
      input_member_indexer=input_member_indexer,
      output_dataset=output_dataset_path,
      output_variables=output_variables | output_wind_components,
      resample_at_nan=False,
      dims_order_input=["member", "time", "longitude", "latitude"],
      dims_order_output=["time", "longitude", "latitude", "level"],
      resample_seed=9999,
  )

  transformations = [
      transforms.StandardizeNested(
          main_field="output",
          input_fields=(*output_variables.keys(),),
          mean=read_stats(
              output_stats_path, output_variables, "mean", use_batched=True
          ),
          std=read_stats(
              output_stats_path, output_variables, "std", use_batched=True
          ),
      ),
      transforms.StandardizeNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          mean=read_stats(
              input_stats_path, input_variables, "mean", use_batched=True
          ),
          std=read_stats(
              input_stats_path, input_variables, "std", use_batched=True
          ),
      ),
  ]
  if output_wind_components:
    transformations.append(
        transforms.ComputeWindSpeedExactNested(
            main_field="output",
            u_field="10m_u_component_of_wind",
            v_field="10m_v_component_of_wind",
            speed_field="10m_magnitude_of_wind",
            mean=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "mean",
            ),
            std=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "std",
            ),
            output_field="wind_speed",
            remove_inputs=True,
        )
    )

  output_fields = (
      (*output_variables.keys(), "wind_speed")
      if output_wind_components
      else (*output_variables.keys(),)
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="output",
          input_fields=output_fields,
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  )

  transformations.append(
      transforms.RandomShuffleChunk(
          input_fields=("x_0",), batch_size=batch_size
      ),
  )

  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=1,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_ensemble_lens2_era5_loader_chunked(
    date_range: tuple[str, str],
    input_dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    input_stats_path: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    input_member_indexer: (
        tuple[dict[str, str], ...] | tuple[types.MappingProxyType, ...]
    ) = (_LENS2_MEMBER_INDEXER,),
    input_variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    output_dataset_path: epath.PathLike = _ERA5_DATASET_PATH,
    output_stats_path: epath.PathLike = _ERA5_STATS_PATH,
    output_variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,  # pylint: disable=dangerous-default-value
    output_wind_components: (
        dict[str, dict[str, Any]] | types.MappingProxyType
    ) = _ERA5_WIND_COMPONENTS,  # pylint: disable=dangerous-default-value
    shuffle: bool = False,
    random_local_shuffle: bool = True,
    batch_ot_shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    num_chunks: int = 1,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
):
  """Creates a loader for ERA5 and LENS2 aligned by date.

  Args:
    date_range: Range for the data used in the dataloader.
    input_dataset_path: Path for the input (lens2) dataset.
    input_stats_path: Path of the Zarr file containing the statistics.
    input_member_indexer: The index for the ensemble members from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    input_variable_names: The names of the variables in the Zarr file to be
      included.
    output_dataset_path: Path for the output (era5) dataset.
    output_stats_path: Path of the Zarr file containing the statistics.
    output_variables: Variables in the output dataset to be chosen.
    output_wind_components:
    shuffle: Whether to randomly pick the data points.
    random_local_shuffle: Whether to randomly shuffle the data points.
    batch_ot_shuffle: Whether to shuffle the data points in the batch, by
      solving a linear alignment problem.
    seed: Random seed for the random number generator.
    batch_size: The size of the batch.
    num_chunks: Number of chunks for each batch. Each chunk is independently
      sampled from the dataset.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.

  Returns:
    a pygrain loader with the LENS2 and ERA5 data set.
  """
  source = DataSourceContiguousEnsemble(
      date_range=date_range,
      batch_size=batch_size,
      input_dataset=input_dataset_path,
      input_variable_names=input_variable_names,
      input_member_indexer=input_member_indexer,
      output_dataset=output_dataset_path,
      output_variables=output_variables | output_wind_components,
      resample_at_nan=False,
      dims_order_input=["member", "time", "longitude", "latitude"],
      dims_order_output=["time", "longitude", "latitude", "level"],
      resample_seed=9999,
  )

  member_indexer = input_member_indexer[0]
  # Just the first one to extract the statistics.
  input_variables = {v: member_indexer for v in input_variable_names}

  transformations = [
      transforms.StandardizeNested(
          main_field="output",
          input_fields=(*output_variables.keys(),),
          mean=read_stats(
              output_stats_path, output_variables, "mean", use_batched=True
          ),
          std=read_stats(
              output_stats_path, output_variables, "std", use_batched=True
          ),
      ),
      transforms.StandardizeNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          mean=read_stats(
              input_stats_path, input_variables, "mean", use_batched=True
          ),
          std=read_stats(
              input_stats_path, input_variables, "std", use_batched=True
          ),
      ),
  ]
  if output_wind_components:
    transformations.append(
        transforms.ComputeWindSpeedExactNested(
            main_field="output",
            u_field="10m_u_component_of_wind",
            v_field="10m_v_component_of_wind",
            speed_field="10m_magnitude_of_wind",
            mean=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "mean",
            ),
            std=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "std",
            ),
            output_field="wind_speed",
            remove_inputs=True,
        )
    )

  output_fields = (
      (*output_variables.keys(), "wind_speed")
      if output_wind_components
      else (*output_variables.keys(),)
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="output",
          input_fields=output_fields,
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  )

  if random_local_shuffle:
    transformations.append(
        transforms.RandomShuffleChunk(
            input_fields=("x_0",), batch_size=batch_size
        ),
    )
  elif batch_ot_shuffle:
    transformations.append(
        transforms.BatchOT(
            input_field="x_0", output_field="x_1", batch_size=batch_size
        ),
    )

  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=num_chunks,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_ensemble_lens2_era5_loader_chunked_with_stats(
    date_range: tuple[str, str],
    input_dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    input_stats_path: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    input_member_indexer: (
        tuple[dict[str, str], ...] | tuple[types.MappingProxyType, ...]
    ) = (_LENS2_MEMBER_INDEXER,),
    input_variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    output_dataset_path: epath.PathLike = _ERA5_DATASET_PATH,
    output_stats_path: epath.PathLike = _ERA5_STATS_PATH,
    output_variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,  # pylint: disable=dangerous-default-value
    output_wind_components: (
        dict[str, dict[str, Any]] | types.MappingProxyType
    ) = _ERA5_WIND_COMPONENTS,  # pylint: disable=dangerous-default-value
    shuffle: bool = False,
    random_local_shuffle: bool = True,
    batch_ot_shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    num_chunks: int = 1,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    time_stamps: bool = False,
    overlapping_chunks: bool = False,
    num_epochs: int | None = None,
):
  """Creates a loader for ERA5 and LENS2 aligned by date.

  Args:
    date_range: Date range for the data used in the dataloader.
    input_dataset_path: Path for the input (lens2) dataset.
    input_stats_path: Path of the Zarr file containing the statistics.
    input_member_indexer: The index for the ensemble members from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    input_variable_names: Input variables to be included in the batch.
    output_dataset_path: Path for the output (era5) dataset.
    output_stats_path: Path of the Zarr file containing the statistics.
    output_variables: Variables in the output dataset to be chosen.
    output_wind_components: Variables including the wind componen to be
      included in the output.
    shuffle: Whether to shuffle the data points.
    random_local_shuffle: Whether to shuffle the data points in the batch.
    batch_ot_shuffle: Whether to shuffle the data points in the batch, by
      solving a linear alignment problem.
    seed: Random seed for the random number generator.
    batch_size: batch size for the contiguous chunk.
    num_chunks: Number of chunks from the.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to include the time stamps in the output.
    overlapping_chunks: Whether to use overlapping chunks.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
    A pygrain loader with the LENS2 and ERA5 data set.
  """
  if overlapping_chunks:
    source = DataSourceContiguousNonOverlappingEnsembleWithStats(
        date_range=date_range,
        batch_size=batch_size,
        input_dataset=input_dataset_path,
        input_variable_names=input_variable_names,
        input_member_indexer=input_member_indexer,
        input_stats_dataset=input_stats_path,
        output_dataset=output_dataset_path,
        output_variables=output_variables | output_wind_components,
        resample_at_nan=False,
        dims_order_input=["member", "time", "longitude", "latitude"],
        dims_order_output=["time", "longitude", "latitude", "level"],
        dims_order_input_stats=["member", "longitude", "latitude", "stats"],
        resample_seed=9999,
        time_stamps=time_stamps,
    )
  else:
    source = DataSourceContiguousEnsembleWithStats(
        date_range=date_range,
        batch_size=batch_size,
        input_dataset=input_dataset_path,
        input_variable_names=input_variable_names,
        input_member_indexer=input_member_indexer,
        input_stats_dataset=input_stats_path,
        output_dataset=output_dataset_path,
        output_variables=output_variables | output_wind_components,
        resample_at_nan=False,
        dims_order_input=["member", "time", "longitude", "latitude"],
        dims_order_output=["time", "longitude", "latitude", "level"],
        dims_order_input_stats=["member", "longitude", "latitude", "stats"],
        resample_seed=9999,
        time_stamps=time_stamps,
    )

  member_indexer = input_member_indexer[0]
  # Just the first one to extract the statistics.
  input_variables = {v: member_indexer for v in input_variable_names}

  transformations = [
      transforms.StandardizeNested(
          main_field="output",
          input_fields=(*output_variables.keys(),),
          mean=read_stats(
              output_stats_path, output_variables, "mean", use_batched=True
          ),
          std=read_stats(
              output_stats_path, output_variables, "std", use_batched=True
          ),
      ),
      transforms.StandardizeNestedWithStats(
          main_field="input",
          mean_field="input_mean",
          std_field="input_std",
          input_fields=(*input_variables.keys(),),
      ),
  ]
  if output_wind_components:
    transformations.append(
        transforms.ComputeWindSpeedExactNested(
            main_field="output",
            u_field="10m_u_component_of_wind",
            v_field="10m_v_component_of_wind",
            speed_field="10m_magnitude_of_wind",
            mean=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "mean",
            ),
            std=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "std",
            ),
            output_field="wind_speed",
            remove_inputs=True,
        )
    )

  output_fields = (
      (*output_variables.keys(), "wind_speed")
      if output_wind_components
      else (*output_variables.keys(),)
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="output",
          input_fields=output_fields,
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  )
  # Also concatenating the statistics.
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input_mean",
          input_fields=(*input_variables.keys(),),
          output_field="channel:mean",
          axis=-1,
          remove_inputs=True,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input_std",
          input_fields=(*input_variables.keys(),),
          output_field="channel:std",
          axis=-1,
          remove_inputs=True,
      ),
  )

  if random_local_shuffle:
    # TODO: Add the option to shuffle the statistics.
    transformations.append(
        transforms.RandomShuffleChunk(
            input_fields=("x_0", "channel:mean", "channel:std"),
            batch_size=batch_size,
        ),
    )
  elif batch_ot_shuffle:
    transformations.append(
        transforms.BatchOT(
            input_field="x_0",
            output_field="x_1",
            batch_size=batch_size,
            mean_input_field="channel:mean",
            std_input_field="channel:std",
        ),
    )

  loader = pygrain.load(
      source=source,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=num_chunks,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_ensemble_lens2_era5_loader_chunked_with_normalized_stats(
    date_range: tuple[str, str],
    input_dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    input_stats_path: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    input_mean_stats_path: epath.PathLike = _LENS2_MEAN_STATS_PATH,
    input_std_stats_path: epath.PathLike = _LENS2_STD_STATS_PATH,
    input_member_indexer: (
        tuple[dict[str, str], ...] | tuple[types.MappingProxyType, ...]
    ) = (_LENS2_MEMBER_INDEXER,),
    input_variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    output_dataset_path: epath.PathLike = _ERA5_DATASET_PATH,
    output_stats_path: epath.PathLike = _ERA5_STATS_PATH,
    output_variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,  # pylint: disable=dangerous-default-value
    output_wind_components: (
        dict[str, dict[str, Any]] | types.MappingProxyType
    ) = _ERA5_WIND_COMPONENTS,  # pylint: disable=dangerous-default-value
    shuffle: bool = False,
    random_local_shuffle: bool = True,
    batch_ot_shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    num_chunks: int = 1,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    time_stamps: bool = False,
    overlapping_chunks: bool = False,
    num_epochs: int | None = None,
):
  """Creates a loader for ERA5 and LENS2 aligned by date.

  Args:
    date_range: Date range for the data used in the dataloader.
    input_dataset_path: Path for the input (lens2) dataset.
    input_stats_path: Path of the Zarr file containing the statistics.
    input_mean_stats_path: Path of the Zarr file containing the mean statistics.
    input_std_stats_path: Path of the Zarr file containing the std statistics.
    input_member_indexer: The index for the ensemble members from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    input_variable_names: Input variables to be included in the batch.
    output_dataset_path: Path for the output (era5) dataset.
    output_stats_path: Path of the Zarr file containing the statistics.
    output_variables: Variables in the output dataset to be chosen.
    output_wind_components: Variables including the wind componen to be
      included in the output.
    shuffle: Whether to shuffle the data points.
    random_local_shuffle: Whether to shuffle the data points in the batch.
    batch_ot_shuffle: Whether to shuffle the data points in the batch, by
      solving a linear alignment problem.
    seed: Random seed for the random number generator.
    batch_size: batch size for the contiguous chunk.
    num_chunks: Number of chunks from the.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to include the time stamps in the output.
    overlapping_chunks: Whether to use overlapping chunks.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
  """

  if overlapping_chunks:
    source = DataSourceContiguousNonOverlappingEnsembleWithStats(
        date_range=date_range,
        batch_size=batch_size,
        input_dataset=input_dataset_path,
        input_variable_names=input_variable_names,
        input_member_indexer=input_member_indexer,
        input_stats_dataset=input_stats_path,
        output_dataset=output_dataset_path,
        output_variables=output_variables | output_wind_components,
        resample_at_nan=False,
        dims_order_input=["member", "time", "longitude", "latitude"],
        dims_order_output=["time", "longitude", "latitude", "level"],
        dims_order_input_stats=["member", "longitude", "latitude", "stats"],
        resample_seed=9999,
        time_stamps=time_stamps,
    )
  else:
    source = DataSourceContiguousEnsembleWithStats(
        date_range=date_range,
        batch_size=batch_size,
        input_dataset=input_dataset_path,
        input_variable_names=input_variable_names,
        input_member_indexer=input_member_indexer,
        input_stats_dataset=input_stats_path,
        output_dataset=output_dataset_path,
        output_variables=output_variables | output_wind_components,
        resample_at_nan=False,
        dims_order_input=["member", "time", "longitude", "latitude"],
        dims_order_output=["time", "longitude", "latitude", "level"],
        dims_order_input_stats=["member", "longitude", "latitude", "stats"],
        resample_seed=9999,
        time_stamps=time_stamps,
    )

  member_indexer = input_member_indexer[0]
  # Just the first one to extract the statistics.
  input_variables = {v: member_indexer for v in input_variable_names}

  transformations = [
      transforms.StandardizeNested(
          main_field="output",
          input_fields=(*output_variables.keys(),),
          mean=read_stats(
              output_stats_path, output_variables, "mean", use_batched=True
          ),
          std=read_stats(
              output_stats_path, output_variables, "std", use_batched=True
          ),
      ),
      transforms.StandardizeNestedWithStats(
          main_field="input",
          mean_field="input_mean",
          std_field="input_std",
          input_fields=(*input_variables.keys(),),
      ),
  ]
  if output_wind_components:
    transformations.append(
        transforms.ComputeWindSpeedExactNested(
            main_field="output",
            u_field="10m_u_component_of_wind",
            v_field="10m_v_component_of_wind",
            speed_field="10m_magnitude_of_wind",
            mean=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "mean",
            ),
            std=read_stats(
                output_stats_path,
                {
                    "10m_magnitude_of_wind": output_wind_components[
                        "10m_u_component_of_wind"
                    ]
                },
                "std",
            ),
            output_field="wind_speed",
            remove_inputs=True,
        )
    )

  output_fields = (
      (*output_variables.keys(), "wind_speed")
      if output_wind_components
      else (*output_variables.keys(),)
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="output",
          input_fields=output_fields,
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  )
  # Also concatenating the statistics.
  empty_dict = {len: {} for len in input_variables.keys()}
  transformations.append(
      transforms.StandardizeNested(
          main_field="input_mean",
          input_fields=(*input_variables.keys(),),
          mean=read_stats(
              input_mean_stats_path, empty_dict, "mean", use_batched=True
          ),
          std=read_stats(
              input_std_stats_path, empty_dict, "mean", use_batched=True
          ),
      ),
  )

  transformations.append(
      transforms.StandardizeNested(
          main_field="input_std",
          input_fields=(*input_variables.keys(),),
          mean=read_stats(
              input_mean_stats_path, empty_dict, "std", use_batched=True
          ),
          std=read_stats(
              input_std_stats_path, empty_dict, "std", use_batched=True
          ),
      ),
  )

  transformations.append(
      transforms.ConcatenateNested(
          main_field="input_mean",
          input_fields=(*input_variables.keys(),),
          output_field="channel:mean",
          axis=-1,
          remove_inputs=True,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input_std",
          input_fields=(*input_variables.keys(),),
          output_field="channel:std",
          axis=-1,
          remove_inputs=True,
      ),
  )

  if random_local_shuffle:
    # TODO: Add the option to shuffle the statistics.
    transformations.append(
        transforms.RandomShuffleChunk(
            input_fields=("x_0", "channel:mean", "channel:std"),
            batch_size=batch_size,
        ),
    )
  elif batch_ot_shuffle:
    transformations.append(
        transforms.BatchOT(
            input_field="x_0",
            output_field="x_1",
            batch_size=batch_size,
            mean_input_field="channel:mean",
            std_input_field="channel:std",
        ),
    )

  loader = pygrain.load(
      source=source,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=num_chunks,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_lens2_loader_chunked_with_normalized_stats(
    date_range: tuple[str, str],
    dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    stats_path: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    mean_stats_path: epath.PathLike = _LENS2_MEAN_STATS_PATH,
    std_stats_path: epath.PathLike = _LENS2_STD_STATS_PATH,
    member_indexer: (
        tuple[dict[str, str], ...] | tuple[types.MappingProxyType, ...]
    ) = (_LENS2_MEMBER_INDEXER,),
    variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    shuffle: bool = False,
    random_local_shuffle: bool = True,
    batch_ot_shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    num_chunks: int = 1,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    time_stamps: bool = False,
    num_epochs: int | None = None,
):
  """Creates a loader for LENS2 dataset with non-overlapping chunks.

  Args:
    date_range: Date range for the data used in the dataloader.
    dataset_path: Path for the input (lens2) dataset.
    stats_path: Path of the Zarr file containing the statistics.
    mean_stats_path: Path of the Zarr file containing the mean statistics.
    std_stats_path: Path of the Zarr file containing the std statistics.
    member_indexer: The index for the ensemble members from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    variable_names: Input variables to be included in the batch.
    shuffle: Whether to shuffle the data points.
    random_local_shuffle: Whether to shuffle the data points in the batch.
    batch_ot_shuffle: Whether to shuffle the data points in the batch, by
      solving a linear alignment problem.
    seed: Random seed for the random number generator.
    batch_size: batch size for the contiguous chunk.
    num_chunks: Number of chunks from the.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to include the time stamps in the output.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
    A pygrain dataloader with the LENS2 data set.
  """

  variables = {v: member_indexer for v in variable_names}

  source = DataSourceContiguousEnsembleNonOverlappingWithStatsLENS2(
      date_range=date_range,
      batch_size=batch_size,
      input_dataset=dataset_path,
      input_variable_names=variable_names,
      input_member_indexer=member_indexer,
      input_stats_dataset=stats_path,
      dims_order_input=["member", "time", "longitude", "latitude"],
      dims_order_input_stats=[
          "member",
          "longitude",
          "latitude",
          "stats",
      ],
      time_stamps=time_stamps,
  )

  # Adding the different transformations.

  # Standardizing and concatenating the input.
  transformations = [
      transforms.StandardizeNestedWithStats(
          main_field="input",
          mean_field="input_mean",
          std_field="input_std",
          input_fields=(*variables.keys(),),
      ),
      transforms.ConcatenateNested(
          main_field="input",
          input_fields=(*variables.keys(),),
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  ]

  # Standardizing the statistics (mean and std)
  empty_dict = {len: {} for len in variables.keys()}
  transformations.append(
      transforms.StandardizeNested(
          main_field="input_mean",
          input_fields=(*variables.keys(),),
          mean=read_stats(
              mean_stats_path, empty_dict, "mean", use_batched=True
          ),
          std=read_stats(std_stats_path, empty_dict, "mean", use_batched=True),
      ),
  )
  transformations.append(
      transforms.StandardizeNested(
          main_field="input_std",
          input_fields=(*variables.keys(),),
          mean=read_stats(mean_stats_path, empty_dict, "std", use_batched=True),
          std=read_stats(std_stats_path, empty_dict, "std", use_batched=True),
      ),
  )

  # Concatenating the statistics.
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input_mean",
          input_fields=(*variables.keys(),),
          output_field="channel:mean",
          axis=-1,
          remove_inputs=True,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input_std",
          input_fields=(*variables.keys(),),
          output_field="channel:std",
          axis=-1,
          remove_inputs=True,
      ),
  )

  if random_local_shuffle:
    transformations.append(
        transforms.RandomShuffleChunk(
            input_fields=("x_0", "channel:mean", "channel:std"),
            batch_size=batch_size,
        ),
    )
  elif batch_ot_shuffle:
    transformations.append(
        transforms.BatchOT(
            input_field="x_0",
            output_field="x_1",
            batch_size=batch_size,
            mean_input_field="channel:mean",
            std_input_field="channel:std",
        ),
    )

  loader = pygrain.load(
      source=source,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=num_chunks,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


def create_lens2_era5_loader_default(
    date_range: tuple[str, str],
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
):
  """Creates a loader for ERA5 and LENS2 aligned by date.

  Args:
    date_range: Range for the data used in the dataloader.
    shuffle: Whether to randomly pick the data points.
    seed: Random seed for the random number generator.
    batch_size: The size of the batch.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.

  Returns:
  """
  input_dataset_path = "/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
  input_stats_path = "/lzepedanunez/data/lens2/stats/all_variables_240x121_lonlat_1961-2000.zarr/"
  input_member_indexer = {"member": "cmip6_1001_001"}
  input_variable_names = [
      "TREFHT",
      "QREFHT",
      "Z200",
      "Z500",
      "PSL",
      "WSPDSRFAV",
  ]
  input_variables = {v: input_member_indexer for v in input_variable_names}

  output_dataset_path = "/weatherbench/datasets/era5_daily_mean/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
  output_stats_path = "/wanzy/data/era5/stats/daily_mean_240x121_all_variables_and_wind_speed_1961-2000.zarr"
  output_variables = {
      "2m_temperature": None,
      "specific_humidity": {"level": 1000},
      "geopotential": {"level": [200, 500]},
      "mean_sea_level_pressure": None,
  }
  output_wind_components = {
      "10m_u_component_of_wind": None,
      "10m_v_component_of_wind": None,
  }

  source = DataSource(
      date_range=date_range,
      input_dataset=input_dataset_path,
      input_variable_names=input_variable_names,
      input_member_indexer=input_member_indexer,
      output_dataset=output_dataset_path,
      output_variables=output_variables | output_wind_components,
      resample_at_nan=False,
      dims_order_input=["member", "time", "longitude", "latitude"],
      dims_order_output=["time", "longitude", "latitude", "level"],
      resample_seed=9999,
  )

  transformations = [
      transforms.StandardizeNested(
          main_field="output",
          input_fields=(*output_variables.keys(),),
          mean=read_stats(output_stats_path, output_variables, "mean"),
          std=read_stats(output_stats_path, output_variables, "std"),
      ),
      transforms.StandardizeNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          mean=read_stats(input_stats_path, input_variables, "mean"),
          std=read_stats(input_stats_path, input_variables, "std"),
      ),
      transforms.ComputeWindSpeedExactNested(
          main_field="output",
          u_field="10m_u_component_of_wind",
          v_field="10m_v_component_of_wind",
          speed_field="10m_magnitude_of_wind",
          mean=read_stats(
              output_stats_path,
              {
                  "10m_magnitude_of_wind": output_wind_components[
                      "10m_u_component_of_wind"
                  ]
              },
              "mean",
          ),
          std=read_stats(
              output_stats_path,
              {
                  "10m_magnitude_of_wind": output_wind_components[
                      "10m_u_component_of_wind"
                  ]
              },
              "std",
          ),
          output_field="wind_speed",
          remove_inputs=True,
      ),
      transforms.ConcatenateNested(
          main_field="output",
          input_fields=(*output_variables.keys(), "wind_speed"),
          output_field="x_1",
          axis=-1,
          remove_inputs=True,
      ),
      transforms.ConcatenateNested(
          main_field="input",
          input_fields=(*input_variables.keys(),),
          output_field="x_0",
          axis=-1,
          remove_inputs=True,
      ),
  ]
  loader = pygrain.load(
      source=source,
      num_epochs=None,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return loader


class DualLens2Era5Dataset:
  """A dataset that combines era5 and lens2 datasets.

  Attributes:
    era5_loader: ERA5 data loader.
    lens2_loader: LENS2 data loader.
  """

  def __init__(self, era5_loader, lens2_loader):
    self.era5_loader = era5_loader
    self.lens2_loader = lens2_loader

  def __iter__(self):
    return self.iter()

  def iter(self):
    era5_iter = iter(self.era5_loader)
    lens2_iter = iter(self.lens2_loader)

    while True:
      era5_batch = next(era5_iter)
      lens2_batch = next(lens2_iter)

      yield era5_batch | lens2_batch


class DualChunkedLens2Era5Dataset:
  """A dataset that combines the chunked era5 and lens2 datasets.

  Attributes:
    era5_loader: ERA5 data loader.
    lens2_loader: LENS2 data loader.
    _tree_reshape: Internal reshaping function.
  """

  def __init__(self, era5_loader, lens2_loader, jit_reshape=False):
    self.era5_loader = era5_loader
    self.lens2_loader = lens2_loader

    # Defining the reshaping function.
    def _tree_reshape(dict_x):
      return jax.tree_util.tree_map(
          lambda x: x.reshape((-1,) + x.shape[2:]), dict_x
      )

    if jit_reshape:
      self._tree_reshape = jax.jit(_tree_reshape, device=jax.devices("cpu")[0])
    else:
      self._tree_reshape = _tree_reshape

  def __iter__(self):
    return self.iter()

  def iter(self):
    era5_iter = iter(self.era5_loader)
    lens2_iter = iter(self.lens2_loader)

    while True:
      era5_batch = next(era5_iter)
      lens2_batch = next(lens2_iter)
      yield self._tree_reshape(era5_batch | lens2_batch)


class AlignedChunkedLens2Era5Dataset:
  """Wrapper that combines the chunked era5 and lens2 datasets.

  We take the chunks and we resize them to the correct dimensions. This comes
  from the fact that the batch size is composed of number of chunks and the size
  of each chunk.

  Attributes:
    loader: Combined LENS2-ERA5 data loader.
    _tree_reshape: Internal reshaping function.
  """

  def __init__(self, loader, jit_reshape=False):
    self.loader = loader

    # Defining the reshaping function.
    def _tree_reshape(dict_x):
      return jax.tree_util.tree_map(
          lambda x: x.reshape((-1,) + x.shape[2:]), dict_x
      )  # similar to jnp.squeeze

    if jit_reshape:
      self._tree_reshape = jax.jit(_tree_reshape, device=jax.devices("cpu")[0])
    else:
      self._tree_reshape = _tree_reshape

  def __iter__(self):
    return self.iter()

  def iter(self):
    iter_lens2_era5 = iter(self.loader)

    while True:
      try:
        batch = next(iter_lens2_era5)
        yield self._tree_reshape(batch)
      except StopIteration:
        break
