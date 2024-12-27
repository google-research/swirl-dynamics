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

import abc
from collections.abc import Callable, Mapping, Sequence
import types
from typing import Any, Literal, SupportsIndex

from etils import epath
import grain.python as pygrain
import jax
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.projects.debiasing.rectified_flow import pygrain_transforms as transforms
import xarray_tensorstore as xrts


Array = jax.Array
PyTree = Any
DynamicsFn = Callable[[Array, Array, PyTree], Array]

_ERA5_VARIABLES = types.MappingProxyType({
    "2m_temperature": None,
    "specific_humidity": {"level": 1000},
    "mean_sea_level_pressure": None,
    "10m_magnitude_of_wind": None,
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


def maybe_expand_dims(
    array: np.ndarray,
    allowed_dims: tuple[int, ...],
    trigger_expand_dims: int,
    axis: int = -1,
) -> np.ndarray:
  """Expands the dimensions of a numpy array if necessary.

  Args:
    array: The numpy array to be possibly expanded.
    allowed_dims: The dimensions that the array can have, raise an error
      otherwise.
    trigger_expand_dims: The dimension that triggers the expansion.
    axis: The axis in which the extra dimension is added.

  Returns:
    The array possibly expanded if its dimension is trigger_expand_dims.
  """
  ndim = array.ndim
  if ndim not in allowed_dims:
    raise ValueError(
        f"The array has {ndim} dimensions, but it should have one of the"
        f" dimensions {allowed_dims}"
    )
  if ndim == trigger_expand_dims:
    array = np.expand_dims(array, axis=axis)
  return array


class SourceInMemoryHDF5(pygrain.RandomAccessDataSource):
  """Source class for a HDF5 file.

  Here we assume the store data is split in subsets with keys : "train",
  "eval", and "test". Inside each, we have the field "u", which is either a
  5-tensor with shape [num_trajectories, num_time_stamps, nx, ny, num_fields],
  or a 4-tensor with shape [num_snapshots, nx, ny, num_fields].
  We also assume that the statistics are only computed from the "train" split.

  Attributes:
    source: numpy array with the data.
    normalize_stats: dictionary with the mean and std statistics of the data.
  """

  def __init__(
      self,
      dataset_path: str,
      split: Literal["train", "eval", "test"],
      spatial_downsample_factor: int = 1,
  ):
    """Load pre-computed trajectories stored in hdf5 file.

    Args:
      dataset_path: Absolute path to dataset file.
      split: Data split, of the following: train, eval, or test.
      spatial_downsample_factor: reduce spatial resolution by factor of x.

    Returns:
      loader, stats (optional): Tuple of dataloader and dictionary containing
                                mean and std stats (if normalize=True, else dict
                                contains NoneType values).
    """
    # Reads data from the hdf5 file.
    snapshots = hdf5_utils.read_single_array(
        dataset_path,
        f"{split}/u",
    )

    # If the data is given by trajectories, we scramble the time stamps.
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

    self.source = snapshots
    self.normalize_stats = {"mean": mean, "std": std}

  def __len__(self) -> int:
    """Returns the number of samples in the source."""
    return self.source.shape[0]

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, np.ndarray]:
    """Returns the data record for a given record key."""
    idx = record_key.__index__()

    if idx >= self.__len__():
      raise IndexError("Index out of range.")
    # here we return a dictionary with "u"
    return {"u": self.source[idx]}


class UnpairedDataLoader:
  """Unpaired dataloader for loading samples from two distributions."""

  def __init__(
      self,
      batch_size: int,
      dataset_path_a: str,
      dataset_path_b: str,
      seed: int,
      split: Literal["train", "eval", "test"],
      spatial_downsample_factor_a: int = 1,
      spatial_downsample_factor_b: int = 1,
      normalize: bool = False,
      normalize_stats_a: dict[str, Array] | None = None,
      normalize_stats_b: dict[str, Array] | None = None,
      drop_remainder: bool = True,
      worker_count: int = 0,
  ):

    loader, normalize_stats_a = create_loader_from_hdf5(
        batch_size=batch_size,
        dataset_path=dataset_path_a,
        seed=seed,
        split=split,
        spatial_downsample_factor=spatial_downsample_factor_a,
        normalize=normalize,
        normalize_stats=normalize_stats_a,
        output_name="x_0",
        drop_remainder=drop_remainder,
        worker_count=worker_count,
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
        output_name="x_1",
        drop_remainder=drop_remainder,
        worker_count=worker_count,
    )
    self.loader_b = iter(loader)

    self.normalize_stats_a = normalize_stats_a
    self.normalize_stats_b = normalize_stats_b

  def __iter__(self):
    return self

  def __next__(self) -> dict[str, Array]:

    b = next(self.loader_b)
    a = next(self.loader_a)

    # Return dictionary with keys "x_0" and "x_1".
    return {**a, **b}


def create_loader_from_hdf5(
    batch_size: int,
    dataset_path: str,
    split: Literal["train", "eval", "test"],
    seed: int = 999,
    shuffle: bool = True,
    spatial_downsample_factor: int = 1,
    normalize: bool = True,
    normalize_stats: dict[str, np.ndarray] | None = None,
    output_name: str | None = None,
    drop_remainder: bool = True,
    worker_count: int = 0,
) -> tuple[pygrain.DataLoader, dict[str, np.ndarray] | None]:
  """Load pre-computed trajectories dumped to hdf5 file.

  Args:
    batch_size: Batch size returned by dataloader. If set to -1, use entire
      dataset size as batch_size.
    dataset_path: Absolute path to dataset file.
    split: Data split - "train", "eval", or "test".
    seed: Random seed to be used in data sampling.
    shuffle: Whether to randomly shuffle the data.
    spatial_downsample_factor: reduce spatial resolution by factor of x.
    normalize: Flag for adding data normalization (subtact mean divide by std.).
    normalize_stats: Dictionary with mean and std stats to avoid recomputing, or
      if they need to be computed from a different dataset.
    output_name: Name of the output feature in the dictionary.
    drop_remainder: Flag for dropping the last batch if it is not complete.
    worker_count: Number of workers to use in the dataloader.

  Returns:
    loader, stats (optional): Tuple of dataloader and dictionary containing
                              mean and std stats (if normalize=True, else dict
                              contains NoneType values).
  """

  # Creates the source file, which is a random access file wrapping a numpy
  # array
  source = SourceInMemoryHDF5(
      dataset_path,
      split=split,
      spatial_downsample_factor=spatial_downsample_factor,
  )

  if normalize_stats is None:
    normalize_stats = source.normalize_stats

  if "mean" not in normalize_stats or "std" not in normalize_stats:
    raise ValueError(
        "The normalize_stats dictionary should contain keys 'mean' and 'std'."
    )

  transformations = []
  if normalize:
    transformations.append(
        transforms.Standardize(
            input_fields=["u",],
            mean={"u": normalize_stats["mean"]},
            std={"u": normalize_stats["std"]},
        )
    )

  if output_name is not None and output_name != "u":
    if not isinstance(output_name, str):
      raise ValueError(
          "The output_name should be a string, but it is a ",
          type(output_name),
      )
    transformations.append(
        transforms.SelectAs(
            select_features=["u",],
            as_features=[output_name,]
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
  return loader, normalize_stats


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


class CommonSource(abc.ABC):
  """A common class for both single and contiguous data sources."""

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the number of samples in each trajectory or ensamble member."""
    pass

  @abc.abstractmethod
  def _compute_len(self, *args) -> int:
    """Compute the length of the data source."""
    pass

  @abc.abstractmethod
  def _compute_time_idx(self, idx: int) -> int | tuple[int, ...] | slice:
    """Compute the time index for the data arrays."""
    pass

  @abc.abstractmethod
  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    """Checks the dimensions and expands last one if needed."""
    pass

  def __init__(
      self,
      date_range: tuple[str, str],
      dataset_path: epath.PathLike,
      variables: Mapping[str, Mapping[str, Any] | None],
      dims_order: Sequence[str] | None = None,
      time_stamps: bool = False,
  ) -> None:
    """Data source constructor with all the shared logic.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      dataset_path: The path of a zarr dataset containing the data.
      variables: The variables to yield from the dataset.
      dims_order: Order of the variables for the dataset.
      time_stamps: Whether to add the time stamps to the samples.
    """
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)
    ds = xrts.open_zarr(dataset_path).sel(time=slice(*date_range))

    if dims_order:
      ds = ds.transpose(*dims_order)
    self._data_arrays = {}

    for v, indexers in variables.items():
      self._data_arrays[v] = ds[v].sel(indexers)

    self._time_stamp = time_stamps
    self._date_range = date_range
    self._time_array = xrts.read(ds["time"]).data

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, np.ndarray]:
    """Returns the data record for a given key."""
    idx = record_key.__index__()
    if not idx < self.__len__():
      raise ValueError(f"Index out of range: {idx} / {self.__len__() - 1}")

    element = {}
    # Computes the time index for the data arrays using the abstract method.
    time_idx = self._compute_time_idx(idx)

    for v, da in self._data_arrays.items():
      array = xrts.read(da.isel(time=time_idx)).data
      element[v] = self._maybe_expands_dims(array)

      if self._time_stamp:
        element["time_stamp"] = self._time_array[time_idx]
    return element


class SingleSource(CommonSource):
  """A source class loading a single sample from a given dataset.

  Each sample has the shape [longitude, latitude, num_fields], where the latest
  is given by the number of fields in the 'variables' argument in the
  constructor.
  """

  def __len__(self) -> int:
    return self._len

  def _compute_len(self, _time_array: np.ndarray) -> int:
    return len(_time_array)

  def _compute_time_idx(self, idx: int) -> int:
    return idx

  def _maybe_expands_dims(self, x: np.ndarray) ->  np.ndarray:
    return maybe_expand_dims(x, allowed_dims=(2, 3), trigger_expand_dims=2)

  def __init__(
      self,
      date_range: tuple[str, str],
      dataset_path: epath.PathLike,
      variables: Mapping[str, Mapping[str, Any] | None],
      dims_order: Sequence[str] | None = None,
      time_stamps: bool = False,
  ):
    """Data source constructor."""
    super().__init__(
        date_range, dataset_path, variables, dims_order, time_stamps
    )
    self._len = self._compute_len(self._time_array)


class ContiguousSource(CommonSource):
  """Class to extract a chunk (overlapping) from a given dataset.

  Each sample has the shape [chunk_size, longitude, latitude, num_fields],
  where the latest is given by the number of fields in the 'variables' argument
  in the constructor. Here, the longitude and latitude dimensions can be swaped
  by setting the 'dims_order' argument in the constructor. In addition, the
  sample is extracted from the time index [idx, idx + chunk_size).
  """

  def __len__(self) -> int:
    return self._len

  def _compute_len(self, _time_array: np.ndarray, chunk_size: int) -> int:
    return len(_time_array) - chunk_size + 1

  def _compute_time_idx(self, idx: int) -> slice:
    return slice(idx, idx + self._chunk_size)

  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    return maybe_expand_dims(x, allowed_dims=(3, 4), trigger_expand_dims=3)

  def __init__(
      self,
      date_range: tuple[str, str],
      chunk_size: int,
      dataset_path: epath.PathLike,
      variables: Mapping[str, Mapping[str, Any] | None],
      dims_order: Sequence[str] | None = None,
      time_stamps: bool = False,
  ):
    """Data source constructor."""
    super().__init__(
        date_range, dataset_path, variables, dims_order, time_stamps
    )
    self._chunk_size = chunk_size
    self._len = self._compute_len(self._time_array, chunk_size)


class DataSourceContiguousEnsembleWithStats:
  """A data source that loads paired daily ERA5- with ensemble LENS2 data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      chunk_size: int,
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
      overlapping_chunks: bool = True,
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      chunk_size: Size of the chunk.
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
      overlapping_chunks: Whether to sample overlapping chunks.
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
    if overlapping_chunks:
      self._len_time = (
          np.min([input_ds.dims["time"], output_ds.dims["time"]]) - chunk_size
      )
    else:
      # Each chunk is fixed so they don't overlap.
      self._len_time = (
          np.min([input_ds.dims["time"], output_ds.dims["time"]]) // chunk_size
      )
    self._len = len(self._indexes) * self._len_time
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps
    self._chunk_size = chunk_size
    self._overlapping_chunks = overlapping_chunks

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

    if self._overlapping_chunks:
      time_slice = slice(idx_time, idx_time + self._chunk_size)
    else:
      time_slice = slice(
          idx_time * self._chunk_size, (idx_time + 1) * self._chunk_size
      )

    sample_input, mean_input, std_input = {}, {}, {}
    # sample output doens't have stats. It is normalized at the loader level.
    sample_output = {}

    for var_key, da in self._input_arrays.items():
      array = xrts.read(
          da[member].isel(time=time_slice)
      ).data
      sample_input[var_key] = maybe_expand_dims(array, (3, 4), 3)

    for var_key, da in self._input_stats_arrays.items():
      mean_array = xrts.read(da[member].sel(stats="mean")).data

      # As there is no time dimension yet the mean and std are either
      # 2- or 3-tensors.
      mean_array = maybe_expand_dims(mean_array, (2, 3), 2)
      mean_input[var_key] = np.tile(
          mean_array, (self._chunk_size,) + (1,) * mean_array.ndim
      )

      std_array = xrts.read(da[member].sel(stats="std")).data
      std_array = maybe_expand_dims(std_array, (2, 3), 2)
      std_input[var_key] = np.tile(
          std_array, (self._chunk_size,) + (1,) * std_array.ndim
      )

    item["input"] = sample_input
    item["input_mean"] = mean_input
    item["input_std"] = std_input

    for var_key, da in self._output_arrays.items():
      array = xrts.read(
          da.isel(time=time_slice)
      ).data
      sample_output[var_key] = maybe_expand_dims(array, (3, 4), 3)

    item["output"] = sample_output

    if self._time_stamps:
      item["input_time_stamp"] = self._input_time_array[time_slice]
      item["output_time_stamp"] = self._output_time_array[time_slice]

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

    sample_input, mean_input, std_input = {}, {}, {}

    for var_key, da in self._input_arrays.items():
      array = xrts.read(
          da[member].isel(
              time=slice(
                  idx_time * self._batch_size, (idx_time + 1) * self._batch_size
              )
          )
      ).data

      sample_input[var_key] = maybe_expand_dims(array, (3, 4), 3)

    for var_key, da in self._input_stats_arrays.items():
      mean_array = xrts.read(da[member].sel(stats="mean")).data
      # There is no time dimension yet.
      mean_array = maybe_expand_dims(mean_array, (2, 3), 2)
      mean_input[var_key] = np.tile(
          mean_array, (self._batch_size,) + (1,) * mean_array.ndim
      )

      std_array = xrts.read(da[member].sel(stats="std")).data
      std_array = maybe_expand_dims(std_array, (2, 3), 2)
      std_input[var_key] = np.tile(
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
      variables=variables,
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
  transformations.append(
      transforms.Concatenate(
          input_fields=(*variables.keys(),),
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


def create_chunked_era5_loader(
    date_range: tuple[str, str],
    dataset_path: epath.PathLike = _ERA5_DATASET_PATH,  # pylint: disable=dangerous-default-value
    stats_path: epath.PathLike = _ERA5_STATS_PATH,  # pylint: disable=dangerous-default-value
    variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,
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
    dataset_path: Path for the dataset.
    stats_path: Path of the Zarr file containing the statistics.
    variables: The names of the variables in the Zarr file to be included.
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

  if batch_size % num_chunks != 0:
    raise ValueError(
        f"Batch size ({batch_size}) must be a multiple of the number of chunks "
        f"({num_chunks})."
    )

  chunk_size = batch_size // num_chunks

  source = ContiguousSource(
      date_range=date_range,
      dataset_path=dataset_path,
      chunk_size=chunk_size,
      variables=variables,
      dims_order=["time", "longitude", "latitude", "level"],
  )

  transformations = [
      transforms.Standardize(
          input_fields=variables.keys(),
          mean=read_stats(stats_path, variables, "mean", use_batched=True),
          std=read_stats(stats_path, variables, "std", use_batched=True),
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

  if batch_size % num_chunks != 0:
    raise ValueError(
        f"Batch size ({batch_size}) must be a multiple of the number of chunks "
        f"({num_chunks})."
    )

  chunk_size = batch_size // num_chunks
  variables = {v: member_indexer for v in variable_names}

  source = ContiguousSource(
      date_range=date_range,
      chunk_size=chunk_size,
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
  source = DataSourceContiguousEnsembleWithStats(
      date_range=date_range,
      chunk_size=batch_size,
      input_dataset=input_dataset_path,
      input_variable_names=input_variable_names,
      input_member_indexer=input_member_indexer,
      input_stats_dataset=input_stats_path,
      output_dataset=output_dataset_path,
      output_variables=output_variables,
      resample_at_nan=False,
      dims_order_input=["member", "time", "longitude", "latitude"],
      dims_order_output=["time", "longitude", "latitude", "level"],
      dims_order_input_stats=["member", "longitude", "latitude", "stats"],
      overlapping_chunks=overlapping_chunks,
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

  transformations.append(
      transforms.ConcatenateNested(
          main_field="output",
          input_fields=(*output_variables.keys(),),
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
    # Only input data is shuffled, along with the input statistics.
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
    shuffle: bool = False,
    random_local_shuffle: bool = True,
    batch_ot_shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    num_chunks: int = 1,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    time_stamps: bool = False,
    normalize_stats: bool = True,
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
    normalize_stats: Whether to normalize the stats.
    overlapping_chunks: Whether to use overlapping chunks.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
  """
  chunk_size = batch_size // num_chunks

  source = DataSourceContiguousEnsembleWithStats(
      date_range=date_range,
      chunk_size=chunk_size,
      input_dataset=input_dataset_path,
      input_variable_names=input_variable_names,
      input_member_indexer=input_member_indexer,
      input_stats_dataset=input_stats_path,
      output_dataset=output_dataset_path,
      output_variables=output_variables,
      resample_at_nan=False,
      dims_order_input=["member", "time", "longitude", "latitude"],
      dims_order_output=["time", "longitude", "latitude", "level"],
      dims_order_input_stats=["member", "longitude", "latitude", "stats"],
      overlapping_chunks=overlapping_chunks,
      resample_seed=9999,
      time_stamps=time_stamps,
  )

  member_indexer = input_member_indexer[0]
  # Just the first one to extract the statistics.
  input_variables = {v: member_indexer for v in input_variable_names}

  transformations = [
      # We don't load the stats of ERA5 in the source so we read from a file the
      # statistics, and normalize the output.
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
      # For the input, the statistics are already in the source.
      transforms.StandardizeNestedWithStats(
          main_field="input",
          mean_field="input_mean",
          std_field="input_std",
          input_fields=(*input_variables.keys(),),
      ),
  ]

  transformations.append(
      transforms.ConcatenateNested(
          main_field="output",
          input_fields=(*output_variables.keys(),),
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
  # Also concatenating and normalizing the statistics.

  if normalize_stats:
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
    # TODO: Add the option to shuffle the statistics in the case
    # that the statistics are taken from different members. In this case is fine
    # as the chunks are all coming from the same LENS2 member.
    transformations.append(
        transforms.RandomShuffleChunk(
            input_fields=("x_0", "channel:mean", "channel:std"),
            batch_size=chunk_size,
        ),
    )
  elif batch_ot_shuffle:
    transformations.append(
        transforms.BatchOT(
            input_field="x_0",
            output_field="x_1",
            batch_size=chunk_size,
            mean_input_field="channel:mean",
            std_input_field="channel:std",
        ),
    )

  # Performs the batching, the size of the leaves of each batch is given by
  # [num_chunks, chunk_size, lon, lat, channel].
  transformations.append(
      pygrain.Batch(
          batch_size=num_chunks, drop_remainder=drop_remainder
      )
  )
  # Reshapes the batch to [batch_size, lon, lat, channel].
  transformations.append(transforms.ReshapeBatch())

  sampler = pygrain.IndexSampler(
      num_records=len(source),
      shuffle=shuffle,
      seed=seed,
      num_epochs=num_epochs,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
  )
  loader = pygrain.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=transformations,
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

  This dataloader is mainly used for evaluation and inference.

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

  # Performs the batching as the size of the leaves of each batch is given by
  # [num_chunks, chunk_size, lon, lat, channel]. We need to reshape the batch
  # to [batch_size, lon, lat, channel].
  transformations.append(
      pygrain.Batch(
          batch_size=num_chunks, drop_remainder=drop_remainder
      )
  )
  # Reshapes the batch to [batch_size, lon, lat, channel].
  transformations.append(transforms.ReshapeBatch())

  sampler = pygrain.IndexSampler(
      num_records=len(source),
      shuffle=shuffle,
      seed=seed,
      num_epochs=num_epochs,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
  )
  loader = pygrain.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=transformations,
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
