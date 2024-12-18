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

"""Class with the data loaders the climatology-based models."""
# TODO encapsulate the functionality and streamline the code.

from collections.abc import Callable, Mapping, Sequence
import types
from typing import Any, Literal, SupportsIndex

from etils import epath
import grain.python as pygrain
import jax
import numpy as np
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
_ERA5_DATASET_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = "/wanzy/data/era5/selected_variables/climat/1p5deg_dailymean_7vars_windspeed_clim_daily_1961_to_2000_31_dw.zarr"

# Interpolated dataset to match the resolution of the ERA5 data set.
_LENS2_DATASET_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
_LENS2_STATS_PATH = "/wanzy/data/lens2/climat/lens2_240x121_lonlat_clim_daily_1961_to_2000_31_dw.zarr"
# pylint: enable=line-too-long

# For training we still use the tuple of dictionaries.
_LENS2_MEMBER_INDEXER = (
    {"member": "cmip6_1001_001"},
    {"member": "cmip6_1021_002"},
    {"member": "cmip6_1041_003"},
    {"member": "cmip6_1061_004"},
)
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV")

_LENS2_MEAN_CLIMATOLOGY_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/lens2/stats/scratch/mean_lens2_240x121_lonlat_clim_daily_1961_to_2000.zarr"
_LENS2_STD_CLIMATOLOGY_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/lens2/stats/scratch/std_lens2_240x121_lonlat_clim_daily_1961_to_2000.zarr"


def read_stats_simple(
    dataset: epath.PathLike,
    variables_names: Sequence[str],
    field: Literal["mean", "std"],
) -> dict[str, np.ndarray]:
  """Reads variables from a zarr dataset and returns as a dict of ndarrays.

  This function is used to read the statistics of the climatologies, and it
  relies in the naming convention of the fields to distinguish between the
  mean and the std. The name of the variable contains the mean, and the std
  is named by adding the suffix "_std".

  Args:
    dataset: The path of the zarr dataset.
    variables_names: The variables to read.
    field: The field to read, either mean or std.

  Returns:
    A dictionary of variables and their statistics.
  """
  ds = xrts.open_zarr(dataset)
  out = {}
  for var in variables_names:
    var_idx = var+"_std" if field == "std" else var
    stats = ds[var_idx].to_numpy()
    assert stats.ndim == 2 or stats.ndim == 3
    stats = np.expand_dims(stats, axis=-1) if stats.ndim == 2 else stats
    out[var] = stats
  return out


class DataSourceEnsembleWithClimatology:
  """A data source that loads paired daily ERA5- with ensemble LENS2 data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      input_dataset: epath.PathLike,
      input_variable_names: Sequence[str],
      input_member_indexer: tuple[Mapping[str, Any], ...],
      input_climatology: epath.PathLike,
      output_dataset: epath.PathLike,
      output_variables: Mapping[str, Any],
      output_climatology: epath.PathLike,
      dims_order_input: Sequence[str] | None = None,
      dims_order_output: Sequence[str] | None = None,
      dims_order_input_stats: Sequence[str] | None = None,
      dims_order_output_stats: Sequence[str] | None = None,
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variable_names: The variables to yield from the input dataset.
      input_member_indexer: The name of the ensemble member to sample from, it
        should be tuple of dictionaries with the key "member" and the value the
        name of the member, to adhere to xarray formating, For example:
        [{"member": "cmip6_1001_001"}, {"member": "cmip6_1021_002"}, ...]
      input_climatology: The path of a zarr dataset containing the input
        statistics.
      output_dataset: The path of a zarr dataset containing the output data.
      output_variables: The variables to yield from the output dataset.
      output_climatology: The path of a zarr dataset containing the output
        statistics.
      dims_order_input: Order of the variables for the input.
      dims_order_output: Order of the variables for the output.
      dims_order_input_stats: Order of the variables for the input statistics.
      dims_order_output_stats: Order of the variables for the output statistics.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Wheter to add the time stamps to the samples.
    """

    # Using lens as input, they need to be modified.
    input_variables = {v: input_member_indexer for v in input_variable_names}

    # Computing the date_range
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    # Open the datasets
    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    output_ds = xrts.open_zarr(output_dataset).sel(time=slice(*date_range))
    # These contain the climatologies
    input_stats_ds = xrts.open_zarr(input_climatology)
    output_stats_ds = xrts.open_zarr(output_climatology)

    # Transpose the datasets if necessary
    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_output:
      output_ds = output_ds.transpose(*dims_order_output)
    if dims_order_output_stats:
      output_stats_ds = output_stats_ds.transpose(*dims_order_output_stats)
    if dims_order_input_stats:
      input_stats_ds = input_stats_ds.transpose(*dims_order_input_stats)

    # selecting the input_arrays
    self._input_arrays = {}
    for v, indexers in input_variables.items():
      self._input_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_arrays[v][idx] = input_ds[v].sel(index)

    # Building the dictionary of xarray datasets to be used for the climatology.
    # Climatological mean of LENS2.
    self._input_mean_arrays = {}
    for v, indexers in input_variables.items():
      self._input_mean_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_mean_arrays[v][idx] = input_stats_ds[v].sel(index)

    # Climatological std of LENS2.
    self._input_std_arrays = {}
    for v, indexers in input_variables.items():
      self._input_std_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_std_arrays[v][idx] = input_stats_ds[v + "_std"].sel(index)

    # Build the output arrays for the different output variables.
    self._output_arrays = {}
    for v, indexers in output_variables.items():
      self._output_arrays[v] = output_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    # Build the output arrays for the different output variables.
    # Climatological mean of ERA5.
    self._output_mean_arrays = {}
    for v, indexers in output_variables.items():
      self._output_mean_arrays[v] = output_stats_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    # Climatological std of ERA5.
    self._output_std_arrays = {}
    for v, indexers in output_variables.items():
      self._output_std_arrays[v] = output_stats_ds[v + "_std"].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    self._output_coords = output_ds.coords

    # The times can be slightly off due to the leap years.
    # Member index.
    self._indexes = [ind["member"] for ind in input_member_indexer]
    self._len_time = np.min([input_ds.dims["time"], output_ds.dims["time"]])
    self._len = len(self._indexes) * self._len_time
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps

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

    # Checking the index for the member and time (of the year)
    idx_member = idx // self._len_time
    idx_time = idx % self._len_time

    member = self._indexes[idx_member]
    date = self._input_time_array[idx_time]
    # computing day of the year.
    dayofyear = (
        date - np.datetime64(str(date.astype("datetime64[Y]")))
    ) / np.timedelta64(1, "D") + 1
    if dayofyear <= 0 or dayofyear > 366:
      raise ValueError(f"Invalid day of the year: {dayofyear}")

    sample_input = {}
    mean_input = {}
    std_input = {}
    mean_output = {}
    std_output = {}

    for v, da in self._input_arrays.items():

      array = xrts.read(da[member].sel(time=date)).data

      # Array is either two-or three-dimensional.
      assert array.ndim == 2 or array.ndim == 3
      sample_input[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 2 else array
      )

    for v, da in self._input_mean_arrays.items():
      mean_array = xrts.read(da[member].sel(dayofyear=int(dayofyear))).data
      mean_input[v] = (
          np.expand_dims(mean_array, axis=-1)
          if mean_array.ndim == 2
          else mean_array
      )

    for v, da in self._input_std_arrays.items():
      std_array = xrts.read(da[member].sel(dayofyear=int(dayofyear))).data
      std_input[v] = (
          np.expand_dims(std_array, axis=-1)
          if std_array.ndim == 2
          else std_array
      )

    item["input"] = sample_input
    item["input_mean"] = mean_input
    item["input_std"] = std_input

    sample_output = {}
    # TODO encapsulate these functions.
    for v, da in self._output_arrays.items():
      array = xrts.read(da.sel(time=date)).data
      assert array.ndim == 2 or array.ndim == 3
      sample_output[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 2 else array
      )

    for v, da in self._output_mean_arrays.items():
      mean_array = xrts.read(da.sel(dayofyear=int(dayofyear))).data
      assert mean_array.ndim == 2 or mean_array.ndim == 3
      mean_output[v] = (
          np.expand_dims(mean_array, axis=-1)
          if mean_array.ndim == 2
          else mean_array
      )

    for v, da in self._output_std_arrays.items():
      std_array = xrts.read(da.sel(dayofyear=int(dayofyear))).data
      # Adds an extra dimension is
      assert std_array.ndim == 2 or std_array.ndim == 3
      std_output[v] = (
          np.expand_dims(std_array, axis=-1)
          if std_array.ndim == 2
          else std_array
      )

    item["output"] = sample_output
    item["output_mean"] = mean_output
    item["output_std"] = std_output

    if self._time_stamps:
      item["input_time_stamp"] = date
      item["input_member"] = member

    return item

  def get_output_coords(self):
    """Returns the coordinates of the output dataset."""
    return self._output_coords


class DataSourceEnsembleWithClimatologyInference:
  """An inferece data source that loads ensemble LENS2 data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      input_dataset: epath.PathLike,
      input_variable_names: Sequence[str],
      input_member_indexer: tuple[Mapping[str, Any], ...],
      input_climatology: epath.PathLike,
      output_variables: Mapping[str, Any],
      output_climatology: epath.PathLike,
      dims_order_input: Sequence[str] | None = None,
      dims_order_input_stats: Sequence[str] | None = None,
      dims_order_output_stats: Sequence[str] | None = None,
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      time_stamps: bool = False,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variable_names: The variables to yield from the input dataset.
      input_member_indexer: The name of the ensemble member to sample from, it
        should be tuple of dictionaries with the key "member" and the value the
        name of the member, to adhere to xarray formating, For example:
        [{"member": "cmip6_1001_001"}, {"member": "cmip6_1021_002"}, ...]
      input_climatology: The path of a zarr dataset containing the input
        statistics.
      output_variables: The variables to yield from the output dataset.
      output_climatology: The path of a zarr dataset containing the output
        statistics.
      dims_order_input: Order of the variables for the input.
      dims_order_input_stats: Order of the variables for the input statistics.
      dims_order_output_stats: Order of the variables for the output statistics.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Wheter to add the time stamps to the samples.
    """

    # Using lens as input, they need to be modified.
    input_variables = {v: input_member_indexer for v in input_variable_names}

    # Computing the date_range
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    # Open the datasets
    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    # These contain the climatologies
    input_stats_ds = xrts.open_zarr(input_climatology)
    output_stats_ds = xrts.open_zarr(output_climatology)

    # Transpose the datasets if necessary
    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_output_stats:
      output_stats_ds = output_stats_ds.transpose(*dims_order_output_stats)
    if dims_order_input_stats:
      input_stats_ds = input_stats_ds.transpose(*dims_order_input_stats)

    # selecting the input_arrays
    self._input_arrays = {}
    for v, indexers in input_variables.items():
      self._input_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_arrays[v][idx] = input_ds[v].sel(index)

    # Building the dictionary of xarray datasets to be used for the climatology.
    # Climatological mean of LENS2.
    self._input_mean_arrays = {}
    for v, indexers in input_variables.items():
      self._input_mean_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_mean_arrays[v][idx] = input_stats_ds[v].sel(index)

    # Climatological std of LENS2.
    self._input_std_arrays = {}
    for v, indexers in input_variables.items():
      self._input_std_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_std_arrays[v][idx] = input_stats_ds[v + "_std"].sel(index)

    # Build the output arrays for the different output variables.
    # Climatological mean of ERA5.
    self._output_mean_arrays = {}
    for v, indexers in output_variables.items():
      self._output_mean_arrays[v] = output_stats_ds[v].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    # Climatological std of ERA5.
    self._output_std_arrays = {}
    for v, indexers in output_variables.items():
      self._output_std_arrays[v] = output_stats_ds[v + "_std"].sel(
          indexers
      )  # pytype : disable=wrong-arg-types

    # The times can be slightly off due to the leap years.
    # Member index.
    self._indexes = [ind["member"] for ind in input_member_indexer]
    self._len_time = input_ds.dims["time"]
    self._len = len(self._indexes) * self._len_time
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps

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

    # Checking the index for the member and time (of the year)
    idx_member = idx // self._len_time
    idx_time = idx % self._len_time

    member = self._indexes[idx_member]
    date = self._input_time_array[idx_time]
    # computing day of the year.
    dayofyear = (
        date - np.datetime64(str(date.astype("datetime64[Y]")))
    ) / np.timedelta64(1, "D") + 1
    if dayofyear <= 0 or dayofyear > 366:
      raise ValueError(f"Invalid day of the year: {dayofyear}")

    sample_input = {}
    mean_input = {}
    std_input = {}
    mean_output = {}
    std_output = {}

    for v, da in self._input_arrays.items():

      array = xrts.read(da[member].sel(time=date)).data

      # Array is either two-or three-dimensional.
      assert array.ndim == 2 or array.ndim == 3
      sample_input[v] = (
          np.expand_dims(array, axis=-1) if array.ndim == 2 else array
      )

    for v, da in self._input_mean_arrays.items():
      mean_array = xrts.read(da[member].sel(dayofyear=int(dayofyear))).data
      mean_input[v] = (
          np.expand_dims(mean_array, axis=-1)
          if mean_array.ndim == 2
          else mean_array
      )

    for v, da in self._input_std_arrays.items():
      std_array = xrts.read(da[member].sel(dayofyear=int(dayofyear))).data
      std_input[v] = (
          np.expand_dims(std_array, axis=-1)
          if std_array.ndim == 2
          else std_array
      )

    item["input"] = sample_input
    item["input_mean"] = mean_input
    item["input_std"] = std_input

    for v, da in self._output_mean_arrays.items():
      mean_array = xrts.read(da.sel(dayofyear=int(dayofyear))).data
      assert mean_array.ndim == 2 or mean_array.ndim == 3
      mean_output[v] = (
          np.expand_dims(mean_array, axis=-1)
          if mean_array.ndim == 2
          else mean_array
      )

    for v, da in self._output_std_arrays.items():
      std_array = xrts.read(da.sel(dayofyear=int(dayofyear))).data
      # Adds an extra dimension is
      assert std_array.ndim == 2 or std_array.ndim == 3
      std_output[v] = (
          np.expand_dims(std_array, axis=-1)
          if std_array.ndim == 2
          else std_array
      )

    item["output_mean"] = mean_output
    item["output_std"] = std_output

    if self._time_stamps:
      item["input_time_stamp"] = date
      item["input_member"] = member

    return item


def create_ensemble_lens2_era5_loader_with_climatology(
    date_range: tuple[str, str],
    input_dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    input_climatology: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    input_mean_stats_path: epath.PathLike = _LENS2_MEAN_CLIMATOLOGY_PATH,
    input_std_stats_path: epath.PathLike = _LENS2_STD_CLIMATOLOGY_PATH,
    input_member_indexer: (
        tuple[dict[str, str], ...] | tuple[types.MappingProxyType, ...]
    ) = _LENS2_MEMBER_INDEXER,
    input_variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    output_dataset_path: epath.PathLike = _ERA5_DATASET_PATH,
    output_climatology: epath.PathLike = _ERA5_STATS_PATH,  # pylint: disable=dangerous-default-value
    output_variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,  # pylint: disable=dangerous-default-value
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 4,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    time_stamps: bool = False,
    num_epochs: int | None = None,
):
  """Creates a loader for ERA5 and LENS2 aligned by date.

  Args:
    date_range: Date range for the data used in the dataloader.
    input_dataset_path: Path for the input (lens2) dataset.
    input_climatology: Path of the Zarr file containing the statistics.
    input_mean_stats_path: Path of the Zarr file containing the mean statistics
      of the climatology (both mean and std).
    input_std_stats_path: Path of the Zarr file containing the std statistics of
      the climatology (both mean and std).
    input_member_indexer: The index for the ensemble members from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    input_variable_names: Input variables to be included in the batch.
    output_dataset_path: Path for the output (era5) dataset.
    output_climatology: Path of the Zarr file containing the climatologies.
    output_variables: Variables in the output dataset to be chosen.
    shuffle: Whether to shuffle the data points.
    seed: Random seed for the random number generator.
    batch_size: batch size for the contiguous chunk.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to include the time stamps in the output.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
  """

  source = DataSourceEnsembleWithClimatology(
      date_range=date_range,
      input_dataset=input_dataset_path,
      input_variable_names=input_variable_names,
      input_member_indexer=input_member_indexer,
      input_climatology=input_climatology,
      output_dataset=output_dataset_path,
      output_variables=output_variables,
      output_climatology=output_climatology,
      resample_at_nan=False,
      dims_order_input=["member", "time", "longitude", "latitude"],
      dims_order_output=["time", "longitude", "latitude", "level"],
      resample_seed=9999,
      time_stamps=time_stamps,
  )

  member_indexer = input_member_indexer[0]
  # Just the first one to extract the statistics.
  input_variables = {v: member_indexer for v in input_variable_names}

  transformations = [
      transforms.StandardizeNestedWithStats(
          main_field="output",
          mean_field="output_mean",
          std_field="output_std",
          input_fields=(*output_variables.keys(),),
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
      transforms.StandardizeNested(
          main_field="input_mean",
          input_fields=(*input_variables.keys(),),
          mean=read_stats_simple(
              input_mean_stats_path, input_variable_names, "mean",
          ),
          std=read_stats_simple(
              input_std_stats_path, input_variable_names, "mean",
          ),
      ),
  )

  transformations.append(
      transforms.StandardizeNested(
          main_field="input_std",
          input_fields=(*input_variables.keys(),),
          mean=read_stats_simple(
              input_mean_stats_path, input_variable_names, "std",
          ),
          std=read_stats_simple(
              input_std_stats_path, input_variable_names, "std",
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

  loader = pygrain.load(
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
  return loader


def create_ensemble_lens2_loader_with_climatology_for_inference(
    date_range: tuple[str, str],
    input_dataset_path: epath.PathLike = _LENS2_DATASET_PATH,  # pylint: disable=dangerous-default-value
    input_climatology: epath.PathLike = _LENS2_STATS_PATH,  # pylint: disable=dangerous-default-value
    input_mean_stats_path: epath.PathLike = _LENS2_MEAN_CLIMATOLOGY_PATH,
    input_std_stats_path: epath.PathLike = _LENS2_STD_CLIMATOLOGY_PATH,
    input_member_indexer: (
        tuple[dict[str, str], ...] | tuple[types.MappingProxyType, ...]
    ) = _LENS2_MEMBER_INDEXER,
    input_variable_names: Sequence[str] = _LENS2_VARIABLE_NAMES,
    output_climatology: epath.PathLike = _ERA5_STATS_PATH,  # pylint: disable=dangerous-default-value
    output_variables: (
        dict[str, dict[str, Any] | None] | types.MappingProxyType
    ) = _ERA5_VARIABLES,  # pylint: disable=dangerous-default-value
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 4,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    time_stamps: bool = False,
    num_epochs: int | None = 1,
):
  """Creates a loader for LENS2 with climatology for inference.

  Args:
    date_range: Date range for the data used in the dataloader.
    input_dataset_path: Path for the input (lens2) dataset.
    input_climatology: Path of the Zarr file containing the statistics.
    input_mean_stats_path: Path of the Zarr file containing the mean statistics
      of the climatology (both mean and std).
    input_std_stats_path: Path of the Zarr file containing the std statistics of
      the climatology (both mean and std).
    input_member_indexer: The index for the ensemble members from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    input_variable_names: Input variables to be included in the batch.
    output_climatology: Path of the Zarr file containing the climatologies.
    output_variables: Variables in the output dataset to be chosen.
    shuffle: Whether to shuffle the data points.
    seed: Random seed for the random number generator.
    batch_size: batch size for the contiguous chunk.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to include the time stamps in the output.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
  """

  source = DataSourceEnsembleWithClimatologyInference(
      date_range=date_range,
      input_dataset=input_dataset_path,
      input_variable_names=input_variable_names,
      input_member_indexer=input_member_indexer,
      input_climatology=input_climatology,
      output_variables=output_variables,
      output_climatology=output_climatology,
      resample_at_nan=False,
      dims_order_input=["member", "time", "longitude", "latitude"],
      resample_seed=9999,
      time_stamps=time_stamps,
  )

  member_indexer = input_member_indexer[0]
  # Just the first one to extract the statistics.
  input_variables = {v: member_indexer for v in input_variable_names}

  transformations = [
      transforms.StandardizeNestedWithStats(
          main_field="input",
          mean_field="input_mean",
          std_field="input_std",
          input_fields=(*input_variables.keys(),),
      ),
  ]

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
      transforms.StandardizeNested(
          main_field="input_mean",
          input_fields=(*input_variables.keys(),),
          mean=read_stats_simple(
              input_mean_stats_path, input_variable_names, "mean",
          ),
          std=read_stats_simple(
              input_std_stats_path, input_variable_names, "mean",
          ),
      ),
  )

  transformations.append(
      transforms.StandardizeNested(
          main_field="input_std",
          input_fields=(*input_variables.keys(),),
          mean=read_stats_simple(
              input_mean_stats_path, input_variable_names, "std",
          ),
          std=read_stats_simple(
              input_std_stats_path, input_variable_names, "std",
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

  transformations.append(
      transforms.ConcatenateNested(
          main_field="output_mean",
          input_fields=(*output_variables.keys(),),
          output_field="output_mean",
          axis=-1,
          remove_inputs=False,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="output_std",
          input_fields=(*output_variables.keys(),),
          output_field="output_std",
          axis=-1,
          remove_inputs=False,
      ),
  )

  loader = pygrain.load(
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
  return loader
