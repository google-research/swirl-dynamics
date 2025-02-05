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

import abc
from collections.abc import Callable, Mapping, Sequence
import types
from typing import Any, Literal, SupportsIndex

from etils import epath
import grain.python as pygrain
import jax
import numpy as np
from swirl_dynamics.projects.debiasing.rectified_flow import data_utils
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


class CommonSourceEnsemble(abc.ABC):
  """A data source that loads daily ERA5- with ensemble LENS2 data.

  Here we consider a loose alignment between the ERA5 and LENS2 data using the
  time stamps. The pairs feed to the model are selected such that the time
  stamps are roughly the same, so the climatological statistics are roughly
  aligned between the two datasets.
  """

  @abc.abstractmethod
  def __len__(self):
    pass

  @abc.abstractmethod
  def _compute_len(self, *args) -> int:
    pass

  @abc.abstractmethod
  def _compute_indices(self, idx) -> tuple[
      str,
      np.datetime64 | Sequence[np.datetime64],
      np.datetime64 | Sequence[np.datetime64],
      int | Sequence[int],
  ]:
    pass

  @abc.abstractmethod
  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    pass

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

    # Using LENS2 as input, they need to be modified.
    input_variables = {v: input_member_indexer for v in input_variable_names}

    # Computing the date_range.
    date_range = jax.tree.map(lambda x: np.datetime64(x, "D"), date_range)

    # Open the datasets.
    input_ds = xrts.open_zarr(input_dataset).sel(time=slice(*date_range))
    output_ds = xrts.open_zarr(output_dataset).sel(time=slice(*date_range))
    # These contain the climatologies
    input_stats_ds = xrts.open_zarr(input_climatology)
    output_stats_ds = xrts.open_zarr(output_climatology)

    # Transpose the datasets if necessary.
    if dims_order_input:
      input_ds = input_ds.transpose(*dims_order_input)
    if dims_order_output:
      output_ds = output_ds.transpose(*dims_order_output)
    if dims_order_output_stats:
      output_stats_ds = output_stats_ds.transpose(*dims_order_output_stats)
    if dims_order_input_stats:
      input_stats_ds = input_stats_ds.transpose(*dims_order_input_stats)

    # Selects the input arrays and builds the dictionary of xarray datasets
    # to be used for the climatology.
    self._input_arrays = {}
    self._input_mean_arrays = {}
    self._input_std_arrays = {}

    for v, indexers in input_variables.items():
      self._input_arrays[v] = {}
      self._input_mean_arrays[v] = {}
      self._input_std_arrays[v] = {}
      for index in indexers:
        idx = tuple(index.values())[0]
        self._input_arrays[v][idx] = input_ds[v].sel(index)
        # We load the arrays with statistics to accelerate the loading.
        self._input_mean_arrays[v][idx] = input_stats_ds[v].sel(index).load()
        self._input_std_arrays[v][idx] = (
            input_stats_ds[v + "_std"].sel(index).load()
        )

    # Build the output arrays for the different output variables and statistics.
    self._output_arrays = {}
    self._output_mean_arrays = {}
    self._output_std_arrays = {}
    for v, indexers in output_variables.items():
      self._output_arrays[v] = output_ds[v].sel(indexers)
      # We load the arrays with statistics to accelerate the loading.
      self._output_mean_arrays[v] = output_stats_ds[v].sel(indexers).load()
      self._output_std_arrays[v] = (
          output_stats_ds[v + "_std"].sel(indexers).load()
      )

    self._output_coords = output_ds.coords

    # The times can be slightly off due to the leap years.
    # Member index.
    self._indexes = [ind["member"] for ind in input_member_indexer]
    self._len_time = np.min([input_ds.dims["time"], output_ds.dims["time"]])
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    # Common time array (mostly avoid leap years)
    self._common_time_array = np.intersect1d(
        self._input_time_array, self._output_time_array
    )
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, Any]:
    """Retrieves record and retry if NaN found."""
    idx = record_key.__index__()
    if not idx < self.__len__():
      raise ValueError(f"Index out of range: {idx} / {self.__len__() - 1}")

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

    # Computes the indices.
    member, date_input, date_output, dayofyear = self._compute_indices(idx)

    sample_input, mean_input, std_input = {}, {}, {}
    sample_output, mean_output, std_output = {}, {}, {}

    # Loop over the variable key and each dataset, and statistics.
    for var_key, da in self._input_arrays.items():
      array = xrts.read(da[member].sel(time=date_input)).data
      sample_input[var_key] = self._maybe_expands_dims(array)

    for var_key, da in self._input_mean_arrays.items():
      mean_array = xrts.read(da[member].sel(dayofyear=dayofyear)).data
      mean_input[var_key] = self._maybe_expands_dims(mean_array)

    for var_key, da in self._input_std_arrays.items():
      std_array = xrts.read(da[member].sel(dayofyear=dayofyear)).data
      std_input[var_key] = self._maybe_expands_dims(std_array)

    item["input"] = sample_input
    item["input_mean"] = mean_input
    item["input_std"] = std_input

    # Loop over the variable key and each dataset, and statistics.
    for var_key, da in self._output_arrays.items():
      array = xrts.read(da.sel(time=date_output)).data
      sample_output[var_key] = self._maybe_expands_dims(array)

    for var_key, da in self._output_mean_arrays.items():
      mean_array = xrts.read(da.sel(dayofyear=dayofyear)).data
      mean_output[var_key] = self._maybe_expands_dims(mean_array)

    for var_key, da in self._output_std_arrays.items():
      std_array = xrts.read(da.sel(dayofyear=dayofyear)).data
      std_output[var_key] = self._maybe_expands_dims(std_array)

    item["output"] = sample_output
    item["output_mean"] = mean_output
    item["output_std"] = std_output

    if self._time_stamps:
      # Adds the time and the ensemble member corresponding to the data.
      item["input_time_stamp"] = date_input
      item["output_time_stamp"] = date_output
      item["input_member"] = member

    return item

  def get_output_coords(self):
    """Returns the coordinates of the output dataset."""
    return self._output_coords


class DataSourceEnsembleWithClimatology(CommonSourceEnsemble):
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
      dims_order_input: Order of the dimensions (time, member, lat, lon, fields)
        of the input variables. If None, the dimensions are not changed from the
        order in the xarray dataset.
      dims_order_output: Order of the dimensions (time, member, lat, lon,
        fields) of the input variables. If None, the dimensions are not changed
        from the order in the xarray dataset.
      dims_order_input_stats: Order of the variables for the input statistics.
      dims_order_output_stats: Order of the variables for the output statistics.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Wheter to add the time stamps to the samples.
    """
    super().__init__(
        date_range,
        input_dataset,
        input_variable_names,
        input_member_indexer,
        input_climatology,
        output_dataset,
        output_variables,
        output_climatology,
        dims_order_input,
        dims_order_output,
        dims_order_input_stats,
        dims_order_output_stats,
        resample_at_nan,
        resample_seed,
        time_stamps)
    self._len = self._compute_len()

  def __len__(self):
    return self._len

  def _compute_len(self) -> int:
    return len(self._indexes) * self._len_time

  def _compute_indices(
      self, idx: int
  ) -> tuple[str, np.datetime64, np.datetime64, int]:
    """Computes the xarray indices for the data.

    Args:
      idx: The index of the sample.

    Returns:
      A tuple with the member, the date of the input, the date of the output,
      and the day of the year.
    """
    # Checking the index for the member and time (of the year)
    idx_member = idx // self._len_time
    idx_time = idx % self._len_time
    member = self._indexes[idx_member]
    date = self._common_time_array[idx_time]
    dayofyear = int((
        date - np.datetime64(str(date.astype("datetime64[Y]")))
    ) / np.timedelta64(1, "D") + 1)

    # Checks the validity of dayofyear.
    if dayofyear <= 0 or dayofyear > 366:
      raise ValueError(f"Invalid day of the year: {dayofyear}")

    return (member, date, date, dayofyear)

  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    return data_utils.maybe_expand_dims(
        x, allowed_dims=(2, 3), trigger_expand_dims=2
    )


class DataSourceEnsembleWithClimatologyInference(CommonSourceEnsemble):
  """A data source that loads ensemble LENS2 data with climatology."""

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
      dims_order_input: Order of the dimensions (time, member, lat, lon, fields)
        of the input variables. If None, the dimensions are not changed from the
        order in the xarray dataset.
      dims_order_output: Order of the dimensions (time, member, lat, lon,
        fields) of the input variables. If None, the dimensions are not changed
        from the order in the xarray dataset.
      dims_order_input_stats: Order of the variables for the input statistics.
      dims_order_output_stats: Order of the variables for the output statistics.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Wheter to add the time stamps to the samples.
    """
    super().__init__(
        date_range,
        input_dataset,
        input_variable_names,
        input_member_indexer,
        input_climatology,
        output_dataset,
        output_variables,
        output_climatology,
        dims_order_input,
        dims_order_output,
        dims_order_input_stats,
        dims_order_output_stats,
        resample_at_nan,
        resample_seed,
        time_stamps)

    # Here the length of the time is the length of the time stamps. As the
    # output samples times do not play a role in the inference. Only the output
    # climatology is of interest.
    self._len_time = len(self._input_time_array)
    self._len = self._compute_len()

  def __len__(self):
    return self._len

  def _compute_len(self) -> int:
    return len(self._indexes) * self._len_time

  def _compute_indices(
      self, idx: int
  ) -> tuple[str, np.datetime64, np.datetime64, int]:
    """Computes the xarray indices for the data.

    Args:
      idx: The index of the sample.

    Returns:
      A tuple with the member, the date of the input, the date of the output,
      and the day of the year.
    """
    # Checking the index for the member and time (of the year)
    idx_member = idx // self._len_time
    idx_time = idx % self._len_time
    member = self._indexes[idx_member]
    date_input = self._common_time_array[idx_time]
    # We don't need the date of the output. But to conform with the interface,
    # we set it to be first date of the output.
    date_output_dummy = self._output_time_array[0]
    dayofyear = int((
        date_input - np.datetime64(str(date_input.astype("datetime64[Y]")))
    ) / np.timedelta64(1, "D") + 1)

    # Checks the validity of dayofyear.
    if dayofyear <= 0 or dayofyear > 366:
      raise ValueError(f"Invalid day of the year: {dayofyear}")

    return (member, date_input, date_output_dummy, dayofyear)

  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    return data_utils.maybe_expand_dims(
        x, allowed_dims=(2, 3), trigger_expand_dims=2
    )


class ContiguousDataSourceEnsembleWithClimatology(CommonSourceEnsemble):
  """A data source loads contiguous chunks of data loosely paired by date."""

  def __init__(
      self,
      date_range: tuple[str, str],
      chunk_size: int,
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
      chunk_size: The size of the chunks to be used.
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
      dims_order_input: Order of the dimensions (time, member, lat, lon, fields)
        of the input variables. If None (the default), the dimensions are not
        changed from the order in the xarray dataset
      dims_order_output: Order of the dimensions (time, member, lat, lon, field)
        of the output variables. If None, the dimensions are not changed from
        the order in the xarray dataset
      dims_order_input_stats: Order of the dimensions of the variables for the
        input statistics. If None, the dimensions are not changed from the
        order in the xarray dataset
      dims_order_output_stats: Order of the dimensions of the variables for the
        output statistics. If None, the dimensions are not changed from the
        order in the xarray dataset
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Whether to add the time stamps to the samples.
    """
    super().__init__(
        date_range,
        input_dataset,
        input_variable_names,
        input_member_indexer,
        input_climatology,
        output_dataset,
        output_variables,
        output_climatology,
        dims_order_input,
        dims_order_output,
        dims_order_input_stats,
        dims_order_output_stats,
        resample_at_nan,
        resample_seed,
        time_stamps)
    # Reduce the length of each member in time to accommodate for the chunks.
    self._len_time = self._len_time - chunk_size
    self._chunk_size = chunk_size
    self._len = self._compute_len()

  def __len__(self):
    return self._len

  def _compute_len(self) -> int:
    return len(self._indexes) * self._len_time

  def _compute_indices(
      self, idx: int
  ) -> tuple[
      str, Sequence[np.datetime64], Sequence[np.datetime64], Sequence[int]
  ]:
    # Checking the index for the member and time (of the year)
    idx_member = idx // self._len_time
    idx_time = idx % self._len_time

    member = self._indexes[idx_member]
    dates = self._common_time_array[idx_time: idx_time + self._chunk_size]

    # Computes the days of the year.
    daysofyear = [int((
        date - np.datetime64(str(date.astype("datetime64[Y]")))
    ) / np.timedelta64(1, "D") + 1) for date in dates]

    # Checks for uniqueness of dates and daysofyear.
    if len(dates) != len(set(dates)):
      raise ValueError(f"Dates are not unique: {dates}")
    if len(daysofyear) != len(set(daysofyear)):
      raise ValueError(f"Dates are not unique: {daysofyear}")

    return (member, dates, dates, daysofyear)

  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    return data_utils.maybe_expand_dims(
        x, allowed_dims=(3, 4), trigger_expand_dims=3
    )


# TODO: Merge this loader with the one below and add a flag.
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
    inference_mode: bool = False,
    num_epochs: int | None = None,
):
  """Creates a loader for ERA5 and LENS2 loosely aligned by date.

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
    inference_mode: Whether to use the inference dataset, and provide a dummy
      output.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
  """

  # In inference mode, we need to use the inference datset, which returns a
  # dummy output.
  if inference_mode:
    source = DataSourceEnsembleWithClimatologyInference(
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

  else:
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

  # Concatenating the statistics.
  # The input statistics are normalized, as they are fed to the model.
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

  # The output statistics are raw, as they are used to return the samples to the
  # original scale (mostly during inference).
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


def create_ensemble_lens2_era5_chunked_loader_with_climatology(
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
    batch_size: int = 32,
    chunk_size: int = 8,
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
    batch_size: Total batch size (from all aggregating all chunks).
    chunk_size: Size of the chunk for the contiguous data.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to include the time stamps in the output.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
  """
  if batch_size % chunk_size != 0:
    raise ValueError(
        "Batch size must be a multiple of the chunk size."
    )
  num_chunks = batch_size // chunk_size

  source = ContiguousDataSourceEnsembleWithClimatology(
      date_range=date_range,
      chunk_size=chunk_size,
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
  # Also standardizing the statistics.
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
  # Concatenating the statistics for the input.
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
  # Concatenating the statistics for the output.
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

  # This block and the source is the only difference with the non-chunked
  # version.
  # TODO: Refactor this for better readability.
  transformations.append(
      pygrain.Batch(
          batch_size=num_chunks, drop_remainder=drop_remainder
      )
  )
  # This one goes at the end.
  # Reshapes the batch to [batch, lon, lat, channel].
  transformations.append(transforms.ReshapeBatch())

  sampler = pygrain.IndexSampler(
      num_records=len(source),
      shuffle=shuffle,
      seed=seed,
      num_epochs=num_epochs,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
  )

  return pygrain.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=transformations,
      worker_count=worker_count,
  )


def create_ensemble_lens2_era5_time_chunked_loader_with_climatology(
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
    batch_size: int = 32,
    chunk_size: int = 8,
    time_batch_size: int = 1,
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
    batch_size: Total batch size (from all aggregating all chunks).
    chunk_size: Size of the chunk for the contiguous data.
    time_batch_size: Size of the time batch.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to include the time stamps in the output.
    num_epochs: Number of epochs, by defaults the loader will run forever.

  Returns:
  """
  if batch_size % chunk_size != 0:
    raise ValueError(
        "Batch size must be a multiple of the chunk size."
    )
  if chunk_size % time_batch_size != 0:
    raise ValueError(
        "Chunk size must be a multiple of the time batch size."
    )
  # The effective batch size is batch_size // time_batch_size.
  num_chunks = batch_size // chunk_size

  source = ContiguousDataSourceEnsembleWithClimatology(
      date_range=date_range,
      chunk_size=chunk_size,
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
  # Also standardizing the statistics.
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
  # Concatenating the statistics for the input.
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
  # Concatenating the statistics for the output.
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

  # Reshapes the batch to [new_chunk_size, lon, lat, channel * time_batch_size],
  # where new_chunk_size = chunk_size // time_batch_size.
  transformations.append(
      transforms.TimeToChannel(time_batch_size=time_batch_size)
  )

  # Here it performs the batching.
  # [num_chunks, new_chunk_size, lon, lat, channel * time_batch_size]
  transformations.append(
      pygrain.Batch(
          batch_size=num_chunks, drop_remainder=drop_remainder
      )
  )

  # Reshapes the batch again.
  # [num_chunks * new_chunk_size, lon, lat, channel * time_batch_size]
  transformations.append(transforms.ReshapeBatch())

  sampler = pygrain.IndexSampler(
      num_records=len(source),
      shuffle=shuffle,
      seed=seed,
      num_epochs=num_epochs,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
  )

  return pygrain.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=transformations,
      worker_count=worker_count,
  )
