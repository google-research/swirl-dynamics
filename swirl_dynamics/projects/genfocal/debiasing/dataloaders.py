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

"""Class with the data loaders the climatology-based models.

Here we present the dataloaders for the climatology-based models, together with
several functions to build such dataloaders and the DataSource class that is
used to load the data.

Following the philosophy behind PyGrain, the dataloaders are built around a
DataSource that reads the data from a Zarr file via indexation, followed by a
series of transformations that normalize and reshape the data.

The DataSource used by the dataloaders is built around a CommonSourceEnsemble
class, which encapsulates the common functionality that is then inherited by the
following classes.

                      - DataSourceEnsembleWithClimatology
CommonSourceEnsemble  - DataSourceEnsembleWithClimatologyInference
                      - ContiguousDataSourceEnsembleWithClimatology

- ContiguousDataSourceEnsembleWithClimatology: This DataSource is the workhorse
for training the climatology-based models. It is designed to produce overlapping
chunks and loads the data by chunks in memory, minimizing the number of reads
from the network. This is crucial for obtaining relatively fast training.

For the evaluation and inference, we use slightly different data loaders. As the
repetitive application of the neural network becomes the bottleneck, we do not
minimize the number of reads from the network. Instead, we use the same
interface as above, but we load snapshot-by-snapshot and then concatenate them
together.

- DataSourceEnsembleWithClimatology: This DataSource is the workhorse for
evaluation during the period where both LENS2 and ERA5 data are available. This
DataSource loads snapshot by snapshot to memory and it is used to produce
non-overlapping chunks, with the corresponding climatologies.

- DataSourceEnsembleWithClimatologyInference: This DataSource is the workhorse
for inference during the evaluation period where only LENS2 data is available.
It is also used to produces non-overlapping chunks.

The DataSources then are used to build the dataloaders, which are responsible
for producing the batches of data for training and evaluation. The DataSources
are also responsible for loading the correct climatologies of both input and
output datasets.
"""

import abc
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import types
from typing import Any, Literal, SupportsIndex

from absl import logging
from etils import epath
import grain.python as pygrain
import jax
import ml_collections
import numpy as np
from swirl_dynamics.projects.genfocal.debiasing import pygrain_transforms as transforms
import xarray_tensorstore as xrts


Array = jax.Array
PyTree = Any
DynamicsFn = Callable[[Array, Array, PyTree], Array]

_ERA5_VARIABLES = types.MappingProxyType({
    "10m_magnitude_of_wind": None,
    "2m_temperature": None,
    "geopotential": {"level": [200, 500]},
    "mean_sea_level_pressure": None,
    "specific_humidity": {"level": 1000},
    "u_component_of_wind": {"level": [200, 850]},
    "v_component_of_wind": {"level": [200, 850]},
})

_LENS2_VARIABLE_NAMES = (
    "WSPDSRFAV",
    "TREFHT",
    "Z200",
    "Z500",
    "PSL",
    "QREFHT",
    "U200",
    "U850",
    "V200",
    "V850",
)

# For training we still use the tuple of dictionaries.
_LENS2_MEMBER_INDEXER = (
    {"member": "cmip6_1001_001"},
    {"member": "cmip6_1021_002"},
    {"member": "cmip6_1041_003"},
    {"member": "cmip6_1061_004"},
)

# pylint: enable=line-too-long
_ERA5_DATASET_PATH = "data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = "data/era5/1p5deg_11vars_windspeed_1961-2000_daily_v2.zarr"

# Interpolated dataset to match the resolution of the ERA5 data set.
_LENS2_DATASET_PATH = (
    "data/lens2/lens2_240x121_lonlat_1960-2020_10_vars_4_train_members.zarr"
)

# Statistics for the LENS2 dataset.
_LENS2_STATS_PATH = "data/lens2/lens2_240x121_10_vars_4_members_lonlat_clim_daily_1961_to_2000_31_dw.zarr"

# Mean and STD of the statistics for the LENS2 dataset.
_LENS2_MEAN_STATS_PATH = (
    "data/lens2/mean_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr"
)
_LENS2_STD_STATS_PATH = (
    "data/lens2/std_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr"
)

# pylint: disable=line-too-long


def read_stats(
    dataset: epath.PathLike,
    variables: Mapping[str, Mapping[str, Any] | None],
    field: Literal["mean", "std"],
    use_batched: bool = False,
) -> dict[str, np.ndarray]:
  """Reads variables from a zarr dataset and returns as a dict of ndarrays.

  This function is used to read the statistics of the climatologies, and it
  relies in the naming convention of the fields to distinguish between the
  mean and the std. The name of the variable contains the mean, and the std
  is named by adding the suffix "_std".

  Args:
    dataset: The path of the zarr dataset.
    variables: The variables to read.
    field: The field to read, either mean or std.
    use_batched: Whether the data loader is batched or not. If it is batched,
      then we need to add an extra dimension to the statistics.

  Returns:
    A dictionary of variables and their statistics.
  """
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
    var_idx = var + "_std" if field == "std" else var
    stats = ds[var_idx].to_numpy()
    assert stats.ndim == 2 or stats.ndim == 3
    stats = np.expand_dims(stats, axis=-1) if stats.ndim == 2 else stats
    out[var] = stats
  return out


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
      load_stats: bool = False,
      max_retries: int = 5,
  ):
    """Data source constructor.

    This is an abstract class, that aggregates the common functionality of the
    data loaders for the climatology-based models.

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
      load_stats: Whether to load the climatology statistics to memory when
        using the data loader. This allows for a faster loading.
      max_retries: The maximum number of retries for NaN or KeyError.
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
        if load_stats:
          self._input_mean_arrays[v][idx] = input_stats_ds[v].sel(index).load()
          self._input_std_arrays[v][idx] = (
              input_stats_ds[v + "_std"].sel(index).load()
          )
        else:
          self._input_mean_arrays[v][idx] = input_stats_ds[v].sel(index)
          self._input_std_arrays[v][idx] = input_stats_ds[v + "_std"].sel(index)

    # Build the output arrays for the different output variables and statistics.
    self._output_arrays = {}
    self._output_mean_arrays = {}
    self._output_std_arrays = {}
    for v, indexers in output_variables.items():
      self._output_arrays[v] = output_ds[v].sel(indexers)
      # We load the arrays with statistics to accelerate the loading.
      if load_stats:
        self._output_mean_arrays[v] = output_stats_ds[v].sel(indexers).load()
        self._output_std_arrays[v] = (
            output_stats_ds[v + "_std"].sel(indexers).load()
        )
      else:
        self._output_mean_arrays[v] = output_stats_ds[v].sel(indexers)
        self._output_std_arrays[v] = output_stats_ds[v + "_std"].sel(indexers)

    self._output_coords = output_ds.coords

    # The times can be slightly off due to the leap years.
    # Member index.
    self._indexes = [ind["member"] for ind in input_member_indexer]
    self._len_time = np.min([input_ds.dims["time"], output_ds.dims["time"]])
    self._input_time_array = xrts.read(input_ds["time"]).data
    self._output_time_array = xrts.read(output_ds["time"]).data
    # Common time array (to avoid issues with leap years).
    self._common_time_array = np.intersect1d(
        self._input_time_array, self._output_time_array
    )
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self._time_stamps = time_stamps
    self._max_retries = max_retries

  def __getitem__(self, record_key: SupportsIndex) -> Mapping[str, Any]:
    """Retrieves record and retry if NaN found."""
    idx = record_key.__index__()
    if not idx < self.__len__():
      raise ValueError(f"Index out of range: {idx} / {self.__len__() - 1}")

    # Retries if KeyError found.
    for _ in range(self._max_retries + 1):
      try:
        item = self.get_item(idx)

        # Checks for NaN in the data. If there is NaN, resample the index.
        if self._resample_at_nan and (
            np.isnan(item["input"]).any() or np.isnan(item["output"]).any()
        ):
          logging.warning("NaN found in data, index %d", idx)
          # Resample the index.
          idx = np.random.default_rng(self._resample_seed + idx).integers(
              0, len(self)
          )
          continue

        return item

      # If KeyError found, resample the index.
      except KeyError:
        logging.warning("Key error encountered %d", idx)
        idx = np.random.default_rng(self._resample_seed + idx).integers(
            0, len(self)
        )
        continue

    raise RuntimeError(f"Failed to get item after {self._max_retries} retries.")

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
      load_stats: bool = True,
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
      load_stats: Whether to load the statistics to accelerate the loading.
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
        time_stamps,
    )
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
    dayofyear = int(
        (date - np.datetime64(str(date.astype("datetime64[Y]"))))
        / np.timedelta64(1, "D")
        + 1
    )

    # Checks the validity of dayofyear.
    if dayofyear <= 0 or dayofyear > 366:
      raise ValueError(f"Invalid day of the year: {dayofyear}")

    return (member, date, date, dayofyear)

  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    return maybe_expand_dims(x, allowed_dims=(2, 3), trigger_expand_dims=2)


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
      load_stats: bool = True,
  ):
    """Data source constructor for the inference.

    The inference data source is different from the training data source as
    the output samples are dummies, and the fact that the chunks are not
    overlapping.

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
      load_stats: Whether to load the statistics to accelerate the loading.
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
        time_stamps,
        load_stats,
    )

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
    # Here we _input_time_array, as we will sample from the input dataset.
    date_input = self._input_time_array[idx_time]
    # We don't need the date of the output. But to conform with the interface,
    # we set it to be first date of the output.
    date_output_dummy = self._output_time_array[0]
    dayofyear = int(
        (date_input - np.datetime64(str(date_input.astype("datetime64[Y]"))))
        / np.timedelta64(1, "D")
        + 1
    )

    # Checks the validity of dayofyear.
    if dayofyear <= 0 or dayofyear > 366:
      raise ValueError(f"Invalid day of the year: {dayofyear}")

    return (member, date_input, date_output_dummy, dayofyear)

  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    return maybe_expand_dims(x, allowed_dims=(2, 3), trigger_expand_dims=2)


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
      load_stats: bool = True,
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
        input statistics. If None, the dimensions are not changed from the order
        in the xarray dataset
      dims_order_output_stats: Order of the dimensions of the variables for the
        output statistics. If None, the dimensions are not changed from the
        order in the xarray dataset
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      time_stamps: Whether to add the time stamps to the samples.
      load_stats: Whether to load the statistics to accelerate the loading.
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
        time_stamps,
    )
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
    dates = self._common_time_array[idx_time : idx_time + self._chunk_size]

    # Computes the days of the year.
    daysofyear = [
        int(
            (date - np.datetime64(str(date.astype("datetime64[Y]"))))
            / np.timedelta64(1, "D")
            + 1
        )
        for date in dates
    ]

    # Checks for uniqueness of dates and daysofyear.
    if len(dates) != len(set(dates)):
      raise ValueError(f"Dates are not unique: {dates}")
    if len(daysofyear) != len(set(daysofyear)):
      raise ValueError(f"Dates are not unique: {daysofyear}")

    return (member, dates, dates, daysofyear)

  def _maybe_expands_dims(self, x: np.ndarray) -> np.ndarray:
    return maybe_expand_dims(x, allowed_dims=(3, 4), trigger_expand_dims=3)


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
) -> pygrain.DataLoader:
  """Creates a loader for ERA5 and LENS2 loosely aligned by date.

  The loader will have the following keys in the batch:
    - x_0: The input (LENS2) data normalized using the input climatology.
    - x_1: The output (ERA5) data normalized using the output climatology.
    - channel:mean: The normalized mean of the LENS2 input, which is fed to the
      model as a conditioning field.
    - channel:std:  The normalized standard deviation of the LENS2 input, which
      is fed to the model as a conditioning field.
    - input_mean: The (unnormalized) climatoligical mean of the input
      distribution.
    - input_std: The (unnormalized) climatoligical standard deviation of the
      input distribution.
    - output_mean: The (unnormalized) climatoligical mean of the output
      distribution.
    - output_std: The (unnormalized) climatoligical standard deviation of the
      output distribution.

    Additionally, if time_stamps is True:
    - input_time_stamp: The time stamp of the input sample.
    - output_time_stamp: The time stamp of the output sample.


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
    A pygrain loader.
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
  # Also normalizing the statistics.
  transformations.append(
      transforms.StandardizeNestedToNewField(
          main_field="input_mean",
          main_output_field="channel:mean",
          input_fields=(*input_variables.keys(),),
          mean=read_stats_simple(
              input_mean_stats_path,
              input_variable_names,
              "mean",
          ),
          std=read_stats_simple(
              input_std_stats_path,
              input_variable_names,
              "mean",
          ),
      ),
  )

  transformations.append(
      transforms.StandardizeNestedToNewField(
          main_field="input_std",
          main_output_field="channel:std",
          input_fields=(*input_variables.keys(),),
          mean=read_stats_simple(
              input_mean_stats_path,
              input_variable_names,
              "std",
          ),
          std=read_stats_simple(
              input_std_stats_path,
              input_variable_names,
              "std",
          ),
      ),
  )

  # Concatenating the statistics.
  # The input statistics are normalized, as they are fed to the model.
  transformations.append(
      transforms.ConcatenateNested(
          main_field="channel:mean",
          input_fields=(*input_variables.keys(),),
          output_field="channel:mean",
          axis=-1,
          remove_inputs=False,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="channel:std",
          input_fields=(*input_variables.keys(),),
          output_field="channel:std",
          axis=-1,
          remove_inputs=False,
      ),
  )

  # We also concatenate the raw input statistics.
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input_mean",
          input_fields=(*input_variables.keys(),),
          output_field="input_mean",
          axis=-1,
          remove_inputs=False,
      ),
  )
  transformations.append(
      transforms.ConcatenateNested(
          main_field="input_std",
          input_fields=(*input_variables.keys(),),
          output_field="input_std",
          axis=-1,
          remove_inputs=False,
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
    raise ValueError("Batch size must be a multiple of the chunk size.")
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
              input_mean_stats_path,
              input_variable_names,
              "mean",
          ),
          std=read_stats_simple(
              input_std_stats_path,
              input_variable_names,
              "mean",
          ),
      ),
  )
  transformations.append(
      transforms.StandardizeNested(
          main_field="input_std",
          input_fields=(*input_variables.keys(),),
          mean=read_stats_simple(
              input_mean_stats_path,
              input_variable_names,
              "std",
          ),
          std=read_stats_simple(
              input_std_stats_path,
              input_variable_names,
              "std",
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
      pygrain.Batch(batch_size=num_chunks, drop_remainder=drop_remainder)
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
    time_to_channel: bool = True,
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
    batch_size: Total batch size (from aggregating all chunks).
    chunk_size: Size of the chunk for the contiguous data.
    time_batch_size: Size of the time batch.
    drop_remainder: Whether to drop the reminder between the full length of the
      dataset and the batchsize.
    worker_count: Number of workers for parallelizing the data loading.
    time_stamps: Whether to include the time stamps in the output.
    num_epochs: Number of epochs, by defaults the loader will run forever.
    time_to_channel: Whether to reshape the batch to [new_chunk_size, lon, lat,
      channel * time_batch_size], where new_chunk_size = chunk_size //
      time_batch_size.

  Returns:
  """
  if batch_size % chunk_size != 0:
    raise ValueError("Batch size must be a multiple of the chunk size.")
  if chunk_size % time_batch_size != 0:
    raise ValueError("Chunk size must be a multiple of the time batch size.")
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
              input_mean_stats_path,
              input_variable_names,
              "mean",
          ),
          std=read_stats_simple(
              input_std_stats_path,
              input_variable_names,
              "mean",
          ),
      ),
  )
  transformations.append(
      transforms.StandardizeNested(
          main_field="input_std",
          input_fields=(*input_variables.keys(),),
          mean=read_stats_simple(
              input_mean_stats_path,
              input_variable_names,
              "std",
          ),
          std=read_stats_simple(
              input_std_stats_path,
              input_variable_names,
              "std",
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

  if time_to_channel:
    # Reshapes to [new_chunk_size, lon, lat, channel * time_batch_size],
    # where new_chunk_size = chunk_size // time_batch_size.
    transformations.append(
        transforms.TimeToChannel(time_batch_size=time_batch_size)
    )
  else:
    # Reshapes to [new_chunk_size, time_batch_size, lon, lat, channel],
    # where new_chunk_size = chunk_size // time_batch_size.
    transformations.append(
        transforms.TimeSplit(time_batch_size=time_batch_size)
    )

  # Here it performs the batching. We can either have the time dimension merged
  # with the channel dimension or by itself.
  # [num_chunks, new_chunk_size, lon, lat, channel * time_batch_size] or
  # [num_chunks, new_chunk_size, time_batch_size, lon, lat, channel]
  transformations.append(
      pygrain.Batch(batch_size=num_chunks, drop_remainder=drop_remainder)
  )

  # Reshapes the batch again.
  # [num_chunks * new_chunk_size, lon, lat, channel * time_batch_size] or
  # [num_chunks * new_chunk_size, time_batch_size, lon, lat, channel]
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataLoadersConfig:
  """Configuration class for the rectified flow model.

  For the batch_size, chunk_size and time_batch_size, the following convention
  is used:
    - time_batch_size: Number of snapshots in each samplet for time-coherent
      dataloader.
    - chunk_size: Size of the chunk for the contiguous data for each process in
      number of snapshots. This is a multiple of the time_batch_size, and
      accounts for the local batch size. Here each sample has time_batch_size
      contiguous snapshots.
    - batch_size: Total batch size in number of snapshots from aggregating all
      chunks accross all processes.

  Attributes:
    date_range: Date range for the data used in the dataloader.
    batch_size: Total batch size (from aggregating all chunks).
    chunk_size: Size of the chunk for the contiguous data (for each process).
    time_batch_size: Number of snapshots in each batch.
    shuffle: Whether to shuffle the data points.
    worker_count: Number of pygrain workers.
    input_dataset_path: Dataset path for the input (LENS2) dataset.
    input_climatology: Dataset path for the input climatology.
    input_mean_stats_path: Dataset path for the ensemble mean statistics of the
      climatology.
    input_std_stats_path: Dataset path for the ensemble std statistics of the
      climatology.
    input_variable_names: Names of the input variables to be debiased.
    input_member_indexer: The index for the ensemble members from which the data
      is extracted. LENS2 dataset has 100 ensemble members under different
      modelling choices.
    output_variables: The names of the output variables to be debiased.
    output_dataset_path: Dataset path for the output (ERA5) dataset.
    output_climatology: Dataset path for the output (ERA5) climatology.
    time_to_channel: Whether to add an extra dimension to the batch for the time
      dimension, or to collapse the time and channel dimensions in the channel
      dimension.
    normalize_stats: Whether to normalize the statistics.
    overlapping_chunks: Whether to use overlapping chunks when extracting the
      data.
    time_coherent: Whether to use the time-coherent, i.e., sequence-to-sequence
      data loader or snapshot-to-snapshot data loader.
  """

  date_range: tuple[str, str]
  batch_size: int
  time_batch_size: int
  chunk_size: int
  shuffle: bool
  worker_count: int
  input_dataset_path: str
  input_climatology: str
  input_mean_stats_path: str
  input_std_stats_path: str
  input_variable_names: tuple[str, ...]
  input_member_indexer: tuple[dict[str, str], ...]
  output_variables: dict[str, dict[str, Any] | None] | types.MappingProxyType
  output_dataset_path: str
  output_climatology: str
  time_to_channel: bool
  normalize_stats: bool
  overlapping_chunks: bool
  time_coherent: bool


def get_train_eval_dataloaders_config(
    config: ml_collections.ConfigDict,
) -> tuple[DataLoadersConfig, DataLoadersConfig]:
  """Returns the data loaders config from the config file."""

  if config.get("use_3d_model", False):
    time_to_channel = False
  else:
    time_to_channel = True

  common_dict_config = dict(
      time_batch_size=config.get("time_batch_size", default=1),
      chunk_size=config.get("chunk_size"),
      shuffle=config.get("shuffle", default=True),
      worker_count=config.get("worker_count", default=0),
      input_dataset_path=config.get("input_dataset_path"),
      input_climatology=config.get("input_climatology"),
      input_mean_stats_path=config.get("input_mean_stats_path"),
      input_std_stats_path=config.get("input_std_stats_path"),
      input_variable_names=config.get("input_variable_names"),
      input_member_indexer=config.get("input_member_indexer"),
      output_variables=config.get("output_variables"),
      output_dataset_path=config.get("output_dataset_path"),
      output_climatology=config.get("output_climatology"),
      time_to_channel=time_to_channel,
      normalize_stats=config.get("normalize_stats"),
      overlapping_chunks=config.get("overlapping_chunks"),
      time_coherent=config.get("time_coherent"),
  )

  train_dict_config = dict(
      date_range=config.get("date_range_train"),
      batch_size=config.get("batch_size_train"),
      **common_dict_config,
  )
  eval_dict_config = dict(
      date_range=config.get("date_range_eval"),
      batch_size=config.get("batch_size_eval"),
      **common_dict_config,
  )
  return (
      DataLoadersConfig(**train_dict_config),
      DataLoadersConfig(**eval_dict_config),
  )


def build_dataloader_from_config(
    config: DataLoadersConfig,
) -> pygrain.DataLoader:
  """Builds a dataloader for the training or evaluation.

  Args:
    config: The configuration for the data loaders.

  Returns:
    A pygrain dataloaders.
  """
  if not config.time_coherent:
    logging.info("Using non-time-coherent data loader")
    # Defines the dataloaders directly.
    dataloader = create_ensemble_lens2_era5_chunked_loader_with_climatology(
        date_range=config.date_range,
        batch_size=config.batch_size,
        chunk_size=config.chunk_size,
        shuffle=True,
        worker_count=config.worker_count,
        input_dataset_path=config.input_dataset_path,
        input_climatology=config.input_climatology,
        input_mean_stats_path=config.input_mean_stats_path,
        input_std_stats_path=config.input_std_stats_path,
        input_variable_names=config.input_variable_names,
        input_member_indexer=config.input_member_indexer,
        output_variables=config.output_variables,
        output_dataset_path=config.output_dataset_path,
        output_climatology=config.output_climatology,
    )
  else:
    logging.info("Using time-coherent data loader.")
    dataloader = (
        create_ensemble_lens2_era5_time_chunked_loader_with_climatology(
            date_range=config.date_range,
            batch_size=config.batch_size,
            chunk_size=config.chunk_size,
            shuffle=True,
            worker_count=config.worker_count,
            input_dataset_path=config.input_dataset_path,
            input_climatology=config.input_climatology,
            input_mean_stats_path=config.input_mean_stats_path,
            input_std_stats_path=config.input_std_stats_path,
            input_variable_names=config.input_variable_names,
            input_member_indexer=config.input_member_indexer,
            output_variables=config.output_variables,
            time_batch_size=config.time_batch_size,
            output_dataset_path=config.output_dataset_path,
            output_climatology=config.output_climatology,
            time_to_channel=config.time_to_channel,
        )
    )

  return dataloader


def build_inference_dataloader(
    config: ml_collections.ConfigDict,
    config_eval: ml_collections.ConfigDict,
    batch_size: int,
    lens2_member_indexer: tuple[dict[str, str], ...] | None = None,
    lens2_variable_names: tuple[str, ...] | None = None,
    era5_variables: dict[str, dict[str, Any]] | None = None,
    date_range: tuple[str, str] | None = None,
    regime: Literal["train", "eval", "test"] = "test",
) -> pygrain.DataLoader:
  """Loads the data loaders.

  Args:
    config: The config file used for the training experiment.
    config_eval: The config file used for the evaluation experiment.
    batch_size: The batch size.
    lens2_member_indexer: The member indexer for the LENS2 dataset, here each
      member indexer is a dictionary with the key "member" and the value is the
      name of the member. In general we will usee only one member indexer at a
      time.
    lens2_variable_names: The names of the variables in the LENS2 dataset.
    era5_variables: The names of the variables in the ERA5 dataset.
    date_range: The date range for the evaluation.
    regime: The regime for the evaluation.

  Returns:
    The dataloader for the inference loop.
  """

  if date_range is None:
    logging.info("Using the default date ranges.")
    if regime == "train":
      date_range = config.date_range_train
    elif regime == "eval" or regime == "test":
      date_range = config.date_range_eval

  # Variables names (if not provided) are extracted from the experiment config
  # file. As they should match the ones used for the training.
  if lens2_variable_names is None:
    lens2_variable_names = config.get(
        "lens2_variable_names", _LENS2_VARIABLE_NAMES
    )

  # In not provided, we use the default values in the config files.
  if era5_variables is None:
    era5_variables = config.get("era5_variables", _ERA5_VARIABLES)
  if isinstance(era5_variables, ml_collections.ConfigDict):
    era5_variables = era5_variables.to_dict()

  # The rest of the variables are extracted from the evaluation config file.
  if lens2_member_indexer is None:
    lens2_member_indexer = config_eval.get(
        "lens2_member_indexer", _LENS2_MEMBER_INDEXER
    )
    if isinstance(lens2_member_indexer, ml_collections.ConfigDict):
      lens2_member_indexer = lens2_member_indexer.to_dict()

  # Extract the paths from the config file or use the default values.
  input_dataset_path = config_eval.get(
      "input_dataset_path", default=_LENS2_DATASET_PATH
  )
  input_climatology = config_eval.get(
      "input_climatology", default=_LENS2_STATS_PATH
  )
  input_mean_stats_path = config_eval.get(
      "input_mean_stats_path", default=_LENS2_MEAN_CLIMATOLOGY_PATH
  )
  input_std_stats_path = config_eval.get(
      "input_std_stats_path", default=_LENS2_STD_CLIMATOLOGY_PATH
  )
  output_dataset_path = config_eval.get(
      "output_dataset_path", default=_ERA5_DATASET_PATH
  )
  output_climatology = config_eval.get(
      "output_climatology", default=_ERA5_STATS_PATH
  )

  logging.info("input_dataset_path: %s", input_dataset_path)
  logging.info("input_climatology: %s", input_climatology)
  logging.info("input_mean_stats_path: %s", input_mean_stats_path)
  logging.info("input_std_stats_path: %s", input_std_stats_path)
  logging.info("output_dataset_path: %s", output_dataset_path)
  logging.info("output_climatology: %s", output_climatology)

  inference_mode = True if regime == "test" else False

  inference_dataloader = create_ensemble_lens2_era5_loader_with_climatology(
      date_range=date_range,
      batch_size=batch_size,
      shuffle=False,
      input_dataset_path=input_dataset_path,
      input_climatology=input_climatology,
      input_mean_stats_path=input_mean_stats_path,
      input_std_stats_path=input_std_stats_path,
      input_variable_names=lens2_variable_names,
      input_member_indexer=lens2_member_indexer,
      output_dataset_path=output_dataset_path,
      output_climatology=output_climatology,
      output_variables=era5_variables,
      time_stamps=True,
      inference_mode=inference_mode,  # Using the inference dataset.
      num_epochs=1,  # This is so the loop stops automatically.
  )

  return inference_dataloader
