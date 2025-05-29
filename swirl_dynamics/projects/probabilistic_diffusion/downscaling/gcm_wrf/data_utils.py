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

"""Utils for manipulating and loading data."""

from collections.abc import Callable, MutableMapping, Sequence
import dataclasses
from typing import Any, Literal

from etils import epath
import grain.python as pygrain
import numpy as np
import pandas as pd
from scipy import interpolate
import xarray as xr
import xarray_tensorstore as xrts

FlatFeatures = MutableMapping[str, Any]
XarrayLibrary = Any


def positive_lon_to_westeast(
    lon: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
  """Converts longitudes from [0, 360] to [-180, 180]."""
  return xr.where(lon > 180, lon - 360, lon)


def westeast_to_positive_lon(
    lon: np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
  """Converts longitudes from [-180, 180] to [0, 360]."""
  return xr.where(lon < 0, lon + 360, lon)


def get_common_times(
    ds: xr.Dataset, date_range: pd.DatetimeIndex
) -> pd.DatetimeIndex:
  """Returns the time intersection between a dataset and a date range.

  It is assumed that the dataset contains a `time` coordinate.

  Args:
    ds: An xarray dataset.
    date_range: The date range used to intersect with the dataset.

  Returns:
    The time intersection between the dataset and the date range.
  """
  ds_times = pd.to_datetime([pd.Timestamp(date) for date in ds.time.values])
  return pd.DatetimeIndex.intersection(ds_times, date_range)


def align_datasets(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset]:
  """Returns input datasets aligned in time."""
  ds2_times = pd.to_datetime([pd.Timestamp(date) for date in ds2.time.values])
  common_times = get_common_times(ds1, ds2_times)
  return ds1.sel(time=common_times), ds2.sel(time=common_times)  # pytype: disable=bad-return-type


def get_common_times_dataset(
    ds: xr.Dataset, date_range: pd.DatetimeIndex
) -> pd.DatetimeIndex:
  """Filters a dataset to cover dates contained in the given date_range."""
  times = get_common_times(ds, date_range)
  return ds.sel(time=times)


def replace_time_with_doy(ds: xr.Dataset) -> xr.Dataset:
  """Replace time coordinate with days of year."""
  return ds.assign_coords({'time': ds.time.dt.dayofyear}).rename(
      {'time': 'dayofyear'}
  )


def select_hour(ds: xr.Dataset, hour: int) -> xr.Dataset:
  """Select given hour of day from Dataset."""
  ds = ds.isel(time=ds.time.dt.hour == hour)
  # Adjust time dimension
  ds = ds.assign_coords({'time': ds.time.astype('datetime64[D]')})
  return ds


def create_window_weights(window_size: int) -> xr.DataArray:
  """Create linearly decaying window weights."""
  assert window_size % 2 == 1, 'Window size must be odd.'
  half_window_size = window_size // 2
  window_weights = np.concatenate([
      np.linspace(0, 1, half_window_size + 1),
      np.linspace(1, 0, half_window_size + 1)[1:],
  ])
  window_weights = window_weights / window_weights.mean()
  window_weights = xr.DataArray(window_weights, dims=['window'])
  return window_weights


def compute_rolling_stat(
    ds: xr.Dataset,
    window_weights: xr.DataArray,
    stat_fn: str | Callable[..., xr.Dataset] = 'mean',
) -> xr.Dataset:
  """Compute rolling climatological statistic.

  Args:
    ds: Dataset to compute climatological statistics over.
    window_weights: Weights used to aggregate data from contiguous days. Linear
      weights can be obtained from `create_window_weights`.
    stat_fn: Climatological statistic to compute. It can be a string ('mean',
      'std') or a function.

  Returns:
    The climatological statistic.
  """
  window_size = len(window_weights)
  half_window_size = window_size // 2  # For padding
  # Stack years
  stacked = xr.concat(
      [
          replace_time_with_doy(ds.sel(time=str(y)))
          for y in np.unique(ds.time.dt.year)
      ],
      dim='year',
  )
  # Fill gap day (366) with values from previous day 365
  stacked = stacked.fillna(stacked.sel(dayofyear=365))
  # Pad edges for perioding window
  stacked = stacked.pad(pad_width={'dayofyear': half_window_size}, mode='wrap')
  # Weighted rolling mean
  stacked = stacked.rolling(dayofyear=window_size, center=True).construct(
      'window'
  )
  if stat_fn == 'mean':
    rolling_stat = stacked.weighted(window_weights).mean(dim=('window', 'year'))
  elif stat_fn == 'std':
    rolling_stat = stacked.weighted(window_weights).std(dim=('window', 'year'))
  else:
    rolling_stat = stat_fn(
        stacked, weights=window_weights, dim=('window', 'year')
    )
  # Remove edges
  rolling_stat = rolling_stat.isel(
      dayofyear=slice(half_window_size, -half_window_size)
  )
  return rolling_stat


def compute_daily_stat(
    obs: xr.Dataset,
    window_size: int,
    clim_years: slice,
    stat_fn: str | Callable[..., xr.Dataset] = 'mean',
) -> xr.Dataset:
  """Compute daily average climatology with running window."""
  # NOTE: Loading seems to be necessary, otherwise computation takes forever
  # Will be converted to xarray-beam pipeline anyway
  obs = obs.load()
  obs_daily = obs.sel(time=clim_years).resample(time='D').mean()
  window_weights = create_window_weights(window_size)
  daily_rolling_clim = compute_rolling_stat(obs_daily, window_weights, stat_fn)
  return daily_rolling_clim


def compute_hourly_stat(
    obs: xr.Dataset,
    window_size: int,
    clim_years: slice,
    hour_interval: int,
    stat_fn: str | Callable[..., xr.Dataset] = 'mean',
) -> xr.Dataset:
  """Compute climatology by day of year and hour of day."""
  obs = obs.compute()
  hours = xr.DataArray(range(0, 24, hour_interval), dims=['hour'])
  window_weights = create_window_weights(window_size)

  hourly_rolling_clim = xr.concat(
      [  # pylint: disable=g-complex-comprehension
          compute_rolling_stat(
              select_hour(obs.sel(time=clim_years), hour),
              window_weights,
              stat_fn,
          )
          for hour in hours
      ],
      dim=hours,
  )
  return hourly_rolling_clim


def read_stats(
    stats_path: epath.PathLike,
    variables: Sequence[str],
    field: Literal['mean', 'std'],
    crop_dict: dict[str, Any] | None = None,
    xr_: XarrayLibrary = xrts,
) -> np.ndarray:
  """Reads variable statistics from zarr and returns as a stacked array.

  The statistics are expected to be stored as xarray variables with a `stats`
  dimension that can take the values `mean` or `std`; see
  compute_global_stats.py for how to generate such datasets.

  Args:
    stats_path: The path to the stats zarr dataset.
    variables: List of variables from which statistics are to be retrieved.
    field: Statistic to retrieve, either the mean or std.
    crop_dict: A dictionary of dimension indices used to crop the input stats
      dataset. Passed to `xarray.Dataset.isel`.
    xr_: The xarray library to use to open zarr files.

  Returns:
    A stacked array of statistics.
  """
  ds = xr_.open_zarr(stats_path)
  if crop_dict is not None:
    ds = ds.isel(**crop_dict)
  out = np.stack(
      [ds[var].sel({'stats': field}).to_numpy() for var in variables], axis=-1
  )
  return out


def read_global_stats(
    stats_path: epath.PathLike,
    variables: Sequence[str],
    field: Literal['mean', 'std'],
    xr_: XarrayLibrary = xrts,
    **sel_kwargs,
) -> np.ndarray:
  """Reads global variable stats from zarr and returns as a stacked array.

  Stacking is done first for each variable along the `level` dimension,
  if present. Stacking is then done along the `variable` dimension. For
  example, if the input variables are `[u, t2]`, `u` has levels `[0, 1, 2]` and
  `t2` has no levels, the output will be an array with shape
  `[..., 4]`, where the last dimension stacks `[u_0, u_1, u_2, t2]]`.

  Args:
    stats_path: The path to the stats zarr dataset. The dataset is assumed to
      contain global statistical moments of all variables, stored as xarray
      variables with names `variable_first`, `variable_second`; see
      compute_global_stats.py for how to generate such datasets.
    variables: List of variables from which stats are to be retrieved.
    field: Statistic to retrieve, either the mean or std.
    xr_: The xarray library to use to open zarr files.
    **sel_kwargs: Dictionary passed to `xarray.Dataset.sel` when loading the
      stats dataset.

  Returns:
    A stacked array of statistics.
  """
  ds = xr_.open_zarr(stats_path).sel(**sel_kwargs)
  out = []
  for var in variables:
    if field == 'mean':
      var_array = ds[var + '_first'].to_numpy().squeeze()
    else:
      var_array = (
          np.sqrt(ds[var + '_second'] - ds[var + '_first'] * ds[var + '_first'])
          .to_numpy()
          .squeeze()
      )

    if 'level' in ds[var + '_first'].dims:
      out.extend(var_array)
    else:
      out.append(var_array)

  return np.stack(out, axis=-1)


@dataclasses.dataclass(frozen=True)
class Standardize(pygrain.MapTransform):
  """Standardize an input_field pixel-wise using pre-computed mean and std."""

  input_field: str
  mean: np.ndarray
  std: np.ndarray

  def map(self, features: FlatFeatures) -> FlatFeatures:
    features[self.input_field] = (
        features[self.input_field] - self.mean
    ) / self.std
    return features


@dataclasses.dataclass(frozen=True)
class RandomMaskout(pygrain.RandomMapTransform):
  """Randomly mask out selected fields with a given value.

  All input fields are replaced with a uniform fill value based on the outcome
  of a Bernoulli draw.
  """

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
class Interpolate(pygrain.MapTransform):
  """Interpolate a selected input field to a new grid.

  This transform assumes that the input field has as the last dimension a
  channel dimension.

  Attributes:
    input_field: The name of the input field to interpolate.
    source_grid: The source grid coordinates, formatted as a tuple of length `D`
      of 1-D arrays, where `D` is the number of dimensions of the input field to
      be interpolated. Each array `i` in the tuple defines the coordinates of
      each input field value along its dimension `i`, as expected by
      `scipy.interpolate.griddata`. For two-dimensional data, D=2, and the two
      arrays may define the latitude and longitude of each data point.
    target_grid: The target grid, with the same format as `source_grid`.
    target_shape: The target shape, excluding batch dimensions.
    method: The interpolation method.
  """

  input_field: str
  source_grid: tuple[np.ndarray, ...]
  target_grid: tuple[np.ndarray, ...]
  target_shape: tuple[int, ...]
  method: str

  def map(self, features: FlatFeatures) -> FlatFeatures:

    inputs = features[self.input_field]
    interp_input_list = []
    for channel in range(inputs.shape[-1]):
      feature = inputs[..., channel]
      shape = feature.shape[: -len(self.target_shape)] + self.target_shape
      interp_input_list.append(
          interpolate.griddata(
              self.source_grid,
              feature.flatten(),
              self.target_grid,
              method=self.method,
          ).reshape(shape)
      )
    features[self.input_field] = np.stack(interp_input_list, axis=-1)
    return features


@dataclasses.dataclass(frozen=True)
class ParseTrainingData(pygrain.MapTransform):
  """Returns data in the form required by the model for training.

  The model expects dictionaries of data with key `x` for the target
  and `cond` for the conditioning inputs. The key `cond` also indexes a
  dictionary of conditioning inputs, with keys `channel:<field>` for each
  conditioning field.
  """

  generate_field: str
  channel_cond_fields: Sequence[str]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    processed = {}
    processed['x'] = features[self.generate_field]
    processed['cond'] = {
        f'channel:{field}': features[field]
        for field in self.channel_cond_fields
    }
    return processed


@dataclasses.dataclass(frozen=True)
class ParseInferenceData(pygrain.MapTransform):
  """Returns data in the form required by the model to do inference.

  The model expects dictionaries of data with key `cond` for the conditioning
  inputs. The key `cond` also indexes a dictionary of conditioning inputs,
  with keys `channel:<field>` for each conditioning field.
  """

  channel_cond_fields: Sequence[str]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    processed = {}
    processed['cond'] = {
        f'channel:{field}': features[field]
        for field in self.channel_cond_fields
    }
    return processed
