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

"""Utils for manipulating and loading data."""

from collections.abc import MutableMapping, Sequence
import dataclasses
from typing import Any, Literal

from etils import epath
import grain.python as pygrain
import numpy as np
import pandas as pd
import xarray as xr
import xarray_tensorstore as xrts

FlatFeatures = MutableMapping[str, Any]


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


def read_stats(
    stats_path: epath.PathLike,
    variables: Sequence[str],
    field: Literal["mean", "std"],
    crop_dict: dict[str, Any] | None = None,
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

  Returns:
    A stacked array of statistics.
  """
  ds = xrts.open_zarr(stats_path)
  if crop_dict is not None:
    ds = ds.isel(**crop_dict)
  out = np.stack(
      [ds[var].sel({"stats": field}).to_numpy() for var in variables], axis=-1
  )
  return out


def read_global_stats(
    stats_path: epath.PathLike,
    variables: Sequence[str],
    field: Literal["mean", "std"],
) -> np.ndarray:
  """Reads global variable stats from zarr and returns as a stacked array.

  Args:
    stats_path: The path to the stats zarr dataset. The dataset is assumed to
      contain global statistical moments of all variables, stored as xarray
      variables with names `variable_first`, `variable_second`; see
      compute_global_stats.py for how to generate such datasets.
    variables: List of variables from which stats are to be retrieved.
    field: Statistic to retrieve, either the mean or std.

  Returns:
    A stacked array of statistics.
  """
  ds = xrts.open_zarr(stats_path)

  if field == "mean":
    out = np.stack(
        [ds[var + "_first"].to_numpy().squeeze() for var in variables], axis=-1
    )
  else:
    out = np.stack(
        [
            np.sqrt(
                ds[var + "_second"] - ds[var + "_first"] * ds[var + "_first"]
            )
            .to_numpy()
            .squeeze()
            for var in variables
        ],
        axis=-1,
    )
  return out


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
class ParseTrainingData(pygrain.MapTransform):
  """Returns data in the form required by the model.

  The model expects dictionaries of data with key `x` for the target
  and `cond` for the conditioning inputs. The key `cond` also indexes a
  dictionary of conditioning inputs, with keys `channel:<field>` for each
  conditioning field.
  """

  generate_field: str
  channel_cond_fields: Sequence[str]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    processed = {}
    processed["x"] = features[self.generate_field]
    processed["cond"] = {
        f"channel:{field}": features[field]
        for field in self.channel_cond_fields
    }
    return processed
