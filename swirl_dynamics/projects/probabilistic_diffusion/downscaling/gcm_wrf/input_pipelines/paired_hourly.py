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

"""Paired GCM-WRF hourly dataset loading functions."""

from collections.abc import MutableMapping, Sequence
import copy
from typing import Any, Literal, SupportsIndex

from absl import logging
from etils import epath
import grain.python as pygrain
import immutabledict
import numpy as np
import pandas as pd
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import data_utils
import xarray as xr
import xarray_tensorstore as xrts

FlatFeatures = MutableMapping[str, Any]
XarrayLibrary = Any

# Index range defining the 9 km domain (D2) inside the 45 km domain (D1).
D2_WITHIN_D1 = {"south_north": range(16, 86), "west_east": range(32, 89)}

# Model calendar constraints for each Earth System Model considered.
DATE_CONSTRAINTS = immutabledict.immutabledict({
    "canesm5": {"avoid_leap_days": True},
    "cesm2": {"avoid_leap_days": True},
    "cnrm-esm2-1": {},
    "ec-earth3-veg": {},
    "ukesm1-0-ll": {"homogenize_months": True},
    "mpi-esm1-2-hr": {},
    "access-cm2": {},
    "miroc6": {},
    "noresm2-mm": {"avoid_leap_days": True},
    "taiesm1": {"avoid_leap_days": True},
})


def _impose_model_calendar(
    model_ds_path: epath.PathLike, date_range: pd.DatetimeIndex
) -> xr.Dataset:
  """Imposes model calendar constraints on a DatetimeIndex.

  Two types of constraints are considered:
   - Avoidance of February 29th on leap years.
   - Calendars with a fixed number of days in a month, 30.

  Args:
    model_ds_path: The path to the model dataset. The model name is extracted
      from the path and used to select the corresponding constraints.
    date_range: The date range to impose the constraints on.

  Returns:
    The date range with the constraints imposed.
  """
  for model_name, constraint in DATE_CONSTRAINTS.items():
    if model_name in model_ds_path:
      if "homogenize_months" in constraint.keys():
        date_range = date_range[date_range.day != 31]
      if "avoid_leap_days" in constraint.keys():
        date_range = date_range[
            ~((date_range.month == 2) & (date_range.day == 29))
        ]
  return date_range


class DataSource:
  """A data source that loads paired hourly GCM-WRF data."""

  def __init__(
      self,
      date_range: tuple[str, str],
      input_dataset: epath.PathLike,
      input_variables: Sequence[str],
      output_dataset: epath.PathLike,
      output_variables: Sequence[str],
      static_input_dataset: epath.PathLike,
      static_input_variables: Sequence[str],
      forcing_dataset: epath.PathLike | None = None,
      use_temporal_inputs: bool = False,
      time_downsample: int = 1,
      resample_at_nan: bool = False,
      resample_seed: int = 9999,
      crop_input: bool = False,
      xr_: XarrayLibrary = xrts,
  ):
    """Data source constructor.

    Args:
      date_range: The date range (in days) applied. Data not falling in this
        range is ignored.
      input_dataset: The path of a zarr dataset containing the input data.
      input_variables: The variables to yield from the input dataset.
      output_dataset: The path of a zarr dataset containing the output data.
      output_variables: The variables to yield from the output dataset.
      static_input_dataset: The path of a zarr dataset containing the static
        features to be used as auxiliary inputs.
      static_input_variables: The variables to yield from the static input
        dataset.
      forcing_dataset: The path of a zarr dataset containing the GHG forcing
        data with yearly resolution.
      use_temporal_inputs: Whether to use as input sinusoidal functions of the
        hour of day and day of year.
      time_downsample: The time downsampling factor (or stride) for both the
        input and output data.
      resample_at_nan: Whether to resample when NaN is detected in the data.
      resample_seed: The random seed for resampling.
      crop_input: Whether to crop the input data to only cover the output
        domain.
      xr_: The xarray library to use to open zarr files.
    """
    date_range = pd.date_range(*date_range, freq=f"{time_downsample}H")
    date_range = _impose_model_calendar(input_dataset, date_range)
    input_ds = xr_.open_zarr(input_dataset)
    input_ds = data_utils.get_common_times_dataset(input_ds, date_range)
    if crop_input:
      input_ds = input_ds.isel(**D2_WITHIN_D1)

    self._input_arrays = [input_ds[v] for v in input_variables]

    output_ds = xr_.open_zarr(output_dataset)
    output_ds = data_utils.get_common_times_dataset(output_ds, date_range)
    self._output_arrays = [output_ds[v] for v in output_variables]
    self._output_coords = output_ds.coords

    # Pre-load normalized static features
    static_input_ds = xr_.open_zarr(static_input_dataset)
    static_features = [
        (static_input_ds[v] - static_input_ds[v].mean())
        / static_input_ds[v].std()
        for v in static_input_variables
    ]
    self._static_features = np.stack(
        [xrts.read(arr).data for arr in static_features], axis=-1
    )

    self._dates = data_utils.get_common_times(input_ds, date_range)
    self._len = input_ds.dims["time"]
    self._time_array = xrts.read(input_ds["time"]).data
    self._resample_at_nan = resample_at_nan
    self._resample_seed = resample_seed
    self.use_temporal_inputs = use_temporal_inputs

    self._forcing, self._forcing_year = None, None
    if forcing_dataset is not None:
      # Pre-load normalized forcing data
      forcing_ds = xr_.open_zarr(forcing_dataset)
      self._forcing_year = [date.year for date in forcing_ds.time.values]
      self._forcing = np.stack(
          [
              (forcing_ds[v] - forcing_ds[v].mean()) / forcing_ds[v].std()
              for v in sorted(forcing_ds.data_vars)
          ],
          axis=-1,
      )
      # input4MIP lacks the last quarter of 2014, so we use the 2015 value
      if self._forcing_year[0] == 2015:
        self._forcing_year.insert(0, 2014)
        self._forcing = np.vstack(
            [self._forcing[np.newaxis, 0, :], self._forcing]
        )

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, np.ndarray]:
    """Retrieves record and retry if NaN found."""
    idx = record_key.__index__()
    if not idx < self._len:
      raise ValueError(f"Index out of range: {idx} / {self._len - 1}")

    item = self.get_item(idx)

    if self._resample_at_nan:
      while np.isnan(item["input"]).any() or np.isnan(item["output"]).any():
        logging.info("NaN detected for day %s", str(item["time_stamp"]))

        rng = np.random.default_rng(self._resample_seed + idx)
        resample_idx = rng.integers(0, len(self))
        item = self.get_item(resample_idx)

    return item

  def get_item(self, idx: int) -> dict[str, np.ndarray]:
    """Returns the data record for a given index."""
    item = {}
    item["input"] = np.stack(
        [xrts.read(arr.isel(time=idx)).data for arr in self._input_arrays],  # pytype: disable=attribute-error
        axis=-1,
    )
    item["output"] = np.stack(
        [xrts.read(arr.isel(time=idx)).data for arr in self._output_arrays],  # pytype: disable=attribute-error
        axis=-1,
    )
    item["static_features"] = self._static_features
    item["time_stamp"] = self._time_array[idx]
    if self._forcing is not None and self._forcing_year is not None:
      idx_year = self._forcing_year.index(
          pd.Timestamp(self._time_array[idx]).year
      )
      item["forcing"] = self._forcing[np.newaxis, np.newaxis, idx_year, :]
    if self.use_temporal_inputs:
      item["temporal_input"] = np.stack(
          [
              np.cos(pd.Timestamp(self._time_array[idx]).hour * np.pi / 12.0),
              np.sin(pd.Timestamp(self._time_array[idx]).hour * np.pi / 12.0),
              np.cos(
                  pd.Timestamp(self._time_array[idx]).dayofyear
                  * 2.0
                  * np.pi
                  / 365.0
              ),
              np.sin(
                  pd.Timestamp(self._time_array[idx]).dayofyear
                  * 2.0
                  * np.pi
                  / 365.0
              ),
          ],
          axis=-1,
      )[np.newaxis, np.newaxis, ...]

    return item

  def get_dates(self):
    return self._dates

  def get_output_coords(self):
    return self._output_coords


def create_dataset(
    date_range: tuple[str, str],
    input_dataset: epath.PathLike,
    input_variables: Sequence[str],
    input_stats: epath.PathLike | None,
    output_dataset: epath.PathLike,
    output_variables: Sequence[str],
    output_stats: epath.PathLike | None,
    static_input_dataset: epath.PathLike,
    static_input_variables: Sequence[str],
    time_downsample: int = 1,
    random_maskout_probability: float = 0.0,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    resample_at_nan: bool = True,
    resample_seed: int = 9999,
    crop_input: bool = False,
    normalization: Literal["local", "global"] = "local",
    forcing_dataset: epath.PathLike | None = None,
    use_temporal_inputs: bool = False,
    xr_: XarrayLibrary = xrts,
):
  """The full GCM-WRF data pipeline."""
  source = DataSource(
      date_range=date_range,
      input_dataset=input_dataset,
      input_variables=input_variables,
      output_dataset=output_dataset,
      output_variables=output_variables,
      static_input_dataset=static_input_dataset,
      static_input_variables=static_input_variables,
      time_downsample=time_downsample,
      resample_at_nan=resample_at_nan,
      resample_seed=resample_seed,
      crop_input=crop_input,
      forcing_dataset=forcing_dataset,
      use_temporal_inputs=use_temporal_inputs,
      xr_=xr_,
  )
  cond_fields = ["input", "static_features"]
  if use_temporal_inputs:
    cond_fields.append("temporal_input")
  if forcing_dataset is not None:
    cond_fields.append("forcing")
  masked_fields = copy.copy(cond_fields)
  in_stat_kwargs = {"xr_": xr_,}
  if normalization == "local":
    read_stats_fn = data_utils.read_stats
    if crop_input:
      in_stat_kwargs["crop_dict"] = D2_WITHIN_D1
  else:
    read_stats_fn = data_utils.read_global_stats

  transformations = []
  if input_stats is not None:
    transformations.append(
        data_utils.Standardize(
            input_field="input",
            mean=read_stats_fn(
                input_stats, input_variables, "mean", **in_stat_kwargs
            ),
            std=read_stats_fn(
                input_stats, input_variables, "std", **in_stat_kwargs
            ),
        )
    )
  if output_stats is not None:
    transformations.append(
        data_utils.Standardize(
            input_field="output",
            mean=read_stats_fn(output_stats, output_variables, "mean", xr_=xr_),
            std=read_stats_fn(output_stats, output_variables, "std", xr_=xr_),
        ),
    )
  transformations.extend([
      data_utils.RandomMaskout(
          input_fields=masked_fields,
          probability=random_maskout_probability,
          fill_value=0.0,
      ),
      data_utils.ParseTrainingData(
          generate_field="output",
          channel_cond_fields=cond_fields,
      ),
  ])

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
