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

r"""Calculate daily max stats for an ERA5 dataset.

The stats represent smoothed average daily max value within a year range. Also
supports downsampling first before computing the stats.

Example usage:

```
python swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5/beam:compute_dailymax_stats.par -- \
    --input_path=/data/era5/selected_variables/0p25deg_hourly_7vars_windspeed_1959-2023.zarr \
    --output_path=/data/era5/stats/0p25deg_7vars_windspeed_2hourly_dailymax_1961-2000_copy.zarr \
    --downsample=2 \
    --start_year=1961 \
    --end_year=2000
```


"""

import functools
from typing import TypeVar

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam as xbeam

Dataset = TypeVar("Dataset", xarray.Dataset, xarray.DataArray)

# Flags.
INPUT_PATH = flags.DEFINE_string("input_path", None, help="Input Zarr path.")
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path.")
START_YEAR = flags.DEFINE_string("start_year", None, "Start of time range.")
END_YEAR = flags.DEFINE_string("end_year", None, "End of time range.")
WINDOW_SIZE = flags.DEFINE_integer("window_size", 61, "Window size.")
DOWNSAMPLE = flags.DEFINE_integer("downsample", 1, "Downsample factor.")
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")

# Constants.
WORKING_CHUNKS = {"latitude": 4, "longitude": 4, "level": 1, "time": -1}
VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "specific_humidity",
    "mean_sea_level_pressure",
    "total_precipitation",
    "geopotential",
]


def replace_time_with_doy(ds: xarray.Dataset) -> xarray.Dataset:
  """Replace time coordinate with days of the year."""
  return ds.assign_coords({"time": ds.time.dt.dayofyear}).rename(
      {"time": "dayofyear"}
  )


def create_window_weights(window_size: int) -> xarray.DataArray:
  """Create linearly decaying window weights."""
  assert window_size % 2 == 1, "Window size must be odd."
  half_window_size = window_size // 2
  window_weights = np.concatenate([
      np.linspace(0, 1, half_window_size + 1),
      np.linspace(1, 0, half_window_size + 1)[1:],
  ])
  window_weights = window_weights / window_weights.mean()
  window_weights = xarray.DataArray(window_weights, dims=["window"])
  return window_weights


def compute_windowed_stat(
    ds: xarray.Dataset, window_weights: xarray.DataArray
) -> xarray.Dataset:
  """Compute rolling-windowed statistics."""
  window_size = len(window_weights)
  half_window_size = window_size // 2
  stacked = xarray.concat(
      [
          replace_time_with_doy(ds.sel(time=str(year)))
          for year in np.unique(ds.time.dt.year)
      ],
      dim="year",
  )
  # Fill gap day (366) with values from previous day 365
  stacked = stacked.fillna(stacked.sel(dayofyear=365))
  # Pad edges for perioding window
  stacked = stacked.pad(pad_width={"dayofyear": half_window_size}, mode="wrap")
  # Weighted rolling mean
  stacked = stacked.rolling(dayofyear=window_size, center=True).construct(
      "window"
  )
  rolling_mean = (
      stacked.weighted(window_weights)
      .mean(dim=("window", "year"))
      .isel(dayofyear=slice(half_window_size, -half_window_size))
  )
  return rolling_mean


def compute_chunk_stats(
    key: xbeam.Key,
    chunk: xarray.Dataset,
    *,
    window_size: int,
) -> tuple[xbeam.Key, xarray.Dataset]:
  """Compute stats for a chunk."""
  dims = (
      ["longitude", "latitude", "level"]
      if "level" in chunk.dims
      else ["longitude", "latitude"]
  )
  new_key = key.with_offsets(time=None, dayofyear=0)
  chunk = chunk.resample(time="D").max()
  output_chunk = compute_windowed_stat(
      ds=chunk, window_weights=create_window_weights(window_size)
  )
  output_chunk = output_chunk.transpose("dayofyear", *dims)
  return new_key, output_chunk


def main(argv):
  input_store = INPUT_PATH.value
  source_dataset, source_chunks = xbeam.open_zarr(
      input_store, consolidated=True
  )

  # Enforce year range.
  start = np.datetime64(START_YEAR.value, "D") if START_YEAR.value else None
  end = np.datetime64(END_YEAR.value, "D") if END_YEAR.value else None
  # NOTE: Rechunk does not like using a subset of variables only.
  source_dataset = source_dataset.sel(time=slice(start, end))
  source_dataset = source_dataset.isel(time=slice(None, None, DOWNSAMPLE.value))

  # Create template.
  template_ds = source_dataset.isel(time=0, drop=True)
  template_ds["latitude"] = np.sort(template_ds["latitude"].to_numpy())
  template_ds["longitude"] = np.sort(template_ds["longitude"].to_numpy())

  template_ds = template_ds.expand_dims(dayofyear=np.arange(366) + 1, axis=1)
  working_time_chunks = {"dayofyear": -1}
  output_time_chunks = {"dayofyear": 1}

  template_ds = template_ds.transpose(
      *working_time_chunks.keys(), "longitude", "latitude", "level"
  )
  template = xbeam.make_template(template_ds)

  # Define intermediate and final chunks.
  stats_working_chunks = dict(**WORKING_CHUNKS, **working_time_chunks)
  del stats_working_chunks["time"]

  output_chunks = {k: v for k, v in source_chunks.items() if k != "time"}
  output_chunks.update(output_time_chunks)

  output_store = OUTPUT_PATH.value

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks, num_threads=16)
        | xbeam.SplitVariables()
        | "Rechunk input"
        >> xbeam.Rechunk(
            source_dataset.sizes,  # pytype: disable=wrong-arg-types
            source_chunks,
            WORKING_CHUNKS,
            itemsize=4,
            max_mem=2**30 * 16,
        )
        | beam.MapTuple(
            functools.partial(
                compute_chunk_stats, window_size=WINDOW_SIZE.value
            )
        )
        | "Rechunk output"
        >> xbeam.Rechunk(
            template.sizes,  # pytype: disable=wrong-arg-types
            stats_working_chunks,
            output_chunks,
            itemsize=4,
            max_mem=2**30 * 16,
        )
        | xbeam.ChunksToZarr(
            output_store, template, output_chunks, num_threads=16
        )
    )


if __name__ == "__main__":
  app.run(main)
