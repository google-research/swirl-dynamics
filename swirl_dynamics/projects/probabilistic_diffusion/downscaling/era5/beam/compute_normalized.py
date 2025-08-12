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

r"""Computes the normalized version of a dataset.

Example usage:

Computing the normalized residual between high-resolution target and the
interpolated low-resolution ERA5:
```
python swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5/beam:compute_normalized.par -- \
    --raw_path=/data/era5/selected_variables/0p25deg_minus_1p5deg_linear_hourly_7vars_windspeed_1959-2023.zarr \
    --stats_path=/data/era5/stats/0p25deg_minus_1p5deg_linear_hourly_7vars_windspeed_1961-2000_dayofyear_hour.zarr \
    --output_path=/data/era5/selected_variables/0p25deg_minus_1p5deg_linear_hourly_7vars_windspeed_1959-2023_dayofyear_hour_normalized.zarr \
    --stats_type=hourly

```

Computing the normalized input:
```
python swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5/beam:compute_normalized.par -- \
    --raw_path=/data/era5/selected_variables/1p5deg_dailymean_7vars_windspeed_1959-2023.zarr \
    --stats_path=/data/era5/stats/1p5deg_7vars_windspeed_1961-2000_all.zarr \
    --output_path=/data/era5/selected_variables/1p5deg_dailymean_7vars_windspeed_1959-2023_all_normalized.zarr \
    --stats_type=all
```


"""

import functools
from typing import TypeVar

from absl import app
from absl import flags
import apache_beam as beam
import xarray
import xarray_beam as xbeam
from zarr.google import gfile_store

Dataset = TypeVar("Dataset", xarray.Dataset, xarray.DataArray)

# Flags.
RAW_DATA_PATH = flags.DEFINE_string(
    "raw_path", None, help="Zarr path containing the unnormalized raw data."
)
STATS_PATH = flags.DEFINE_string(
    "stats_path", None, help="Zarr path containing the normalization stats."
)
STATS_TYPE = flags.DEFINE_string(
    "stats_type",
    "hourly",
    help="Type of stats. One of [`all`, `daily` or `hourly`].",
)
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path.")
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")


def normalize(
    key: xbeam.Key,
    raw_chunk: xarray.Dataset,
    stats_ds: xarray.Dataset,
    stats_type: str,
) -> tuple[xbeam.Key, xarray.Dataset]:
  """Normalizes a chunk with the corresponding mean and std."""
  indexers = {
      dim: raw_chunk[dim]
      for dim in ["longitude", "latitude", "level"]
      if dim in raw_chunk.coords
  }
  stats_chunk = stats_ds.sel(indexers=indexers)
  match stats_type:
    case "all":
      pass
    case "daily":
      stats_chunk = stats_chunk.sel(
          dayofyear=raw_chunk["time"].dt.dayofyear
      ).drop_vars(["dayofyear"])
    case "hourly":
      stats_chunk = stats_chunk.sel(
          dayofyear=raw_chunk["time"].dt.dayofyear,
          hour=raw_chunk["time"].dt.hour,
      ).drop_vars(["dayofyear", "hour"])
    case _:
      raise ValueError(
          f"Unknown stats type: {stats_type}. Currently supported: [`all`,"
          " `daily` or `hourly`]."
      )
  normalized_chunk = (
      raw_chunk - stats_chunk.sel(stats="mean", drop=True)
  ) / stats_chunk.sel(stats="std", drop=True)
  return key, normalized_chunk


def main(argv):
  raw_store = gfile_store.GFileStore(RAW_DATA_PATH.value)
  raw_ds, raw_chunks = xbeam.open_zarr(raw_store)

  stats_store = gfile_store.GFileStore(STATS_PATH.value)
  stats_ds = xarray.open_zarr(stats_store, chunks=None)

  template = xbeam.make_template(raw_ds)

  output_store = OUTPUT_PATH.value
  output_store = gfile_store.GFileStore(output_store)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(raw_ds, raw_chunks, num_threads=16)
        | beam.MapTuple(
            functools.partial(
                normalize, stats_ds=stats_ds, stats_type=STATS_TYPE.value
            )
        )
        | xbeam.ChunksToZarr(output_store, template, raw_chunks, num_threads=16)
    )


if __name__ == "__main__":
  app.run(main)
