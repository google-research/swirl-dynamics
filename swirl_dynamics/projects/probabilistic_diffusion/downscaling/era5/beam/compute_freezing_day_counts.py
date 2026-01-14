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

r"""Compute freezing day counts (Northern Hemisphere winter) for every pixel.

Computes the number of consecutive periods of N days with daily min
2 m temperature below a certain threshold. The default thresholds are
0 Celsius (32 Farenheit) and -2.22 Celsius (28 Farenheit), which are the freeze
and hard freeze thresholds respectively.

The start and end years represent the first and last "season years" considered.
A season year is defined as the year when the Northern Hemisphere winter season
starts. Therefore, the 2011 winter seasons is December 2011 to February 2012.

Example usage:

```
python swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5/beam:compute_freezing_day_counts.par -- \
    --samples=<SAMPLES_PATH> \
    --output_path=<OUTPUT_PATH> \
    --season_year_start=2011 \
    --season_year_end=2020
```

"""

import functools

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.eval import utils as eval_utils
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils as pipeline_utils
import xarray as xr
import xarray_beam as xbeam

Variable = pipeline_utils.DatasetVariable

# Flags.
SAMPLES = flags.DEFINE_string("samples", None, help="Sample dataset Zarr path.")
REFERENCE = flags.DEFINE_string(
    "reference", None, help="Reference dataset Zarr path."
)
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path.")
THRESHOLDS = flags.DEFINE_list(
    "thresholds",
    [
        "270.298",
        "273.15",
    ],
    help="Thresholds (K) for freezing day counts.",
)
LENGTHS = flags.DEFINE_list(
    "lengths",
    ["1", "3", "5", "7"],
    help="Number of days of sustained daily mins that can induce freezing.",
)
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")
SEASON_YEAR_START = flags.DEFINE_integer(
    "season_year_start", 2001, help="Starting season year for evaluation."
)
SEASON_YEAR_END = flags.DEFINE_integer(
    "season_year_end", 2010, help="Ending season year for evaluation."
)

MONTHS = flags.DEFINE_list(
    "months",
    ["12", "1", "2"],
    help=(
        "Winter season months for evaluation. Winter seasons are considered "
        "periods starting after September and ending before August."
    ),
)
NUM_THREADS = flags.DEFINE_integer(
    "num_threads", None, help="Number of threads per worker for Flume."
)

DTYPE = np.int16


SAMPLE_VARIABLES = [
    Variable("2m_temperature", None, "2mT"),
]


def count_freeze_advisories(
    key: xbeam.Key,
    ds: xr.Dataset,
    *,
    thresholds: list[float],
    lengths: list[int],
):
  """Counts freeze advisories on a single pixel chunk."""
  assert len(ds.member) == len(ds.longitude) == len(ds.latitude) == 1

  # Create a template for the season_year_time daily coordinate.
  time_template = eval_utils.to_winter_season_years(
      ds.time.resample(time="D").min()
  )

  ds = eval_utils.to_winter_season_years(ds)

  count = np.zeros((len(thresholds), len(lengths)))
  for year in np.unique(ds.season_year_time.dt.year):

    t2m = ds["2mT"].sel(
        season_year_time=ds.season_year_time.dt.year.isin([year]), drop=True
    )
    # Resample to daily min, requires monotonic coordinate.
    t2m = t2m.sortby("season_year_time")
    ds_min = t2m.resample(season_year_time="D").min(skipna=True)

    # Recover seasonal ordering.
    season_year_template = time_template.sel(
        season_year_time=time_template.season_year_time.dt.year.isin([year]),
        drop=True,
    )
    ds_min = ds_min.reindex_like(season_year_template)

    ds_min = ds_min.where(
        ds_min.season_year_time.isin(ds.season_year_time.values), drop=True
    )
    ts_min = ds_min.to_numpy()
    assert ts_min.ndim == 4  # (member, season_year_time, lon, lat)
    ts_min = np.squeeze(ts_min)
    assert ts_min.ndim == 1
    count += eval_utils.count_below_threshold(ts_min, thresholds, lengths)

  count = count[np.newaxis, np.newaxis, np.newaxis, :]
  count_da = xr.DataArray(
      data=count.astype(DTYPE),
      dims=("member", "longitude", "latitude", "thresholds", "lengths"),
      coords={
          "member": ds.coords["member"].values,
          "longitude": ds.coords["longitude"].values,
          "latitude": ds.coords["latitude"].values,
          "thresholds": np.asarray(thresholds),
          "lengths": np.asarray(lengths),
      },
  )
  count_ds = xr.Dataset({"freeze_day_counts": count_da})

  new_key = key.with_offsets(time=None, thresholds=0, lengths=0)
  return new_key, count_ds


def main(argv):

  # Include next year to get January, February, ... of last season.
  years = list(range(SEASON_YEAR_START.value, SEASON_YEAR_END.value + 2))
  months = [int(m) for m in MONTHS.value]
  thresholds = [float(t) for t in THRESHOLDS.value]
  lengths = [int(l) for l in LENGTHS.value]

  sample_ds = xr.open_zarr(SAMPLES.value, consolidated=True)
  sample_ds = eval_utils.select_time(sample_ds, years, months)

  sample_vars = [var.rename for var in SAMPLE_VARIABLES]
  for var in sample_vars:
    if var not in sample_ds:
      raise ValueError(
          f"Variable {var} expected but not found in sample dataset."
      )
  sample_ds = sample_ds[sample_vars]

  # If a reference path is provided, the script will compute statistics on
  # the reference dataset instead of the inference dataset.
  if REFERENCE.value is not None:
    sample_ds = eval_utils.get_reference_ds(
        REFERENCE.value, SAMPLE_VARIABLES, sample_ds
    )
    sample_ds = eval_utils.select_time(sample_ds, years, months)

  in_chunks = {
      "time": -1,
      "member": -1,
      "longitude": 1,
      "latitude": 1,
  }
  working_chunks_in = {
      "time": -1,
      "member": 1,
      "longitude": 1,
      "latitude": 1,
  }
  working_chunks_out = {
      "member": 1,
      "longitude": 1,
      "latitude": 1,
      "thresholds": -1,
      "lengths": -1,
  }
  target_chunks = {
      "member": -1,
      "longitude": -1,
      "latitude": -1,
      "thresholds": 1,
      "lengths": 1,
  }

  out_sizes = dict(sample_ds.sizes)
  out_sizes.update({
      "thresholds": len(thresholds),
      "lengths": len(lengths),
  })
  del out_sizes["time"]

  # Create template.
  template_ds = xr.Dataset(
      data_vars={
          "freeze_day_counts": (
              ["member", "longitude", "latitude", "thresholds", "lengths"],
              np.empty(
                  (
                      sample_ds.sizes["member"],
                      sample_ds.sizes["longitude"],
                      sample_ds.sizes["latitude"],
                      len(thresholds),
                      len(lengths),
                  ),
                  dtype=DTYPE,
              ),
              {
                  "units": (
                      "count"
                      f" ({SEASON_YEAR_START.value}-{SEASON_YEAR_END.value})"
                  ),
                  "long_name": (
                      "Freeze day counts from daily min 2 m temperature"
                      f" in the months {months[0]}-{months[-1]},"
                      f" ({SEASON_YEAR_START.value}-{SEASON_YEAR_END.value})"
                  ),
              },
          )
      },
      coords={
          "member": sample_ds["member"].values,
          "longitude": sample_ds["longitude"].values,
          "latitude": sample_ds["latitude"].values,
          "thresholds": np.asarray(thresholds),
          "lengths": np.asarray(lengths),
      },
  )
  template = xbeam.make_template(template_ds)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(
            sample_ds, in_chunks, num_threads=NUM_THREADS.value
        )
        | xbeam.SplitChunks(working_chunks_in)
        | beam.MapTuple(
            functools.partial(
                count_freeze_advisories,
                thresholds=thresholds,
                lengths=lengths,
            )
        )
        | xbeam.Rechunk(
            dim_sizes=out_sizes,
            source_chunks=working_chunks_out,
            target_chunks=target_chunks,
            itemsize=np.dtype(DTYPE).itemsize,
            max_mem=2**29,  # 512MB
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=template,
            zarr_chunks=target_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == "__main__":
  app.run(main)
