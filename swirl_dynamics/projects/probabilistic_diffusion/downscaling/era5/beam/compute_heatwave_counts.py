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

r"""Compute heatwave counts for every pixel.

Example usage:

```
python swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5/beam:compute_heatwave_counts.par -- \
    --samples=/data/era5/inference/conus/bcsd_jja10s_ref61to00_8samples_pixel_chunks.zarr \
    --output_path=/genbcsr_eval/conus/bcsd_100members_8samples_jja10s/heatwave/unnormalized_counts.zarr \
    --year_start=2011 \
    --year_end=2020 \
    --months=6,7,8
```

"""

import functools
from typing import TypeVar

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.eval import utils as eval_utils
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils as pipeline_utils
import xarray as xr
import xarray_beam as xbeam

Dataset = TypeVar("Dataset", xr.Dataset, xr.DataArray)
Variable = pipeline_utils.DatasetVariable

# Flags.
SAMPLES = flags.DEFINE_string("samples", None, help="Sample dataset Zarr path.")
REFERENCE = flags.DEFINE_string(
    "reference", None, help="Reference dataset Zarr path."
)
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path.")
THRESHOLDS = flags.DEFINE_list(
    "thresholds",
    ["300", "305", "314", "327"],
    help="Thresholds (K) for heatwave counts.",
)
LENGTHS = flags.DEFINE_list(
    "lengths",
    ["1", "3", "5", "7"],
    help=(
        "Number of days of sustained heat conditions to qualify as a heatwave."
    ),
)
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")
YEAR_START = flags.DEFINE_integer(
    "year_start", 2001, help="Starting year for evaluation."
)
YEAR_END = flags.DEFINE_integer(
    "year_end", 2010, help="Ending year for evaluation."
)
MONTHS = flags.DEFINE_list(
    "months", ["6", "7", "8"], help="Months for evaluation."
)

DTYPE = np.int16

ZS_REFERENCE = "/weatherbench/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
SAMPLE_VARIABLES = [
    Variable("2m_temperature", None, "2mT"),
    Variable("10m_magnitude_of_wind", None, "10mW"),
    Variable("specific_humidity", {"level": [1000]}, "Q1000"),
    Variable("mean_sea_level_pressure", None, "MSL"),
]


def _count_heatwaves(
    x: np.ndarray, thresholds: list[float], lengths: list[int]
):
  """Counts heatwaves on a single continuous time series."""
  count = np.zeros((len(thresholds), len(lengths)))
  for i, thres in enumerate(thresholds):
    exceeds_threshold = x >= thres
    for j, length in enumerate(lengths):
      assert length < len(x)
      streaks = 0
      k = 0
      while k <= len(exceeds_threshold) - length:
        if all(exceeds_threshold[k : k + length]):
          streaks += 1
          k += length
        else:
          k += 1
      count[i, j] = streaks
  return count


def count_heatwaves(
    key: xbeam.Key,
    ds: xr.Dataset,
    *,
    thresholds: list[float],
    lengths: list[int],
):
  """Counts heatwaves on a single pixel chunk."""
  assert len(ds.member) == len(ds.longitude) == len(ds.latitude) == 1
  years = np.unique(ds.time.dt.year)

  count = np.zeros((len(thresholds), len(lengths)))
  for year in years:
    ds_ = ds["HI"].sel(time=ds.time.dt.year.isin([year]), drop=True)
    # Extract daily max series.
    ds_max = ds_.resample(time="D").max()
    ds_max = ds_max.where(ds_max.time.isin(ds.time.values), drop=True)
    ts_max = ds_max.to_numpy()
    assert ts_max.ndim == 4  # (member, time, lon, lat)
    ts_max = np.squeeze(ts_max)
    assert ts_max.ndim == 1
    # Count heatwaves.
    count += _count_heatwaves(ts_max, thresholds, lengths)

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
  count_ds = xr.Dataset({"heatwave_counts": count_da})

  new_key = key.with_offsets(time=None, thresholds=0, lengths=0)
  return new_key, count_ds


def add_derived_variables(ds: xr.Dataset) -> xr.Dataset:
  """Adds temperature (F), relative humidity and heat index to dataset."""
  zs_reference = "/weatherbench/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
  ds = eval_utils.add_zs(zs_reference)(ds)

  ds = eval_utils.apply_ufunc(
      ds, eval_utils.T_fahrenheit, input_vars=["2mT"], output_var="2mTF"
  )
  ds = eval_utils.apply_ufunc(
      ds,
      eval_utils.relative_humidity,
      input_vars=["2mT", "Q1000", "MSL", "ZS"],
      output_var="RH",
  )
  ds = eval_utils.apply_ufunc(
      ds, eval_utils.noaa_heat_index, input_vars=["2mTF", "RH"], output_var="HI"
  )
  return ds


def main(argv):

  years = list(range(YEAR_START.value, YEAR_END.value + 1))
  months = [int(m) for m in MONTHS.value]
  thresholds = [float(t) for t in THRESHOLDS.value]
  lengths = [int(l) for l in LENGTHS.value]

  sample_ds = xr.open_zarr(SAMPLES.value, consolidated=True)
  sample_ds = add_derived_variables(sample_ds)
  sample_ds = eval_utils.select_time(sample_ds, years, months)

  # If a reference path is provided, the script will compute statistics on
  # the reference dataset instead of the inference dataset.
  if REFERENCE.value is not None:
    sample_ds = eval_utils.get_reference_ds(
        REFERENCE.value, SAMPLE_VARIABLES, sample_ds
    )
    sample_ds = add_derived_variables(sample_ds)
    sample_ds = eval_utils.select_time(sample_ds, years, months)

  in_chunks = {"time": -1, "member": -1, "longitude": 1, "latitude": 1}
  working_chunks_in = {"time": -1, "member": 1, "longitude": 1, "latitude": 1}
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
          "heatwave_counts": (
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
        | xbeam.DatasetToChunks(sample_ds, in_chunks, num_threads=16)
        | xbeam.SplitChunks(working_chunks_in)
        | beam.MapTuple(
            functools.partial(
                count_heatwaves,
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
            num_threads=16,
        )
    )


if __name__ == "__main__":
  app.run(main)
