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

"""Compute distribution error for every pixel, specifically for winter season."""

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from scipy import stats
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
ERR_TYPE = flags.DEFINE_string(
    "err_type", "bias", help="Error type. Either 'bias' or 'wass'."
)
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")
YEAR_START = flags.DEFINE_integer(
    "year_start", 2001, help="Starting year for evaluation."
)
YEAR_END = flags.DEFINE_integer(
    "year_end", 2010, help="Ending year for evaluation."
)
MONTHS = flags.DEFINE_list(
    "months", ["12", "1", "2"], help="Months for evaluation."
)

SAMPLE_VARIABLES = [
    Variable("2m_temperature", None, "2mT"),
    Variable("10m_magnitude_of_wind", None, "10mW"),
    Variable("2m_specific_humidity", None, "2mQ"),
    Variable("mean_sea_level_pressure", None, "MSL"),
]
PERCENTILES = [1, 5, 10, 90, 95, 99]


def _remove_nans(arr: np.ndarray) -> np.ndarray:
  return arr[~np.isnan(arr)]


def compute_err(
    key: xbeam.Key,
    datasets: dict[str, list[xr.Dataset]],
    err_type: str = "bias",
) -> tuple[xbeam.Key, xr.Dataset]:
  """Compute error between sample and reference datasets."""
  sample, ref = datasets["sample"], datasets["ref"]
  assert len(sample) == len(ref) == 1
  sample_ds, ref_ds = sample[0], ref[0]
  res = {}
  for var in sample_ds:
    assert len(ref_ds.longitude) == len(ref_ds.latitude) == 1
    assert len(sample_ds.longitude) == len(sample_ds.latitude) == 1
    if err_type == "bias":
      mean_dims = [
          c for c in ref_ds[var].dims if c not in ["longitude", "latitude"]
      ]
      mean_ref = ref_ds[var].mean(dim=mean_dims, skipna=True)
      mean_sample = sample_ds[var].mean(dim=mean_dims, skipna=True)
      res[var] = mean_ref - mean_sample
    elif err_type == "wass":
      ref_data = _remove_nans(ref_ds[var].to_numpy().ravel())
      sample_data = _remove_nans(sample_ds[var].to_numpy().ravel())
      wass = stats.wasserstein_distance(ref_data, sample_data)
      res[var] = xr.DataArray(
          data=np.asarray(wass).reshape(
              len(ref_ds.longitude), len(ref_ds.latitude)
          ),
          dims=("longitude", "latitude"),
          coords={
              "longitude": ref_ds.longitude,
              "latitude": ref_ds.latitude,
          },
      )
    elif err_type == "percentile":
      ref_data = ref_ds[var].to_numpy().ravel()
      ref_quantiles = np.nanpercentile(ref_data, PERCENTILES)
      sample_data = sample_ds[var].to_numpy().ravel()
      sample_quantiles = np.nanpercentile(sample_data, PERCENTILES)
      percentile_err = ref_quantiles - sample_quantiles
      res[var] = xr.DataArray(
          data=np.asarray(percentile_err).reshape(
              len(ref_ds.longitude), len(ref_ds.latitude), len(PERCENTILES)
          ),
          dims=("longitude", "latitude", "percentile"),
          coords={
              "longitude": ref_ds.longitude,
              "latitude": ref_ds.latitude,
              "percentile": np.asarray(PERCENTILES),
          },
      )
      key = key.with_offsets(percentile=0)
    else:
      raise ValueError(f"Unknown err_type: {err_type}")

  key = key.with_offsets(time=None, member=None)
  return key, xr.Dataset(res)


def add_windchill(ds: xr.Dataset) -> xr.Dataset:
  """Adds windchill variable to dataset."""
  ds = eval_utils.apply_ufunc(
      ds,
      eval_utils.wind_chill_temperature,
      input_vars=["2mT", "10mW"],
      output_var="WCT",
  )
  return ds


def main(argv):

  years = list(range(YEAR_START.value, YEAR_END.value + 1))
  months = [int(m) for m in MONTHS.value]

  sample_ds = xr.open_zarr(SAMPLES.value, consolidated=True)
  for var in SAMPLE_VARIABLES:
    if var.rename not in sample_ds:
      raise ValueError(
          f"Variable {var.rename} expected but not found in sample dataset."
      )
  sample_ds = sample_ds[[v.rename for v in SAMPLE_VARIABLES]]
  sample_ds = add_windchill(sample_ds)
  sample_ds = eval_utils.select_time(sample_ds, years, months)

  ref_ds = eval_utils.get_reference_ds(
      REFERENCE.value, SAMPLE_VARIABLES, sample_ds
  )
  ref_ds = add_windchill(ref_ds)
  ref_ds = eval_utils.select_time(ref_ds, years, months)

  work_chunks = {"time": -1, "member": -1, "longitude": 1, "latitude": 1}
  target_chunks = {"longitude": -1, "latitude": -1}

  # Create template.
  err_template = sample_ds.mean(dim=("time", "member"))
  if ERR_TYPE.value == "percentile":
    err_template = err_template.expand_dims(
        percentile=np.asarray(PERCENTILES), axis=-1
    )
    target_chunks["percentile"] = -1
  template = xbeam.make_template(err_template)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    sample_pcolls = root | "Sample to chunks" >> xbeam.DatasetToChunks(
        sample_ds, work_chunks, num_threads=16
    )

    ref_pcolls = root | "Reference to chunks" >> xbeam.DatasetToChunks(
        ref_ds, work_chunks, num_threads=16
    )

    _ = (
        {"sample": sample_pcolls, "ref": ref_pcolls}
        | beam.CoGroupByKey()
        | beam.MapTuple(compute_err, err_type=ERR_TYPE.value)
        | xbeam.ConsolidateChunks(target_chunks=target_chunks)
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            target_chunks,
            num_threads=16,
        )
    )


if __name__ == "__main__":
  app.run(main)
