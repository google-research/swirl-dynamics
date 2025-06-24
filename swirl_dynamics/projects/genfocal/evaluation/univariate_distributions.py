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

r"""Compute univariate distribution error for all (lon, lat) coordinates.

Example usage:

```
SAMPLE_PATH=<sample_zarr_path>
REFERENCE_PATH=<reference_zarr_path>
OUTPUT_PATH=<output_zarr_path>
ZS_PATH=<surface_geopotential_zarr_path>

python swirl_dynamics/projects/genfocal/evaluation/univariate_distributions.py -- \
    --samples=${SAMPLE_PATH} \
    --reference=${REFERENCE_PATH} \
    --output_path=${OUTPUT_PATH} \
    --zs_path=${ZS_PATH} \
    --err_type=bias \
    --year_start=2010 \
    --year_end=2019 \
    --months=6,7,8
```

"""

from typing import TypeVar

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from scipy import stats
from swirl_dynamics.projects.genfocal.evaluation import utils
from swirl_dynamics.projects.genfocal.super_resolution import data
import xarray as xr
import xarray_beam as xbeam

Dataset = TypeVar("Dataset", xr.Dataset, xr.DataArray)
Variable = data.DatasetVariable

# Command line arguments
SAMPLE_PATH = flags.DEFINE_string(
    "sample_path", None, help="Sample dataset Zarr path."
)
REFERENCE = flags.DEFINE_string(
    "reference", None, help="Reference dataset Zarr path."
)
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path.")
ZS_PATH = flags.DEFINE_string(
    "zs_path", None, help="Path to the surface geopotential data."
)
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
    "months", ["6", "7", "8"], help="Months for evaluation."
)
NORMALIZE_ERROR = flags.DEFINE_bool(
    "normalize_error",
    False,
    help="Whether to normalize error by the std of the reference.",
)

SAMPLE_VARIABLES = [
    Variable("2m_temperature", None, "T2m"),
    Variable("10m_magnitude_of_wind", None, "W10m"),
    Variable("specific_humidity", {"level": [1000]}, "Q1000"),
    Variable("mean_sea_level_pressure", None, "MSL"),
]
PERCENTILES = [1, 5, 10, 90, 95, 99]


def compute_err(
    key: xbeam.Key,
    datasets: dict[str, list[Dataset]],
    err_type: str = "bias",
    normalize_error: bool = False,
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
      res[var] = ref_ds[var].mean(dim=mean_dims) - sample_ds[var].mean(
          dim=mean_dims
      )
      if normalize_error:
        res[var] = res[var] / ref_ds[var].std(dim=mean_dims)
    elif err_type == "wass":
      ref_data = ref_ds[var].to_numpy().ravel()
      sample_data = sample_ds[var].to_numpy().ravel()
      wass = stats.wasserstein_distance(ref_data, sample_data)
      if normalize_error:
        wass = wass / np.std(ref_data)
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
      ref_quantiles = np.percentile(ref_data, PERCENTILES)
      sample_data = sample_ds[var].to_numpy().ravel()
      sample_quantiles = np.percentile(sample_data, PERCENTILES)
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


def add_derived_variables(ds: xr.Dataset, zs_data_path: str) -> xr.Dataset:
  """Adds temperature (F), relative humidity and heat index to dataset.

  Args:
    ds: The dataset to add derived variables to.
    zs_data_path: The path to the surface geopotential data. A Zarr dataset with
      a `geopotential_at_surface` variable and safe to query for the longitude
      and latitude coordinates of `ds`.

  Returns:
    The dataset with derived variables added.
  """
  ds = utils.add_zs(zs_data_path)(ds)

  ds = utils.apply_ufunc(
      ds, utils.T_fahrenheit, input_vars=["T2m"], output_var="T2m_F"
  )
  ds = utils.apply_ufunc(
      ds,
      utils.relative_humidity,
      input_vars=["T2m", "Q1000", "MSL", "ZS"],
      output_var="RH",
  )
  ds = utils.apply_ufunc(
      ds, utils.heat_index, input_vars=["T2m_F", "RH"], output_var="HI"
  )
  return ds


def main(argv):

  years = list(range(YEAR_START.value, YEAR_END.value + 1))
  months = [int(m) for m in MONTHS.value]

  sample_ds = xr.open_zarr(SAMPLE_PATH.value, consolidated=True)
  sample_ds = add_derived_variables(sample_ds, ZS_PATH.value)
  sample_ds = utils.select_time(sample_ds, years, months)

  ref_ds = utils.get_reference_ds(REFERENCE.value, SAMPLE_VARIABLES, sample_ds)
  ref_ds = add_derived_variables(ref_ds, ZS_PATH.value)
  ref_ds = utils.select_time(ref_ds, years, months)

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
        | beam.MapTuple(
            compute_err,
            err_type=ERR_TYPE.value,
            normalize_error=NORMALIZE_ERROR.value,
        )
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
