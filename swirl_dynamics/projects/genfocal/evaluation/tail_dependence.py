# Copyright 2026 The swirl_dynamics Authors.
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

r"""A beam pipeline to compute the upper tail dependence of two variables.

The upper tail dependence is defined following eq. 13 and 16 of Schmidt and
Stadtmuller (2006; https://doi.org/10.1111/j.1467-9469.2005.00483.x). The
upper tail dependence coefficient is defined in terms of a parameter k, which
defines the sample rank threshold above which the dependence is computed.
Given a sample size m, k corresponds to the quantile q = (1 - k / m) of the
marginal empirical distributions of the two variables. Therefore, k and q are
related as follows:

                        q * m = m - k -> k = m * (1 - q)

As shown by Schmidt and Stadtmuller, the coefficient can be estimated over a
range of values of k, for which it remains relatively constant (their Fig. 5).
This script computes the upper tail dependence for all k values within a
specified empirical quantile range [q1, q2]. The asymptotic limit for the
underlying tail dependence holds at the limit of m -> infinity, k -> infinity,
with k/m -> 0.

Given a dataset with variables of shape (sample, time, *spatial_dims), the
script computes the upper tail dependence over the flattened time and sample
dimensions. That is, samples and times are considered i.i.d. The output is a
dataset with shape (quantile, ...), where quantile is the quantile of the sample
rank threshold above which the dependence is computed.

The script also implements a `sign_change` option for the second variable, which
is useful for computing the tail dependence of extrema of different signs in the
chosen variables. For instance, one may want to compute the tail dependence of
high temperature and low specific humidity extremes.

Example usage:

```
INFERENCE_PATH=<inference_zarr_path>
OUTPUT_PATH=<output_zarr_path>

python swirl_dynamics/projects/genfocal/evaluation/tail_dependence.py \
  --inference_path=${INFERENCE_PATH} \
  --output_path=${OUTPUT_PATH} \
  --quantile_range=0.9,0.95 \
  --variables=T2,Q2 \
  --start_year=2010 \
  --end_year=2019 \
  --months=6,7,8 \
  --hour_of_day=18
```
"""

import functools

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.genfocal.evaluation import utils
from swirl_dynamics.projects.genfocal.super_resolution import data
import xarray as xr
import xarray_beam as xbeam

_DEFAULT_QUANTILE_RANGE = ["0.9", "0.99"]

# Command line arguments
QUANTILE_RANGE = flags.DEFINE_list(
    "quantile_range",
    _DEFAULT_QUANTILE_RANGE,
    help=(
        "Quantiles between which to compute tail dependence at all meaningful"
        " granularities."
    ),
)
QUANTILE_POINTS = flags.DEFINE_integer(
    "quantile_points",
    50,
    help="Number of quantile points (evenly spaced in the range).",
)
VARIABLES = flags.DEFINE_list(
    "variables",
    ["2mT", "Q1000"],
    help="Variables to compute tail dependence on.",
)
SIGN_CHANGE = flags.DEFINE_bool(
    "sign_change",
    False,
    help=(
        "Whether to change the sign of the second variable before computing"
        " tail dependence."
    ),
)
INFERENCE_PATH = flags.DEFINE_string(
    "inference_path", None, help="Input Zarr path."
)
# If a reference path is provided, the script will compute tail dependence on
# the reference dataset instead of the inference dataset.
REFERENCE_PATH = flags.DEFINE_string(
    "reference_path", None, help="Reference Zarr path."
)
OUTPUT_PATH = flags.DEFINE_string("output_path", None, help="Output Zarr path.")
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    "rechunk_itemsize", 4, help="Itemsize for rechunking."
)
YEAR_START = flags.DEFINE_integer(
    "year_start", 2001, help="Starting year for evaluation."
)
YEAR_END = flags.DEFINE_integer(
    "year_end", 2010, help="Ending year for evaluation."
)
MONTHS = flags.DEFINE_list("months", ["8"], help="Months for evaluation.")
HOUR_OF_DAY = flags.DEFINE_integer(
    "hour_of_day", None, help="Hour of day in UTC time."
)


Variable = data.DatasetVariable
SAMPLE_VARIABLES = [
    Variable("2m_temperature", None, "T2m"),
    Variable("10m_magnitude_of_wind", None, "W10m"),
    Variable("specific_humidity", {"level": [1000]}, "Q1000"),
    Variable("mean_sea_level_pressure", None, "MSL"),
]


def eval_upper_tail_dependence_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    variables: tuple[str, str],
    agg_dims: tuple[str, ...],
    quantiles: np.ndarray,
    sign_change: bool = False,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Evaluate upper tail dependence for two variables over a chunk of data.

  The upper tail dependence is defined following eq. 13 and 16 of Schmidt and
  Stadtmuller (2006; https://doi.org/10.1111/j.1467-9469.2005.00483.x).

  Args:
    obs_key: They key to the data chunk.
    obs_chunk: The data chunk.
    variables: The variables to compute tail dependence on.
    agg_dims: The dimensions to aggregate samples, rank, and compute the tail
      dependence over.
    quantiles: The quantile values (1 - k / m) to compute tail dependence at.
      Values at quantiles are averaged.
    sign_change: Whether to change the sign of the second variable before
      computing tail dependence.

  Returns:
    The key and chunk of the upper tail dependence dataset, which has variable
    name `variables[0] + "_" + variables[1] + "_tail_dependence"` with shape
    (quantile, *spatial_dims).
  """
  offsets = {agg_dim: None for agg_dim in agg_dims}
  retained_dims = tuple(set(obs_chunk.dims) - set(agg_dims))
  retained_coords = {dim: obs_chunk.coords[dim].values for dim in retained_dims}

  # Stacked dimensions along last axis.
  var1_chunk = obs_chunk[variables[0]].stack({"samples": agg_dims}).values
  var2_chunk = obs_chunk[variables[1]].stack({"samples": agg_dims}).values
  var2_chunk = -var2_chunk if sign_change else var2_chunk

  sample_size = var1_chunk.shape[-1]

  var1_quantiles = np.nanquantile(var1_chunk, quantiles, axis=-1)
  var2_quantiles = np.nanquantile(var2_chunk, quantiles, axis=-1)

  deps = []
  for var1_quantile, var2_quantile, q in zip(
      var1_quantiles, var2_quantiles, quantiles
  ):
    var1_upper = var1_chunk > var1_quantile[..., np.newaxis]
    var2_upper = var2_chunk > var2_quantile[..., np.newaxis]
    joint_upper = np.logical_and(var1_upper, var2_upper)
    deps.append(np.sum(joint_upper, -1) / (sample_size * q))

  deps = np.stack(deps, axis=0).mean(axis=0)

  tdc_name = f"{variables[0]}_{variables[1]}_tail_dependence"
  tdc_chunk = xr.Dataset(
      data_vars={tdc_name: (retained_dims, deps)}, coords=retained_coords
  )

  tdc_key = obs_key.with_offsets(**offsets)
  return tdc_key, tdc_chunk


def main(argv: list[str]) -> None:

  quantile_range = tuple([float(q) for q in QUANTILE_RANGE.value])
  if len(quantile_range) != 2:
    raise ValueError(
        f"Quantile range must have length 2, but got {quantile_range}."
    )
  variables = tuple(VARIABLES.value)
  sign_change = SIGN_CHANGE.value
  spatial_dims = ["longitude", "latitude"]

  inference_ds, input_chunks = xbeam.open_zarr(INFERENCE_PATH.value)

  # If a reference path is provided, the script will compute tail dependence on
  # the reference dataset instead of the inference dataset.
  if REFERENCE_PATH.value is not None:
    inference_ds = utils.get_reference_ds(
        REFERENCE_PATH.value, SAMPLE_VARIABLES, inference_ds
    )
    del input_chunks
    input_chunks = {"longitude": -1, "latitude": -1, "time": 48, "member": -1}

  retained_coords = {dim: inference_ds[dim].values for dim in spatial_dims}

  years = list(range(YEAR_START.value, YEAR_END.value + 1))
  months = [int(m) for m in MONTHS.value]
  inference_ds = utils.select_time(inference_ds, years, months)

  # Compute tail dependence on the data only at a specific hour of day to reduce
  # the effects of diurnal cycles.
  if HOUR_OF_DAY.value is not None:
    hour_mask = inference_ds.time.dt.hour.isin([HOUR_OF_DAY.value])
    inference_ds = inference_ds.sel(time=hour_mask, drop=True)

  # Chunks for temporal and sample reduce
  in_working_chunks = {
      k: -1 for k in input_chunks.keys() if k not in spatial_dims
  }
  agg_dims = tuple(in_working_chunks.keys())
  in_working_chunks.update({"longitude": 1, "latitude": 1})

  quantiles = np.linspace(
      quantile_range[0], quantile_range[1], QUANTILE_POINTS.value
  )

  # Chunks for tdc dataset: go from pixel chunks to one piece.
  out_working_sizes = {k: inference_ds.sizes[k] for k in spatial_dims}
  out_working_chunks = {k: 1 for k in spatial_dims}
  out_chunks = {k: -1 for k in spatial_dims}

  # Define template
  tdc_name = f"{variables[0]}_{variables[1]}_tail_dependence"
  tdc_template = np.empty(
      (
          inference_ds.sizes[spatial_dims[0]],
          inference_ds.sizes[spatial_dims[1]],
      ),
  )
  template_ds = xr.Dataset(
      data_vars={tdc_name: (spatial_dims, tdc_template)},
      coords=retained_coords,
  )
  template = xbeam.make_template(template_ds)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Read
    _ = (
        root
        | xbeam.DatasetToChunks(inference_ds, input_chunks, split_vars=False)
        | "RechunkIn"
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            inference_ds.sizes,
            input_chunks,
            in_working_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | "EvaluateTailDependence"
        >> beam.MapTuple(
            functools.partial(
                eval_upper_tail_dependence_chunk,
                agg_dims=agg_dims,
                quantiles=quantiles,
                variables=variables,
                sign_change=sign_change,
            )
        )
        | "RechunkOut"
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            out_working_sizes,
            out_working_chunks,
            out_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=template,
            zarr_chunks=out_chunks,
        )
    )


if __name__ == "__main__":
  app.run(main)
