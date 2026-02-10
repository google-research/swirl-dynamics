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

r"""Script to perform downscaling with the STAR-ESDM method.

This script implements the STAR-ESDM method for bias correction and downscaling,
following Hayhoe et al (2024): https://doi.org/10.1029/2023EF004107. This method
is supposed to relax the stationarity assumptions of other empirical statistical
downscaling methods. The methodology is applied to a single domain
discretization (the high-resolution grid), similarly to the STAR-ESDM paper.

The method assumes we have the following information from the low-resolution
dataset:
- A low-resolution dataset to be downscaled,
- a third-order parametric fit of its long-term trend (per pixel),
- its detrended climatology over a "training" period for which we have
  high-resolution data. That is, the climatology of the low-resolution data
  after the parametric trend has been removed (see
  `compute_detrended_climatology.py`),
- its detrended "dynamic" climatology, which denotes the current climatology for
  the dates to be downscaled (e.g., the 2090-2100 detrended climatology for
  downscaling years in the 2090s), and
- its temporal mean over the "training" period for which we have high-resolution
  data.

Regarding the target high-resolution dataset used to downscale the
low-resolution dataset, we assume we have:
- its detrended climatology over the "training" period,
- its temporal mean over the "training" period, used for debiasing.

All these terms are further described in Hayhoe et al (2024). In this
implementation, we assume climatologies have hourly granularity.

The final result consists of the sum of three terms, as sketched in Fig. 1 of
Hayhoe et al (2024):

1. The debiased long-term trend of the low-resolution data. This is the
   long-term trend of the low-resolution data, plus the difference between the
   mean of the high-resolution data and the mean of the low-resolution data over
   the training period, for each location and variable.

2. The "dynamically-adjusted" detrended climatological mean of the
   high-resolution data. This is constructed as the climatological mean of the
   high-resolution data over the training period, plus the difference between
   the dynamic climatology of the low-resolution data and the climatology
   of the low-resolution data over the training period, for each location and
   variable.

3. The quantile-mapped anomaly of the low-resolution data. The low-resolution
   sample is detrended, and its probability is computed according to the CDF of
   the detrended dynamic low-resolution climatology. Then, the anomalies
   corresponding to this probability according to the detrended high-resolution
   climatology, and the detrended low-resolution climatology are computed. The
   ratio of these two anomalies is retained. The final anomaly is computed as
   the product of the anomaly with respect to the detrended dynamic
   low-resolution climatology, and the precomputed ratio. See equations 4 and 5
   of Hayhoe et al (2024) for more details.

The method relies on quasi-Gaussian assumptions, and is therefore only
applicable to variables with a quasi-Gaussian distribution. Modifications are
required for variables such as precipitation.

Example usage:

```
FORCING_DATASET=<parent_dir>/canesm5_r1i1p2f1_ssp370_bc

INPUT_TREND=<input_trend>
INPUT_TEMPORAL_MEAN=<input_mean>
INPUT_DETRENDED_CLIM=<input_clim>
INPUT_DETRENDED_DYNAMIC_CLIM=<input_dyn_clim>

TARGET_TEMPORAL_MEAN=<target_mean>
TARGET_DETRENDED_CLIM=<target_clim>


INPUT_DATA=${FORCING_DATASET}/hourly_d01_cubic_interpolated_to_d02_with_prates.zarr
TIME_START=2094
TIME_STOP=2097
OUTPUT_PATH=${FORCING_DATASET}/baselines/staresdm_with_prates_from_canesm5_${TIME_START}_${TIME_STOP}.zarr

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/staresdm.py \
  --input_data=${INPUT_DATA} \
  --input_detrended_clim=${INPUT_DETRENDED_CLIM} \
  --input_detrended_dynamic_clim=${INPUT_DETRENDED_DYNAMIC_CLIM} \
  --input_temporal_mean=${INPUT_TEMPORAL_MEAN} \
  --input_trend=${INPUT_TREND} \
  --target_detrended_clim=${TARGET_DETRENDED_CLIM} \
  --target_temporal_mean=${TARGET_TEMPORAL_MEAN} \
  --time_start=${TIME_START} \
  --time_stop=${TIME_STOP} \
  --output_path=${OUTPUT_PATH}
```

"""

import typing

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import beam_utils
import xarray as xr
import xarray_beam as xbeam


INPUT_DATA = flags.DEFINE_string(
    'input_data',
    None,
    help='Zarr path pointing to the input data to be processed.',
)
INPUT_TREND = flags.DEFINE_string(
    'input_trend',
    None,
    help=(
        'Zarr path pointing to the input trend, stored in terms of np.polyfit'
        ' coefficients.'
    ),
)
INPUT_TEMPORAL_MEAN = flags.DEFINE_string(
    'input_temporal_mean',
    None,
    help='Zarr path pointing to the input temporal mean.',
)
INPUT_DETRENDED_CLIM = flags.DEFINE_string(
    'input_detrended_clim',
    None,
    help='Zarr path pointing to the input detrended climatology.',
)
INPUT_DETRENDED_DYNAMIC_CLIM = flags.DEFINE_string(
    'input_detrended_dynamic_clim',
    None,
    help='Zarr path pointing to the input detrended dynamic climatology.',
)
TARGET_DETRENDED_CLIM = flags.DEFINE_string(
    'target_detrended_clim',
    None,
    help='Zarr path pointing to the target detrended climatology.',
)
TARGET_TEMPORAL_MEAN = flags.DEFINE_string(
    'target_temporal_mean',
    None,
    help='Zarr path pointing to the target temporal mean.',
)
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
TIME_DIM = flags.DEFINE_string(
    'time_dim', 'time', help='Name for the time dimension to slice data on.'
)
TIME_START = flags.DEFINE_string(
    'time_start',
    None,
    help='ISO 8601 timestamp (inclusive) at which to start evaluation',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    None,
    help='ISO 8601 timestamp (inclusive) at which to stop evaluation',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _staresdm_on_chunks(
    source: xr.Dataset,
    *,
    input_clim: xr.Dataset,
    input_dynamic_clim: xr.Dataset,
    input_trend: xr.Dataset,
    input_temporal_mean: xr.Dataset,
    target_clim: xr.Dataset,
    target_temporal_mean: xr.Dataset,
) -> xr.Dataset:
  """Process an input data chunk with the STAR-ESDM downscaling method.

  All input climatologies are assumed to have hourly granularity.

  Args:
    source: The source data chunk to be processed with the STAR-ESDM method.
    input_clim: The detrended climatology of the source low-resolution data.
    input_dynamic_clim: The dynamic detrended climatology of the source
      low-resolution data.
    input_trend: The trend of the source low-resolution data, stored in terms of
      np.polyfit coefficients. For each variable, the trend coefficients are
      stored in variables with name <variable>_polyfit_coefficients.
    input_temporal_mean: The temporal mean of the source low-resolution data.
    target_clim: The detrended climatology of the target high-resolution data.
    target_temporal_mean: The temporal mean of the target high-resolution data.

  Returns:
    The downscaled chunks using the STAR-ESDM method as an xarray Dataset.
  """
  variables = [str(key) for key in source.keys()]

  # Component 1: Debiased long-term trend.
  staresdm = typing.cast(
      xr.Dataset,
      target_temporal_mean[variables] - input_temporal_mean[variables],
  )

  # Detrend the source data and add trend value to staresdm.
  for source_var in variables:
    coeff = input_trend[str(source_var) + '_polyfit_coefficients']
    source_trend = xr.polyval(source['time'], coeff)
    source[str(source_var)] = source[str(source_var)] - source_trend
    staresdm[str(source_var)] = staresdm[str(source_var)] + source_trend

  staresdm = staresdm.transpose('time', ...)

  # Component 2: Dynamically-adjusted high-resolution climatological mean.
  sel = dict(
      dayofyear=source['time'].dt.dayofyear,
      hour=source['time'].dt.hour,
      drop=True,
  )

  # Static input low-resolution climatology.
  input_clim_mean = beam_utils.get_climatology_mean(
      input_clim, variables, **sel
  )
  input_clim_std = beam_utils.get_climatology_std(input_clim, variables, **sel)
  # Dynamic input low-resolution climatology.
  input_dynamic_clim_mean = beam_utils.get_climatology_mean(
      input_dynamic_clim, variables, **sel
  )
  input_dynamic_clim_std = beam_utils.get_climatology_std(
      input_dynamic_clim, variables, **sel
  )
  # Target high-resolution climatology.
  target_clim_mean = beam_utils.get_climatology_mean(
      target_clim, variables, **sel
  )
  target_clim_std = beam_utils.get_climatology_std(
      target_clim, variables, **sel
  )

  staresdm = staresdm + target_clim_mean
  staresdm = staresdm + (input_dynamic_clim_mean - input_clim_mean)

  # Component 3: Quantile-mapped low-resolution anomaly.

  # Standardize with respect to the input dynamic climatology.
  source_standard = (source - input_dynamic_clim_mean) / input_dynamic_clim_std

  # Construct proxy of high-resolution dynamic anomaly.
  source_hr_anom = source_standard * target_clim_std
  source_lr_anom = source_standard * input_clim_std
  source_lr_dyn_anom = source_standard * input_dynamic_clim_std
  source_hr_dyn_anom = source_hr_anom * source_lr_dyn_anom / source_lr_anom

  # Add debiased anomaly to the computation
  staresdm = staresdm + source_hr_dyn_anom

  # TODO: Add wet days correction to precipitation variables.
  # Drop hour and dayofyear dimensions.
  return staresdm.drop_vars(['hour', 'dayofyear'])


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  if TIME_START.value is None or TIME_STOP.value is None:
    return ds
  selection = {TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)}
  return ds.sel({k: v for k, v in selection.items() if k in ds.dims})


def main(argv: list[str]) -> None:

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_DATA.value)
  source_dataset = _impose_data_selection(source_dataset)

  input_clim = xr.open_zarr(INPUT_DETRENDED_CLIM.value)
  input_dynamic_clim = xr.open_zarr(INPUT_DETRENDED_DYNAMIC_CLIM.value)
  input_trend = xr.open_zarr(INPUT_TREND.value)
  input_temporal_mean = xr.open_zarr(INPUT_TEMPORAL_MEAN.value)
  target_clim = xr.open_zarr(TARGET_DETRENDED_CLIM.value)
  target_temporal_mean = xr.open_zarr(TARGET_TEMPORAL_MEAN.value)

  source_chunks = {k: source_chunks[k] for k in source_chunks}
  in_working_chunks = source_chunks.copy()
  in_working_chunks['time'] = 1

  output_chunks = source_chunks.copy()
  output_chunks['time'] = 1
  unassigned_coords = {
      dim: np.arange(source_dataset.sizes[dim])
      for dim in source_dataset.dims
      if dim not in source_dataset.coords
  }
  template = xbeam.make_template(source_dataset).assign_coords(
      **unassigned_coords
  )

  # Static kwargs for _staresdm_on_chunks.
  staresdm_kwargs = dict(
      input_clim=input_clim,
      input_dynamic_clim=input_dynamic_clim,
      input_trend=input_trend,
      input_temporal_mean=input_temporal_mean,
      target_clim=target_clim,
      target_temporal_mean=target_temporal_mean,
  )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(source_dataset, in_working_chunks)
        | 'STAR-ESDM'
        >> beam.MapTuple(
            lambda k, v: (
                k,
                _staresdm_on_chunks(v, **staresdm_kwargs),
            )
        )
        | xbeam.ConsolidateChunks(output_chunks)
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            output_chunks,
        )
    )


if __name__ == '__main__':
  app.run(main)
