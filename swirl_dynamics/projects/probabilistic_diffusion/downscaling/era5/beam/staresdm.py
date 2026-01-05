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

r"""Script to perform downscaling with the STAR-ESDM method.

This script implements the STAR-ESDM method for bias correction and downscaling,
following Hayhoe et al (2024): https://doi.org/10.1029/2023EF004107. This method
is supposed to relax the stationarity assumptions of other empirical statistical
downscaling methods. The methodology is applied to a low-resolution dataset
whose surface coordinates can be interpolated to the high-resolution grid. That
is, we assume that the low-resolution domain contains the high-resolution
domain, at a lower resolution.

The method assumes we have the following information from the low-resolution
inputs:
- A low-resolution dataset of daily frequency to be downscaled,
- a third-order parametric fit of its long-term trend (per pixel),
- its detrended daily climatology over a "training" period for which we have
  high-resolution data. That is, the climatology of the low-resolution data
  after the parametric trend has been removed (see
  `compute_detrended_climatology.py`),
- its detrended "dynamic" daily climatology, which denotes the current
  climatology for the dates to be downscaled (e.g., the 2090-2100 detrended
  climatology for downscaling years in the 2090s), and
- its temporal mean over the "training" period for which we have high-resolution
  data.

Note that the detrended dynamic climatology must cover roughly 10 years and
be centered around the test period. For instance, if we are interested in
downscaling the 2050s in addition to the 2090s, we need to have the detrended
dynamic climatology over the 2050-2060 period.

Regarding the target high-resolution dataset used to downscale the
low-resolution dataset, we assume we have:
- its detrended daily climatology over the "training" period,
- its temporal mean over the "training" period, used for debiasing.

All these terms are further described in Hayhoe et al (2024). In this
implementation, we assume climatologies have daily granularity.

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

The original STAR-ESDM method does not perform temporal disaggregation. This
script implements an additional temporal disaggregation step from daily to
hourly resolution following the
BCSD method (see https://doi.org/10.1029/2001JD000659).

"""

import functools
import typing

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import beam_utils
import xarray as xr
import xarray_beam as xbeam


_LENS_TO_ERA = {
    'TREFHT': '2m_temperature',
    'WSPDSRFAV': '10m_magnitude_of_wind',
    'QREFHT': '2m_specific_humidity',
    'PSL': 'mean_sea_level_pressure',
}

_OUTPUT_RENAME = {
    '2m_temperature': '2mT',
    '10m_magnitude_of_wind': '10mW',
    'specific_humidity': '2mQ',
    'mean_sea_level_pressure': 'MSL',
}

INPUT_DATA = flags.DEFINE_string(
    'input_data',
    None,
    help='Zarr path pointing to the input data to be processed.',
)
INPUT_TREND = flags.DEFINE_string(
    'input_trend',
    None,
    help=(
        'Zarr path pointing to the long-term trend of the input data, stored as'
        ' np.polyfit coefficients.'
    ),
)
INPUT_TEMPORAL_MEAN = flags.DEFINE_string(
    'input_temporal_mean',
    None,
    help=(
        'Zarr path pointing to the input temporal mean over the training'
        ' period.'
    ),
)
INPUT_DETRENDED_CLIM = flags.DEFINE_string(
    'input_detrended_clim',
    None,
    help=(
        'Zarr path pointing to the input detrended climatology over the'
        ' training period.'
    ),
)
INPUT_DETRENDED_DYNAMIC_CLIM = flags.DEFINE_string(
    'input_detrended_dynamic_clim',
    None,
    help=(
        'Zarr path pointing to the input detrended dynamic climatology,'
        ' which is the detrended climatology over the test period.'
    ),
)
TARGET_DETRENDED_CLIM = flags.DEFINE_string(
    'target_detrended_clim',
    None,
    help=(
        'Zarr path pointing to the target detrended climatology over the'
        ' training period.'
    ),
)
TARGET_TEMPORAL_MEAN = flags.DEFINE_string(
    'target_temporal_mean',
    None,
    help=(
        'Zarr path pointing to the target temporal mean over the training'
        ' period.'
    ),
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
SELECT_MONTHS = flags.DEFINE_list(
    'select_months',
    None,
    help='List of months to select from the input data.',
)
HOUR_RESOLUTION = flags.DEFINE_integer(
    'hour_resolution',
    1,
    help='Time resolution (in hours) for the historical reference data.',
)
SAMPLES_PER_MEMBER = flags.DEFINE_integer(
    'samples_per_member',
    8,
    help='Number of samples to generate per member.',
)
HIST_REF = flags.DEFINE_string(
    'hist_ref',
    None,
    help=(
        'Zarr path pointing to the historical reference data (assumed in hourly'
        ' resolution) used for temporal disaggregation.'
    ),
)
REF_START_YEAR = flags.DEFINE_string(
    'ref_start', '1961', help='Starting year for the historical reference data.'
)
REF_STOP_YEAR = flags.DEFINE_string(
    'ref_stop', '2000', help='End year for the historical reference data.'
)
PIXEL_CHUNKS = flags.DEFINE_boolean(
    'pixel_chunks',
    False,
    help=(
        'Whether to have pixel chunks (chunk size = 1 for longitude and'
        ' latitude; -1 for member). If `False` will make member chunks instead'
        ' (chunk size = 1 for member, -1 for longitude and latitude).'
    ),
)
NUM_DAYS_PER_CHUNK = flags.DEFINE_integer(
    'num_days_per_chunk',
    30,
    help='Number of days to include in each chunk.',
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

  All input climatologies are assumed to have daily granularity. In terms of
  spatial resolution, all `input_` datasets are assumed to be co-located with
  the source data chunk, while the `target_` datasets can be in a different
  grid. Results are interpolated to the target grid.

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

  input_temporal_mean_interp = input_temporal_mean.interp(
      longitude=target_clim.coords['longitude'],
      latitude=target_clim.coords['latitude'],
      method='cubic',
  )

  # Component 1: Debiased long-term trend.
  staresdm = typing.cast(
      xr.Dataset,
      target_temporal_mean[variables] - input_temporal_mean_interp[variables],
  )

  # Detrend the source data and add trend value to staresdm.
  for source_var in variables:
    coeff = input_trend[str(source_var) + '_polyfit_coefficients']
    source_trend = xr.polyval(source['time'], coeff)
    source[str(source_var)] = source[str(source_var)] - source_trend
    source_trend_interp = source_trend.interp(
        longitude=target_clim.coords['longitude'],
        latitude=target_clim.coords['latitude'],
        method='cubic',
    )
    staresdm[str(source_var)] = staresdm[str(source_var)] + source_trend_interp

  staresdm = staresdm.transpose('time', ...)

  # Component 2: Dynamically-adjusted high-resolution climatological mean.
  sel = dict(dayofyear=source['time'].dt.dayofyear, drop=True)

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

  # Add high-resolution climatology mean to staresdm.
  staresdm = staresdm + target_clim_mean
  # Add dynamic climatological adjustment
  staresdm = staresdm + (input_dynamic_clim_mean - input_clim_mean).interp(
      longitude=target_clim.coords['longitude'],
      latitude=target_clim.coords['latitude'],
      method='cubic',
  )

  # Component 3: Quantile-mapped low-resolution anomaly.

  # Standardize with respect to the input dynamic climatology.
  source_standard = (source - input_dynamic_clim_mean) / input_dynamic_clim_std

  # Construct proxy of high-resolution dynamic anomaly.
  source_hr_anom = (
      source_standard.interp(
          longitude=target_clim.coords['longitude'],
          latitude=target_clim.coords['latitude'],
          method='cubic',
      )
      * target_clim_std
  )
  source_lr_anom = source_standard * input_clim_std
  source_lr_dyn_anom = source_standard * input_dynamic_clim_std
  source_hr_dyn_anom = source_hr_anom * (
      source_lr_dyn_anom / source_lr_anom
  ).interp(
      longitude=target_clim.coords['longitude'],
      latitude=target_clim.coords['latitude'],
      method='cubic',
  )

  # Add debiased anomaly to the computation
  staresdm = staresdm + source_hr_dyn_anom

  # TODO: Add wet days correction to precipitation variables.
  # Drop dayofyear dimensions.
  staresdm = staresdm.drop_vars('dayofyear').transpose(
      'member', 'time', 'longitude', 'latitude'
  )
  return staresdm.compute()


# TODO: In the following we do a simple mean shift but multiplicative
# variables should be handled differently.
def _td_on_chunks(
    source_key: xbeam.Key,
    source: xr.Dataset,
    *,
    hist_ref: xr.Dataset,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Performs temporal disaggregation on a data chunk."""
  # Select the same dates in the historical references (as the source chunk).
  mask = np.logical_and(
      hist_ref.time.dt.month.isin(source.time.dt.month),
      hist_ref.time.dt.day.isin(source.time.dt.day),
  )
  hist_ref_chunk = hist_ref.sel(time=mask, drop=True)
  # Subtract the daily mean from the daily values in the historical references.
  hist_ref_chunk_daily_mean = hist_ref_chunk.groupby('time.date').mean('time')
  hist_ref_chunk_daily_anom = (
      hist_ref_chunk.groupby('time.date') - hist_ref_chunk_daily_mean
  )
  hist_ref_chunk_daily_anom = hist_ref_chunk_daily_anom.reset_coords(
      'date', drop=True
  )
  hist_ref_years = np.unique(hist_ref_chunk_daily_anom.time.dt.year)

  time = source.time.values
  assert len(time) == 1
  new_time = np.arange(
      time[0],
      time[0] + np.timedelta64(1, 'D'),
      np.timedelta64(HOUR_RESOLUTION.value, 'h'),
  )

  # Form a sample for every year in historical reference. The STAR-ESDM output
  # is added as the daily mean.
  out_samples = []
  for m in range(len(source.member.values)):
    source_member = source.isel(member=[m], drop=True)
    sample_years = np.random.choice(
        hist_ref_years, SAMPLES_PER_MEMBER.value, replace=True
    )
    for i, y in enumerate(sample_years):
      sample_mask = hist_ref_chunk_daily_anom.time.dt.year == y
      sample = hist_ref_chunk_daily_anom.sel(time=sample_mask, drop=True)
      sample = sample.assign_coords(time=new_time)
      sample = sample + source_member.reindex(time=new_time, method='ffill')
      # New member name is now '{lens_member}_{sample_idx}'
      new_members = np.char.add(sample.member.values, f'_{i}')
      sample = sample.assign_coords(member=new_members)
      sample = sample.transpose('member', 'time', 'longitude', 'latitude')
      out_samples.append(sample)

  source_td = xr.concat(out_samples, dim='member')
  source_td = source_td.rename(_OUTPUT_RENAME)
  source_td = source_td.assign_coords(time=new_time)
  new_key = source_key.with_offsets(
      time=int(source_key.offsets['time'] * 24 / HOUR_RESOLUTION.value)
  )
  return new_key, source_td.compute()


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  """Imposes time range and month selection on the input dataset."""
  ds_sel = xr.Dataset(ds.get(list(_LENS_TO_ERA.keys()))).rename(_LENS_TO_ERA)
  if TIME_START.value is None and TIME_STOP.value is None:
    return ds_sel

  selection = {'time': slice(TIME_START.value, TIME_STOP.value)}
  selected = ds_sel.sel(
      {k: v for k, v in selection.items() if k in ds.dims}, drop=True
  )
  if SELECT_MONTHS.value:
    months = [int(m) for m in SELECT_MONTHS.value]
    selected = selected.sel(time=selected.time.dt.month.isin(months), drop=True)
  return selected


def _select_rename(
    ds: xr.Dataset, sel_vars: list[str], rename: dict[str, str]
) -> xr.Dataset:
  """Selects and renames variables in a dataset."""
  return xr.Dataset(ds.get(sel_vars)).rename(rename)


def main(argv: list[str]) -> None:

  lens_vars = list(_LENS_TO_ERA.keys())
  lens_vars.extend([v + '_std' for v in _LENS_TO_ERA.keys()])

  era_vars = list(_LENS_TO_ERA.values())
  era_vars.extend([v + '_std' for v in _LENS_TO_ERA.values()])

  clim_dict = {k + '_std': v + '_std' for k, v in _LENS_TO_ERA.items()}
  clim_dict.update(_LENS_TO_ERA)

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_DATA.value)
  source_dataset = _impose_data_selection(source_dataset)

  input_clim = _select_rename(
      xr.open_zarr(INPUT_DETRENDED_CLIM.value), lens_vars, clim_dict
  )
  input_dynamic_clim = _select_rename(
      xr.open_zarr(INPUT_DETRENDED_DYNAMIC_CLIM.value), lens_vars, clim_dict
  )
  input_temporal_mean = _select_rename(
      xr.open_zarr(INPUT_TEMPORAL_MEAN.value), lens_vars, clim_dict
  )

  lens_trends = [v + '_polyfit_coefficients' for v in _LENS_TO_ERA.keys()]
  trend_dict = {
      k + '_polyfit_coefficients': v + '_polyfit_coefficients'
      for k, v in _LENS_TO_ERA.items()
  }
  input_trend = _select_rename(
      xr.open_zarr(INPUT_TREND.value), lens_trends, trend_dict
  )

  target_clim = xr.Dataset(
      xr.open_zarr(TARGET_DETRENDED_CLIM.value).get(era_vars)
  )
  target_temporal_mean = xr.Dataset(
      xr.open_zarr(TARGET_TEMPORAL_MEAN.value).get(era_vars)
  )

  hist_ref = xr.open_zarr(HIST_REF.value).get(list(_LENS_TO_ERA.values()))
  hist_ref = hist_ref.sel(
      time=slice(REF_START_YEAR.value, REF_STOP_YEAR.value), drop=True
  )
  hist_ref = hist_ref.isel(
      time=slice(None, None, HOUR_RESOLUTION.value), drop=True
  )

  source_chunks = {k: source_chunks[k] for k in source_chunks if k != 'level'}
  logging.info('source_chunks: %s', source_chunks)

  # Define working and output chunks.
  in_working_chunks = {'time': 1, 'member': -1, 'longitude': -1, 'latitude': -1}
  if PIXEL_CHUNKS.value:
    split_chunks = {
        'time': int(24 / HOUR_RESOLUTION.value),
        'member': -1,
        'longitude': 1,
        'latitude': 1,
    }
    output_chunks = {
        # `NUM_DAYS_PER_CHUNK` should be bigger.
        'time': int(24 / HOUR_RESOLUTION.value) * NUM_DAYS_PER_CHUNK.value,
        'member': -1,
        'longitude': 1,
        'latitude': 1,
    }
  else:
    split_chunks = {
        'time': int(24 / HOUR_RESOLUTION.value),
        'member': 1,
        'longitude': -1,
        'latitude': -1,
    }
    output_chunks = {
        # `NUM_DAYS_PER_CHUNK` should be smaller.
        'time': int(24 / HOUR_RESOLUTION.value) * NUM_DAYS_PER_CHUNK.value,
        'member': 1,
        'longitude': -1,
        'latitude': -1,
    }

  template = xbeam.make_template(source_dataset)
  template = template.interp(
      longitude=target_clim.coords['longitude'],
      latitude=target_clim.coords['latitude'],
      method='nearest',  # Just a template; method does not matter.
  )
  # Increase time resolution.
  offsets = np.arange(24, step=HOUR_RESOLUTION.value, dtype='timedelta64[h]')
  new_time = template.time.values[:, np.newaxis] + offsets
  new_time = new_time.flatten()
  template = template.reindex(time=new_time, method='ffill')

  template = template.rename(_OUTPUT_RENAME)
  # Update member coordinates.
  source_members = source_dataset.member.values
  new_members = [
      [f'{member}_{i}' for i in range(SAMPLES_PER_MEMBER.value)]
      for member in source_members
  ]
  new_members = np.concatenate(new_members)
  template = template.reindex(member=new_members, method='ffill')
  # Must change coord types. Otherwise they are in old format (short string and
  # daily time).
  template = template.assign_coords(member=new_members.astype('<U20'))
  template = template.assign_coords(
      time=template.time.values.astype('datetime64[ns]')  # ('datetime64[h]')
  )
  template = template.transpose('member', 'time', 'longitude', 'latitude')
  logging.info('template: %s', template)

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
        | 'TD'
        >> beam.MapTuple(functools.partial(_td_on_chunks, hist_ref=hist_ref))
        | xbeam.SplitChunks(target_chunks=split_chunks)
        | xbeam.ConsolidateChunks(target_chunks=output_chunks)
        | xbeam.ChunksToZarr(OUTPUT_PATH.value, template, output_chunks)
    )


if __name__ == '__main__':
  app.run(main)
