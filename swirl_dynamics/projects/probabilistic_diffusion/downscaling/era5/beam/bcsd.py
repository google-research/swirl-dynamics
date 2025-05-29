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

r"""Flume script to perform downscaling with a variant of the BCSD method.

This script implements the Bias Correction and Spatial Disaggregation (BCSD)
method for downscaling, following Wood et al (2002):
https://doi.org/10.1029/2001JD000659.

The method assumes we have a low-resolution dataset, its climatology, and the
spatially filtered and unfiltered climatology of a high-resolution dataset. The
input low-resolution dataset and all low-resolution climatologies (i.e., the
input data climatology and the filtered target climatology) are assumed to
be defined in one (lat, lon) grid. The high-resolution climatology is
assumed to be defined in a different (lat, lon) grid that coincides or is
internal to the low-resolution grid. That is, if the low-resolution grid is
global, the high-resolution grid can either be global or a subregion of the
low-resolution grid.

In terms of temporal resolution, the input data and all climatologies are
assumed to be daily. The output of this BCSD script is at multi-hourly
resolution.

The bias-correction step is performed by computing the quantiles of the input
dataset with respect to its climatology, and mapping them to the full field
values corresponding to the same percentiles in the filtered high-resolution
climatology.

The spatial disaggregation consists (for most variables) of subtracting the
climatological mean of the filtered high-resolution field from the
bias-corrected low-resolution field, and adding the climatological mean of the
unfiltered high-resolution field. Since the low-resolution and high-resolution
climatologies are defined on different grids, the addition operations are
performed after interpolation.

For some variables (e.g., precipitation), we use a multiplicative correction
instead: we multiply the bias-corrected low-resolution field by the ratio of the
filtered and unfiltered climatological means. The variables to be corrected
multiplicatively are specified in the `multiplicative_vars` flag.

Let the low-resolution data correspond to time averages over periods `T`. In the
temporal disaggregation step, for each sample resulting from the spatial
disaggregation step, we sample a high-resolution data sequence over period `T`
and the same time of the year as the spatially disaggregated sample. The
high--resolution sample sequence is drawn at random from the dataset used to
construct the high-resolution climatology. Once the sample sequence is drawn,
we scale it such that the time average over `T` matches the spatially
disaggregated sample. Finally, we use this scaled sequence as the final BCSD
sample.

"""

import functools
import typing

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
import xarray as xr
import xarray_beam as xbeam


_LENS_TO_ERA = {
    'TREFHT': '2m_temperature',
    'WSPDSRFAV': '10m_magnitude_of_wind',
    'QREFHT': 'specific_humidity',
    'PSL': 'mean_sea_level_pressure',
}

_OUTPUT_RENAME = {
    '2m_temperature': '2mT',
    '10m_magnitude_of_wind': '10mW',
    'specific_humidity': 'Q1000',
    'mean_sea_level_pressure': 'MSL',
}

INPUT_DATA = flags.DEFINE_string(
    'input_data',
    None,
    help='Zarr path pointing to the input data to be processed.',
)
SELECT_MONTHS = flags.DEFINE_list(
    'select_months',
    None,
    help='List of months to select from the input data.',
)
INPUT_DATA_CLIM = flags.DEFINE_string(
    'input_data_clim',
    None,
    help='Zarr path pointing to the climatology of the input data.',
)
FILTERED_TARGET_CLIM = flags.DEFINE_string(
    'filtered_target_clim',
    None,
    help=(
        'Zarr path pointing to the climatology of the spatially filtered target'
        ' data.'
    ),
)
TARGET_CLIM = flags.DEFINE_string(
    'target_clim',
    None,
    help='Zarr path pointing to the climatology of the unfiltered target data.',
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
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
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
METHOD = flags.DEFINE_enum(
    'method',
    'gaussian',
    ['gaussian', 'quantile_interpolation'],
    'Method to use for quantile mapping.',
)
MULTIPLICATIVE_VARS = flags.DEFINE_list(
    'multiplicative_vars',
    [],
    'List of variables to process using multiplicative correction.',
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


def _get_climatology_mean(
    climatology: xr.Dataset, variables: list[str], **sel_kwargs
) -> xr.Dataset:
  """Returns the climatological mean of the given variables.

  The climatology dataset is assumed to have been produced through
  the weatherbench2 compute_climatology.py script, and statistics
  `mean`, and `std`. The convention is that the climatological means do not
  have a suffix, and standard deviations have a `_std` suffix.

  Args:
    climatology: The climatology dataset.
    variables: The variables to extract from the climatology.
    **sel_kwargs: Additional selection criteria for the variables.

  Returns:
    The climatological mean of the given variables.
  """
  # try:
  climatology_mean = climatology[variables]
  return typing.cast(xr.Dataset, climatology_mean.sel(**sel_kwargs).compute())


def _get_climatology_std(
    climatology: xr.Dataset, variables: list[str], **sel_kwargs
) -> xr.Dataset:
  """Returns the climatological standard deviation of the given variables.

  The climatology dataset is assumed to have been produced through
  the weatherbench2 compute_climatology.py script, and statistics
  `mean`, and `std`. The convention is that the climatological means do not
  have a suffix, and standard deviations have a `_std` suffix.

  Args:
    climatology: The climatology dataset.
    variables: The variables to extract from the climatology.
    **sel_kwargs: Additional selection criteria for the variables.

  Returns:
    The climatological standard deviation of the given variables.
  """
  clim_std_dict = {key + '_std': key for key in variables}  # pytype: disable=unsupported-operands
  # try:
  climatology_std = climatology[list(clim_std_dict.keys())].rename(
      clim_std_dict
  )
  return typing.cast(xr.Dataset, climatology_std.sel(**sel_kwargs).compute())


def _bcsd_on_chunks(
    source_key: xbeam.Key,
    source: xr.Dataset,
    *,
    original_clim: xr.Dataset,
    filtered_clim: xr.Dataset,
    target_clim: xr.Dataset,
    method: str = 'gaussian',
) -> tuple[xbeam.Key, xr.Dataset]:
  """Process an input data chunk with the BCSD method.

  Args:
    source_key: The key for the source data chunk.
    source: The source data chunk to be processed with the BCSD method. It is
      assumed to contain a single time slice.
    original_clim: The climatology of the source data.
    filtered_clim: The climatology of the spatially filtered target data,
      co-located with the source data.
    target_clim: The climatology of the unfiltered target data, potentially in
      different spatial locations than the source data.
    method: The method to use for quantile mapping.

  Returns:
    The source key and the BCSD data chunk.
  """
  sel = dict(
      dayofyear=source['time'].dt.dayofyear,
      drop=True,
  )
  variables = [str(key) for key in source.keys()]
  if method == 'gaussian':
    clim_mean = _get_climatology_mean(original_clim, variables, **sel)
    clim_std = _get_climatology_std(original_clim, variables, **sel)

    # Standardize with respect to the original climatology.
    source_standard = (source - clim_mean) / clim_std

    # Get value of the same quantile in the filtered climatology, keep anom.
    filtered_clim_std = _get_climatology_std(filtered_clim, variables, **sel)
    source_bc_anom = source_standard * filtered_clim_std

    # Interpolate anomaly to target grid.
    source_bc_anom_interp = source_bc_anom.interp(
        longitude=target_clim.coords['longitude'],
        latitude=target_clim.coords['latitude'],
        method='cubic',
    )

    # Add anom to the mean of the unfiltered climatology.
    target_clim_mean = _get_climatology_mean(target_clim, variables, **sel)
    source_bcsd = target_clim_mean + source_bc_anom_interp

    # Use multiplicative correction for precipitation variables
    if MULTIPLICATIVE_VARS.value:
      filtered_clim_mean = _get_climatology_mean(
          filtered_clim, MULTIPLICATIVE_VARS.value, **sel
      )
      for var in MULTIPLICATIVE_VARS.value:
        # Compute the full field corresponding to the filtered climatology
        # quantile. Then, keep the multiplicative anomaly.
        bc_mult_anom = (source_bc_anom[var] + filtered_clim_mean[var]) / (
            filtered_clim_mean[var]
        )

        # Interpolate anomaly to target grid.
        bc_mult_anom_interp = bc_mult_anom.interp(
            longitude=target_clim.coords['longitude'],
            latitude=target_clim.coords['latitude'],
            method='cubic',
        )

        # Multiply the anomaly by the target climatology.
        var_bcsd = bc_mult_anom_interp * target_clim_mean[var]
        source_bcsd = source_bcsd.drop_vars(var).assign(**{var: var_bcsd})

    source_bcsd = source_bcsd.drop_vars('dayofyear').transpose(
        'member', 'time', 'longitude', 'latitude'
    )
    return source_key, source_bcsd.compute()
  else:
    raise ValueError(f'BCSD method {method} not yet implemented.')


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

  # Form a sample for every year in historical reference. The BCSD output is
  # added as the daily mean.
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
  source_td = source_td.assign_coords(time=new_time.astype('datetime64[ns]'))
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


def main(argv: list[str]) -> None:

  lens_vars = list(_LENS_TO_ERA.keys())
  lens_vars.extend([v + '_std' for v in _LENS_TO_ERA.keys()])

  era_vars = list(_LENS_TO_ERA.values())
  era_vars.extend([v + '_std' for v in _LENS_TO_ERA.values()])

  clim_dict = {k + '_std': v + '_std' for k, v in _LENS_TO_ERA.items()}
  clim_dict.update(_LENS_TO_ERA)

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_DATA.value)
  source_dataset = _impose_data_selection(source_dataset)

  original_clim = xr.Dataset(
      xr.open_zarr(INPUT_DATA_CLIM.value).get(lens_vars)
  ).rename(clim_dict)

  filtered_clim = xr.Dataset(
      xr.open_zarr(FILTERED_TARGET_CLIM.value).get(era_vars)
  ).sel(level=1000, drop=True)

  target_clim = xr.Dataset(xr.open_zarr(TARGET_CLIM.value).get(era_vars))
  target_clim = target_clim.sel(level=1000, drop=True)

  hist_ref = xr.open_zarr(HIST_REF.value).get(list(_LENS_TO_ERA.values()))
  hist_ref = hist_ref.sel(level=1000, drop=True)
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
  template_start = template.time.values[0]
  template_end = template.time.values[-1] + np.timedelta64(1, 'D')
  new_time = np.arange(
      template_start, template_end, np.timedelta64(HOUR_RESOLUTION.value, 'h')
  )
  template = template.reindex(time=new_time, method='ffill')
  template = template.sel(
      time=template.time.dt.month.isin([int(m) for m in SELECT_MONTHS.value]),
      drop=True,
  )
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
      time=template.time.values.astype('datetime64[ns]')
  )
  template = template.transpose('member', 'time', 'longitude', 'latitude')
  logging.info('template: %s', template)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(source_dataset, in_working_chunks)
        | beam.MapTuple(
            functools.partial(
                _bcsd_on_chunks,
                original_clim=original_clim,
                filtered_clim=filtered_clim,
                target_clim=target_clim,
                method=METHOD.value,
            ),
        )
        | beam.MapTuple(functools.partial(_td_on_chunks, hist_ref=hist_ref))
        | xbeam.SplitChunks(target_chunks=split_chunks)
        | xbeam.ConsolidateChunks(target_chunks=output_chunks)
        | xbeam.ChunksToZarr(OUTPUT_PATH.value, template, output_chunks)
    )


if __name__ == '__main__':
  app.run(main)
