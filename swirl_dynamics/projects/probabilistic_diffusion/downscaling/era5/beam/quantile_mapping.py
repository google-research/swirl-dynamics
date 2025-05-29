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

r"""Flume script to perform quantile mapping.

This is essentially the "BC" part of the BCSD method (see bcsd.py for full
description of the algorithm).

Example Usage:

```
python swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5/beam:quantile_mapping.par -- \
  --input_data=/data/lens2/lens2_240x121_lonlat.zarr \
  --input_data_clim=/data/lens2/climat/lens2_240x121_lonlat_clim_daily_1961_to_2000_31_dw.zarr \
  --filtered_target_clim=/data/era5/selected_variables/climat/1p5deg_dailymean_7vars_windspeed_clim_daily_1961_to_2000_31_dw.zarr \
  --time_start=1960 \
  --time_stop=2100 \
  --method=gaussian \
  --output_path=/data/baselines/gaussian_qm/lens2_240x121_1960to2100.zarr
```

"""

import functools
import typing

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam


_LENS_TO_ERA = {
    'TREFHT': '2m_temperature',
    'WSPDSRFAV': '10m_magnitude_of_wind',
    'QREFHT': 'specific_humidity',
    'PSL': 'mean_sea_level_pressure',
}

INPUT_DATA = flags.DEFINE_string(
    'input_data',
    None,
    help='Zarr path pointing to the input data to be processed.',
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
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
METHOD = flags.DEFINE_enum(
    'method',
    'gaussian',
    ['gaussian', 'quantile_interpolation'],
    'Method to use for quantile mapping.',
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


def _qm_on_chunks(
    source_key: xbeam.Key,
    source: xr.Dataset,
    *,
    original_clim: xr.Dataset,
    filtered_clim: xr.Dataset,
    method: str = 'gaussian',
) -> tuple[xbeam.Key, xr.Dataset]:
  """Process an input data chunk with quantile mapping.

  Args:
    source_key: The key for the source data chunk.
    source: The source data chunk to be processed with the BCSD method. It is
      assumed to contain a single time slice.
    original_clim: The climatology of the source data.
    filtered_clim: The climatology of the spatially filtered target data,
      co-located with the source data.
    method: The method to use for quantile mapping.

  Returns:
    The source key and the BCSD data chunk.
  """
  sel = dict(dayofyear=source['time'].dt.dayofyear, drop=True)
  lonlat = {'longitude': source.longitude, 'latitude': source.latitude}
  variables = [str(key) for key in source.keys()]
  if method == 'gaussian':
    # Interp ensures grids match; otherwise results can be in different shape
    clim_mean = _get_climatology_mean(original_clim, variables, **sel).interp(
        lonlat, method='nearest'
    )
    clim_std = _get_climatology_std(original_clim, variables, **sel).interp(
        lonlat, method='nearest'
    )

    # Standardize with respect to the original climatology.
    source_standard = (source - clim_mean) / clim_std

    # Get value of the same quantile in the filtered climatology, keep anom.
    filtered_clim_mean = _get_climatology_mean(
        filtered_clim, variables, **sel
    ).interp(lonlat, method='nearest')
    filtered_clim_std = _get_climatology_std(
        filtered_clim, variables, **sel
    ).interp(lonlat, method='nearest')

    source_qmapped = source_standard * filtered_clim_std + filtered_clim_mean
    source_qmapped = source_qmapped.drop_vars('dayofyear').transpose(
        'member', 'time', 'longitude', 'latitude'
    )
    return source_key, source_qmapped.compute()
  else:
    raise ValueError(f'BCSD method {method} not yet implemented.')


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  """Imposes time range and month selection on the input dataset."""
  ds_sel = xr.Dataset(ds.get(list(_LENS_TO_ERA.keys()))).rename(_LENS_TO_ERA)
  if TIME_START.value is None and TIME_STOP.value is None:
    return ds_sel

  selection = {'time': slice(TIME_START.value, TIME_STOP.value)}
  selected = ds_sel.sel(
      {k: v for k, v in selection.items() if k in ds.dims}, drop=True
  )
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

  source_chunks = {k: source_chunks[k] for k in source_chunks if k != 'level'}
  logging.info('source_chunks: %s', source_chunks)

  template = xbeam.make_template(source_dataset)
  template = template.transpose('member', 'time', 'longitude', 'latitude')
  logging.info('template: %s', template)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks)
        | beam.MapTuple(
            functools.partial(
                _qm_on_chunks,
                original_clim=original_clim,
                filtered_clim=filtered_clim,
                method=METHOD.value,
            )
        )
        | xbeam.ChunksToZarr(OUTPUT_PATH.value, template, source_chunks)
    )


if __name__ == '__main__':
  app.run(main)
