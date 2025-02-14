# Copyright 2024 The swirl_dynamics Authors.
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

r"""CLI to compute long-term detrended climatologies.

This script computes the climatology of a dataset after subtraction of its
long-term time trend. The long-term trend is assumed to be defined by
polynomial coefficients obtained using the `numpy.polyfit` function, and stored
in a separate dataset. See `compute_parametric_trends.py` for details. The trend
coefficients for each variable are assumed to be stored in a variable with the
same name as the original variable, but with a `_polyfit_coefficients` suffix.

Example usage:

```
# Path of the input dataset.
INPUT_PATH=<input_zarr_path>

# Path of the trend coefficients.
TREND_PATH=<trend_zarr_path>

# Year range of the computed climatology.
START_YEAR=2020
END_YEAR=2090

# Path of the output dataset to be produced.
OUTPUT_PATH=<output_zarr_path>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/compute_detrended_climatology.py \
  --input_path=${INPUT_PATH} \
  --output_path=${OUTPUT_PATH} \
  --trend_path=${TREND_PATH} \
  --window_size=31 \
  --frequency=hourly \
  --start_year=${START_YEAR} \
  --end_year=${END_YEAR}
```

"""

import functools
from typing import Callable, Optional, Union

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import data_utils
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import flag_utils
import xarray as xr
import xarray_beam as xbeam


# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
TREND_PATH = flags.DEFINE_string(
    'trend_path', None, help='Zarr path to long-term trend coefficients.'
)
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
FREQUENCY = flags.DEFINE_string(
    'frequency',
    'hourly',
    (
        'Frequency of the computed climatology. "hourly": Compute the'
        ' climatology per day of year and hour of day. "daily": Compute the'
        ' climatology per day of year.'
    ),
)
HOUR_INTERVAL = flags.DEFINE_integer(
    'hour_interval',
    1,
    help='Which intervals to compute hourly climatology for.',
)
WINDOW_SIZE = flags.DEFINE_integer(
    'window_size', 61, help='Window size in days to average over.'
)
START_YEAR = flags.DEFINE_integer(
    'start_year', 1990, help='Inclusive start year of climatology'
)
END_YEAR = flags.DEFINE_integer(
    'end_year', 2020, help='Inclusive end year of climatology'
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
WORKING_CHUNKS = flag_utils.DEFINE_chunks(
    'working_chunks',
    '',
    help=(
        'Chunk sizes overriding input chunks to use for computing climatology, '
        'e.g., "longitude=10,latitude=10".'
    ),
)
OUTPUT_CHUNKS = flag_utils.DEFINE_chunks(
    'output_chunks',
    '',
    help='Chunk sizes overriding input chunks to use for storing climatology',
)
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize', 4, help='Itemsize for rechunking.'
)
STATISTICS = flags.DEFINE_list(
    'statistics',
    ['mean'],
    help='Statistics to compute from "mean", "std", "seeps", "quantile".',
)
SPATIAL_DIMS = flags.DEFINE_list(
    'spatial_dims',
    ['south_north', 'west_east'],
    help='Name of the spatial dimensions of the dataset.',
)
QUANTILES = flags.DEFINE_list('quantiles', [], 'List of quantiles to compute.')
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)


class Quantile:
  """Compute quantiles."""

  def __init__(self, quantiles: list[float]):
    self.quantiles = quantiles

  def compute(
      self,
      ds: xr.Dataset,
      dim: tuple[str, ...],
      weights: Optional[xr.Dataset] = None,
  ):
    if weights is not None:
      ds = ds.weighted(weights)  # pytype: disable=wrong-arg-types
    return ds.quantile(self.quantiles, dim=dim)


def detrend_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    trend_ds: xr.Dataset,
    spatial_dims: tuple[str, str],
    chunks: dict[str, int],
) -> tuple[xbeam.Key, xr.Dataset]:
  """Subtract long-term trend of a chunk of 2D climate data over time.

  Args:
    obs_key: The key indexing the chunk.
    obs_chunk: The dataset chunk, including the entire time series of interest.
    trend_ds: The dataset containing the long-term trend coefficients.
    spatial_dims: The name of the two spatial dimensions of the dataset.
    chunks: The chunk sizes of the dataset, which include the `spatial_dims`.

  Returns:
    The key and the detrended dataset chunk.
  """
  offset_1 = obs_key.offsets[spatial_dims[0]]
  offset_2 = obs_key.offsets[spatial_dims[1]]
  chunksize_1 = chunks[spatial_dims[0]]
  chunksize_2 = chunks[spatial_dims[1]]
  trend_ds = trend_ds.isel(**{
      spatial_dims[0]: slice(offset_1, offset_1 + chunksize_1),
      spatial_dims[1]: slice(offset_2, offset_2 + chunksize_2),
  })
  for obs in list(obs_chunk.keys()):
    coeff = trend_ds[str(obs) + '_polyfit_coefficients']
    obs_chunk[str(obs)] = obs_chunk[str(obs)] - xr.polyval(
        obs_chunk['time'], coeff
    )
  return obs_key, obs_chunk


def compute_stat_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    frequency: str,
    window_size: int,
    clim_years: slice,
    statistic: Union[str, Callable[..., xr.Dataset]] = 'mean',
    hour_interval: Optional[int] = None,
    quantiles: Optional[list[float]] = None,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Compute climatology on a chunk."""
  if statistic not in ['mean', 'std', 'quantile']:
    raise NotImplementedError(f'stat {statistic} not implemented.')
  offsets = dict(dayofyear=0)
  if frequency == 'hourly':
    offsets['hour'] = 0
  clim_key = obs_key.with_offsets(time=None, **offsets)
  if statistic != 'mean':
    clim_key = clim_key.replace(
        vars={f'{var}_{statistic}' for var in clim_key.vars}
    )
    for var in obs_chunk:
      obs_chunk = obs_chunk.rename({var: f'{var}_{statistic}'})
  if statistic == 'quantile':
    statistic = Quantile(quantiles).compute
  compute_kwargs = {
      'obs': obs_chunk,
      'window_size': window_size,
      'clim_years': clim_years,
      'stat_fn': statistic,
  }

  if frequency == 'hourly':
    clim_chunk = data_utils.compute_hourly_stat(
        **compute_kwargs, hour_interval=hour_interval
    )
  elif frequency == 'daily':
    clim_chunk = data_utils.compute_daily_stat(**compute_kwargs)
  else:
    raise NotImplementedError(
        f'Climatological frequency {frequency} not implemented.'
    )
  return clim_key, clim_chunk


def main(argv: list[str]) -> None:
  spatial_dims = tuple(SPATIAL_DIMS.value)
  obs, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  ## Assign missing coords
  coords = {
      k: list(range(obs.sizes[k]))
      for k in list(obs.dims)
      if k not in list(obs.coords)
  }
  obs = obs.assign_coords(**coords)

  trend_ds = xr.open_zarr(TREND_PATH.value)

  # Convert object-type coordinates to string.
  # Required to avoid: https://github.com/pydata/xarray/issues/3476
  for coord_name, coord in obs.coords.items():
    if coord.dtype == 'object':
      obs[coord_name] = coord.astype(str)

  # drop static variables, for which the climatology calculation would fail
  obs = obs.drop_vars([k for k, v in obs.items() if 'time' not in v.dims])

  input_chunks_without_time = {
      k: v for k, v in input_chunks.items() if k != 'time'
  }

  if FREQUENCY.value == 'daily':
    stat_kwargs = {}
    clim_chunks = dict(dayofyear=-1)
    clim_dims = dict(dayofyear=1 + np.arange(366))
  elif FREQUENCY.value == 'hourly':
    stat_kwargs = dict(hour_interval=HOUR_INTERVAL.value)
    clim_chunks = dict(hour=-1, dayofyear=-1)
    clim_dims = dict(
        hour=np.arange(0, 24, HOUR_INTERVAL.value), dayofyear=1 + np.arange(366)
    )
  else:
    raise NotImplementedError(f'frequency {FREQUENCY.value} not implemented.')

  working_chunks = input_chunks_without_time.copy()
  working_chunks.update(WORKING_CHUNKS.value)
  in_working_chunks = dict(working_chunks, time=-1)
  out_working_chunks = dict(working_chunks, **clim_chunks)

  output_chunks = input_chunks_without_time.copy()
  output_chunks.update(clim_chunks)
  output_chunks.update(OUTPUT_CHUNKS.value)

  clim_template = (
      xbeam.make_template(obs).isel(time=0, drop=True).expand_dims(clim_dims)
  )

  raw_vars = list(clim_template)

  quantiles = [float(q) for q in QUANTILES.value]
  for stat in STATISTICS.value:
    if stat not in ['mean']:
      for var in raw_vars:
        if stat == 'quantile':
          if not quantiles:
            raise ValueError(
                'Cannot compute stat `quantile` without specifying --quantiles.'
            )
          quantile_dim = xr.DataArray(
              quantiles, name='quantile', dims=['quantile']
          )
          temp = clim_template[var].expand_dims(quantile=quantile_dim)
          if 'hour' in temp.dims:
            temp = temp.transpose('hour', 'quantile', ...)
        else:
          temp = clim_template[var]
        clim_template = clim_template.assign({f'{var}_{stat}': temp})
  # Mean has no suffix. Delete no suffix variables if no mean required
  if 'mean' not in STATISTICS.value:
    for var in raw_vars:
      clim_template = clim_template.drop(var)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Read and Rechunk
    pcoll = (
        root
        | xbeam.DatasetToChunks(
            obs,
            input_chunks,
            split_vars=True,
            num_threads=NUM_THREADS.value,
        )
        | 'RechunkIn'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            obs.sizes,
            input_chunks,
            in_working_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | 'Detrend'
        >> beam.MapTuple(
            functools.partial(
                detrend_chunk,
                trend_ds=trend_ds,
                spatial_dims=spatial_dims,
                chunks=in_working_chunks,
            )
        )
    )

    # Branches to compute statistics
    pcolls = []
    for stat in STATISTICS.value:
      # Mean and Std branches
      pcoll_tmp = pcoll | f'{stat}' >> beam.MapTuple(
          functools.partial(
              compute_stat_chunk,
              frequency=FREQUENCY.value,
              window_size=WINDOW_SIZE.value,
              clim_years=slice(str(START_YEAR.value), str(END_YEAR.value)),
              statistic=stat,
              quantiles=quantiles,
              **stat_kwargs,
          )
      )
      pcolls.append(pcoll_tmp)

    # Rechunk and write output
    _ = (
        pcolls
        | beam.Flatten()
        | 'RechunkOut'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            clim_template.sizes,
            out_working_chunks,
            output_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=clim_template,
            zarr_chunks=output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
