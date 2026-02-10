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

r"""Script to convert cumulative precipitation to precipitation rates.

The input dataset is expected to have variables RAINNC, RAINC, and SNOW.
The first two are cumulative precipitation amounts that accumulate throughout
the year. SNOW is the current snow water equivalent in kg/m2. The output dataset
will have variables RAIN_{period_1}h, RAIN_{period_2}h, ...,
which are the precipitation rates over periods of climatological interest; as
well as SNOW_{period_1}h, SNOW_{period_2}h, ..., which are the snow rates over
different periods of time, in kg/m2/s.

Contrary to cumulative precipitation, SNOW is the actual level of snow water
equivalent per location. This means that the rate of change of SNOW can be
negative, indicative of snow melt. Therefore, negative values of the snow rate
should not be filtered out.

Example usage:

```
INPUT_EXPERIMENT=<parent_dir>/access-cm2_r5i1p1f1_ssp370_bc
PATH_WITHOUT_FORMAT=hourly_d02

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/input_pipelines/compute_precip_rates.py \
  --accumulation_periods=24,48,72 \
  --input_path=${INPUT_EXPERIMENT}/${PATH_WITHOUT_FORMAT}.zarr \
  --output_path=${INPUT_EXPERIMENT}/${PATH_WITHOUT_FORMAT}_precip_snow_rates.zarr
```

"""

from absl import app
from absl import flags
import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam


# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
START_YEAR = flags.DEFINE_integer(
    'start_year', None, help='Inclusive start year of climatology'
)
END_YEAR = flags.DEFINE_integer(
    'end_year', None, help='Inclusive end year of climatology'
)
ACCUMULATION_PERIODS = flags.DEFINE_list(
    'accumulation_periods',
    ['6', '12', '24'],
    help='Periods of climatological interest for accumulation.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize', 4, help='Itemsize for rechunking.'
)


def _convert_precip_to_rates_chunk(
    obs_chunk: xr.Dataset, *, periods: list[int]
) -> xr.Dataset:
  """Convert precipitation to precipitation rates from a dataset chunk.

  Args:
    obs_chunk: The dataset chunk to process.
    periods: The periods of climatological interest for rate computation.

  Returns:
    A dataset chunk including the requested rates, and without RAINNC, RAINC,
    and SNOW.
  """
  rain_chunk = obs_chunk['RAINNC'] + obs_chunk['RAINC']
  snow_chunk = obs_chunk['SNOW']
  rates_chunk = obs_chunk
  for period in periods:
    # Compute precipitation rate.
    precip_rate = rain_chunk - rain_chunk.shift(time=period)
    # Accumulations have yearly negative discontinuities, so we filter them out.
    # Other than those, there may be small negative values due to interpolation.
    rates_chunk = rates_chunk.assign(
        **{f'RAIN_{period}h': xr.where(precip_rate > 0, precip_rate, 0)}
    )

    # Compute snow rates. Negative rates indicate snow melt.
    snow_rate = snow_chunk - snow_chunk.shift(time=period)
    rates_chunk = rates_chunk.assign(**{f'SNOW_{period}h': snow_rate})

  return rates_chunk.drop_vars(['RAINNC', 'RAINC', 'SNOW'])


def main(argv: list[str]) -> None:
  analysis_vars = ['RAINC', 'RAINNC', 'SNOW']
  periods = list(int(v) for v in ACCUMULATION_PERIODS.value)
  obs, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  # Only retain analysis_vars.
  obs = obs.get(analysis_vars)
  assert isinstance(obs, xr.Dataset)
  if START_YEAR.value is not None and END_YEAR.value is not None:
    time_slice = slice(str(START_YEAR.value), str(END_YEAR.value))
    obs = obs.sel(time=time_slice)

  time_working_chunks = dict(time=-1, west_east=1, south_north=1)

  # Add new vars to template
  template = xbeam.make_template(obs)
  for rate in periods:
    template = template.assign({f'RAIN_{rate}h': template['RAINNC']})
    template = template.assign({f'SNOW_{rate}h': template['SNOW']})
  for var in analysis_vars:
    template = template.drop(var)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:

    _ = (
        root
        | xbeam.DatasetToChunks(obs, input_chunks, split_vars=False)
        | 'RechunkToTimeSlices'
        >> xbeam.Rechunk(
            # Convert to string to satisfy pytype.
            {str(k): v for k, v in obs.sizes.items()},
            input_chunks,
            time_working_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _convert_precip_to_rates_chunk(v, periods=periods),
            )
        )
        | 'RechunkBack'
        >> xbeam.Rechunk(
            {str(k): v for k, v in obs.sizes.items()},
            time_working_chunks,
            input_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            input_chunks,
        )
    )

if __name__ == '__main__':
  app.run(main)
