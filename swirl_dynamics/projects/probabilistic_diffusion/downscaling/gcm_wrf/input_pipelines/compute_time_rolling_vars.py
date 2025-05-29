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

r"""Script to compute time-rolling variables.

We consider an input dataset with variables [foo, bar, ...] that have a
time dimension. Given a list of periods of interest (in increments at the
temporal resolution of the input data), this script returns a dataset with
variables that represent time-rolling operations on the input variables for the
given periods:
- Variables accumulated over periods of time, passed to
  `variables_to_accumulate`. If foo is the variable, and 3 is the period, the
  output variable will be foo_sum_3days.
- Variables changes over periods of time, passed to `variables_to_subtract`. If
  foo is the variable, and 3 is the period, the output variable will be
  foo_diff_3days.
- Variables averaged over periods of time, passed to `variables_to_average`. If
  foo is the variable, and 3 is the period, the output variable will be
  foo_mean_3days.

"""

from typing import Literal

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
ROLLING_PERIODS = flags.DEFINE_list(
    'rolling_periods',
    ['3'],
    help='Periods of climatological interest for rolling operations.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize', 4, help='Itemsize for rechunking.'
)
VARIABLES_TO_ACCUMULATE = flags.DEFINE_list(
    'variables_to_accumulate',
    ['prec', 'prec_snow', 'sfc_runoff'],
    help='Variables to accumulate over rolling periods.',
)
VARIABLES_TO_SUBTRACT = flags.DEFINE_list(
    'variables_to_subtract',
    ['snow'],
    help='Variables to subtract over rolling periods.',
)
VARIABLES_TO_AVERAGE = flags.DEFINE_list(
    'variables_to_average',
    ['ivt', 't2'],
    help='Variables to average over rolling periods.',
)
NUM_THREADS_TO_ZARR = flags.DEFINE_integer(
    'num_threads_to_zarr',
    8,
    help='Number of threads to use for writing to Zarr.',
)


def _compute_time_change(
    obs_chunk: xr.Dataset, *, variables: list[str], periods: list[int]
) -> xr.Dataset:
  """Obtain time changes in variables from a dataset chunk.

  Args:
    obs_chunk: The dataset chunk to process.
    variables: The variables to compute time changes for.
    periods: The periods of climatological interest for change computation.

  Returns:
    A dataset chunk including the changes of the requested variables, over
    the requested periods.
  """
  output_chunk = obs_chunk
  for var in variables:
    var_chunk = obs_chunk[var]
    for period in periods:
      # Compute backwards difference change.
      change = var_chunk - var_chunk.shift(time=period)
      if period == 1:
        suffix = '1day'
      else:
        suffix = f'{period}days'
      output_chunk = output_chunk.assign(**{f'{var}_diff_{suffix}': change})

  return output_chunk.drop_vars(variables)


def _compute_rolling_operation(
    obs_chunk: xr.Dataset,
    *,
    variables: list[str],
    periods: list[int],
    operation: Literal['sum', 'mean'],
) -> xr.Dataset:
  """Compute rolling operation in variables from a dataset chunk.

  The operation is computed always keeping the current time as the last time
  in the rolling window.

  Args:
    obs_chunk: The dataset chunk to process.
    variables: The variables to compute rolling operations for.
    periods: The periods of climatological interest for accumulation.
    operation: The rolling operation to perform.

  Returns:
    A dataset chunk including the rolling operation of the requested variables,
    over the requested periods.
  """
  output_chunk = obs_chunk
  for var in variables:
    var_chunk = obs_chunk[var]
    for period in periods:
      if operation == 'sum':
        acc = var_chunk.rolling(time=period, center=False).sum()
      elif operation == 'mean':
        acc = var_chunk.rolling(time=period, center=False).mean()
      else:
        raise ValueError(f'Unknown operation: {operation}.')
      output_chunk = output_chunk.assign(
          {f'{var}_{operation}_{period}days': acc}
      )

  return output_chunk.drop_vars(variables)


def main(argv: list[str]) -> None:
  analysis_vars = (
      VARIABLES_TO_ACCUMULATE.value
      + VARIABLES_TO_SUBTRACT.value
      + VARIABLES_TO_AVERAGE.value
  )
  periods = list(int(v) for v in ROLLING_PERIODS.value)
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

  # For subtracted variables, we always add the one day diff.
  for var in VARIABLES_TO_SUBTRACT.value:
    template[f'{var}_diff_1day'] = template[var].assign_attrs(
        {'units': f'{template[var].attrs["units"]}/d'}
    )

  for period in periods:
    for var in VARIABLES_TO_ACCUMULATE.value:
      template = template.assign({f'{var}_sum_{period}days': template[var]})
      template[f'{var}_sum_{period}days'] = template[var].assign_attrs(
          {'units': f'{template[var].attrs["units"].split("/d")[0]}/{period}d'}
      )

    for var in VARIABLES_TO_AVERAGE.value:
      template = template.assign({f'{var}_mean_{period}days': template[var]})

    for var in VARIABLES_TO_SUBTRACT.value:
      template = template.assign({f'{var}_diff_{period}days': template[var]})
      template[f'{var}_diff_{period}days'] = template[var].assign_attrs(
          {'units': f'{template[var].attrs["units"]}/{period}d'}
      )

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
                _compute_time_change(
                    v,
                    periods=[1] + periods,
                    variables=VARIABLES_TO_SUBTRACT.value,
                ),
            )
        )
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _compute_rolling_operation(
                    v,
                    periods=periods,
                    variables=VARIABLES_TO_ACCUMULATE.value,
                    operation='sum',
                ),
            )
        )
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _compute_rolling_operation(
                    v,
                    periods=periods,
                    variables=VARIABLES_TO_AVERAGE.value,
                    operation='mean',
                ),
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
            num_threads=NUM_THREADS_TO_ZARR.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
