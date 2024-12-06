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

r"""Script to combine xarray datasets through arithmetic operations.

Example usage:

```
INPUT_EXPERIMENT=<parent_dir>/canesm5_r1i1p2f1_ssp370_bc

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/input_pipelines/combine_datasets.py \
  --input_path_1=${INPUT_EXPERIMENT}/hourly_d02_precip_snow_rates.zarr \
  --input_path_2=${INPUT_EXPERIMENT}/hourly_d01_cubic_interpolated_to_d02_precip_snow_rates.zarr \
  --output_path=${INPUT_EXPERIMENT}/hourly_diff_d02_d01_cubic_interpolated_to_d02_precip_snow_rates.zarr \
  --method=subtract
```

"""

from absl import app
from absl import flags
import apache_beam as beam
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import data_utils
import xarray as xr
import xarray_beam as xbeam


INPUT_PATH_1 = flags.DEFINE_string(
    'input_path_1',
    None,
    help='Input Zarr path pointing to first dataset to combine.',
)
INPUT_PATH_2 = flags.DEFINE_string(
    'input_path_2',
    None,
    help='Input Zarr path pointing to second dataset to combine.',
)
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
METHOD = flags.DEFINE_enum(
    'method',
    'subtract',
    ['subtract', 'add', 'merge'],
    'Method to use for combining datasets.',
)
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
TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'time_chunk_size',
    24,
    help='Chunk size for the time dimension of the output.',
)
SPATIAL_DIMS = flags.DEFINE_list(
    'spatial_dims',
    ['south_north', 'west_east'],
    help='Spatial dimensions of the datasets.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _combine_chunks(
    sources: list[xr.Dataset], method: str = 'subtract'
) -> xr.Dataset:
  """Combine two chunks by performing arithmetic operations between them.

  Args:
    sources: The datasets to be combined, ordered with respect to the desired
      combination method.
    method: The combination method. Add, subtract, and merge are currently
      implemented.

  Returns:
    The combined chunks as an xarray Dataset.
  """
  if len(sources) != 2:
    raise ValueError(f'Expected 2 sources, got {len(sources)}.')
  if method == 'subtract':
    return sources[0] - sources[1]
  elif method == 'add':
    return sources[0] + sources[1]
  elif method == 'merge':
    return xr.merge(sources)
  else:
    raise ValueError(f'Unknown combination method: {method}.')


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  if TIME_START.value is None or TIME_STOP.value is None:
    return ds
  selection = {TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)}
  return ds.sel({k: v for k, v in selection.items() if k in ds.dims})


def main(argv: list[str]) -> None:

  spatial_dims = tuple(SPATIAL_DIMS.value)

  source_dataset_1, source_chunks_1 = xbeam.open_zarr(INPUT_PATH_1.value)
  source_dataset_2, source_chunks_2 = xbeam.open_zarr(INPUT_PATH_2.value)

  source_dataset_1 = _impose_data_selection(source_dataset_1)
  source_dataset_2 = _impose_data_selection(source_dataset_2)
  source_dataset_1, source_dataset_2 = data_utils.align_datasets(
      source_dataset_1, source_dataset_2
  )
  source_datasets = [source_dataset_1, source_dataset_2]
  source_chunks = {
      k: min(
          max(source_chunks_1[k], source_chunks_2[k]), source_dataset_1.dims[k]
      )
      for k in source_chunks_1
  }
  in_working_chunks = source_chunks
  in_working_chunks[TIME_DIM.value] = TIME_CHUNK_SIZE.value
  output_chunks = in_working_chunks

  template = (xbeam.make_template(source_dataset_1)).transpose(
      TIME_DIM.value, spatial_dims[0], spatial_dims[1]
  )
  new_vars = set(source_dataset_2.data_vars) - set(source_dataset_1.data_vars)
  for variable in new_vars:
    template = template.assign(
        {variable: template[list(source_dataset_1.data_vars)[0]]}
    )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(source_datasets, in_working_chunks)
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _combine_chunks(v, method=METHOD.value),
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
