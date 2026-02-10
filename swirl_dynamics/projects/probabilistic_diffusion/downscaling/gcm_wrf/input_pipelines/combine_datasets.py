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
from collections.abc import Mapping

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


def _propagate_missing_coords(
    dataset_1: xr.Dataset,
    dataset_2: xr.Dataset,
    chunks_1: Mapping[str, int],
    chunks_2: Mapping[str, int],
) -> tuple[xr.Dataset, dict[str, int]]:
  """Ensures that the second dataset has the same coordinates as the first.

  This is ensured both for the dataset and the chunking scheme. The function
  assumes that the first dataset has more coordinates than the second dataset.

  Args:
    dataset_1: The first dataset.
    dataset_2: The second dataset.
    chunks_1: The chunks of the first dataset.
    chunks_2: The chunks of the second dataset.

  Returns:
    The updated dataset_2 and the common chunking scheme.
  """
  assert len(dataset_1.coords) >= len(dataset_2.coords)

  dataset_2 = dataset_2.assign_coords(
      {k: v for k, v in dataset_1.coords.items() if k not in dataset_2.coords}
  )
  # Get common chunks.
  chunks = {
      k: min(
          max(chunks_1[k], chunks_2[k]),
          dataset_2.dims[k],
      )
      for k in chunks_2
  }
  # Get chunks exclusive to the first dataset.
  chunks.update({k: v for k, v in chunks_1.items() if k not in chunks_2})
  return dataset_2, chunks


def main(argv: list[str]) -> None:

  spatial_dims = tuple(SPATIAL_DIMS.value)

  source_dataset_1, source_chunks_1 = xbeam.open_zarr(INPUT_PATH_1.value)
  source_dataset_2, source_chunks_2 = xbeam.open_zarr(INPUT_PATH_2.value)

  # In order to load chunks from multiple datasets, we need to make sure
  # that they have the same coordinates. If the first dataset has more
  # coordinates, we assign the second dataset the missing coordinates from the
  # first dataset, and vice versa. Note that this does not affect the dimensions
  # of the data arrays, which are determined by the actual data.
  if len(source_dataset_1.coords) > len(source_dataset_2.coords):
    source_dataset_2, source_chunks = _propagate_missing_coords(
        source_dataset_1, source_dataset_2, source_chunks_1, source_chunks_2
    )
  else:
    source_dataset_1, source_chunks = _propagate_missing_coords(
        source_dataset_2, source_dataset_1, source_chunks_2, source_chunks_1
    )

  source_dataset_1 = _impose_data_selection(source_dataset_1)
  source_dataset_2 = _impose_data_selection(source_dataset_2)
  source_dataset_1, source_dataset_2 = data_utils.align_datasets(
      source_dataset_1, source_dataset_2
  )
  source_datasets = [source_dataset_1, source_dataset_2]
  in_working_chunks = source_chunks
  in_working_chunks[TIME_DIM.value] = TIME_CHUNK_SIZE.value
  output_chunks = in_working_chunks

  template_1 = (xbeam.make_template(source_dataset_1)).transpose(
      TIME_DIM.value, ..., *spatial_dims
  )
  template_2 = (xbeam.make_template(source_dataset_2)).transpose(
      TIME_DIM.value, ..., *spatial_dims
  )
  # Output variables are the union of the two datasets.
  if METHOD.value == 'merge':
    template = xr.merge([template_1, template_2])
  # Output variables for ['add', 'subtract'] methods.
  else:
    template = template_1 + template_2

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
