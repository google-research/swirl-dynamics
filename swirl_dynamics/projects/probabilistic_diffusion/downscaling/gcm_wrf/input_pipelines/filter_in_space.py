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

r"""Script to filter WUS data in space.

Example usage:
```
INPUT_EXPERIMENT=<parent_dir>/canesm5_r1i1p2f1_ssp370_bc

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/input_pipelines/filter_in_space.py \
  --input_path=${INPUT_EXPERIMENT}/hourly_d02_with_prates.zarr \
  --output_path=${INPUT_EXPERIMENT}/hourly_filtered_d02_with_prates.zarr
```

"""

import functools

from absl import app
from absl import flags
import apache_beam as beam
from scipy import ndimage
import xarray as xr
import xarray_beam as xbeam


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
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
SCALE = flags.DEFINE_float(
    'scale',
    2.0,
    help='Scale (in pixels) used to construct the Gaussian filter.',
)
MODE = flags.DEFINE_string(
    'mode',
    'nearest',
    help=(
        'Determines how the input array is extended when the filter overlaps a'
        ' border.'
    ),
)
VARIABLES = flags.DEFINE_list(
    'variables',
    None,
    help='Variables retained in the output dataset.',
)
SPATIAL_DIMS = flags.DEFINE_list(
    'spatial_dims',
    ['south_north', 'west_east'],
    help='Spatial dimensions to filter over.',
)
TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'time_chunk_size',
    24,
    help='Chunk size for the time dimension of the output.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _filter_chunk_in_space(
    source: xr.Dataset,
    *,
    scale: float = 2.0,
    mode: str = 'nearest',
    spatial_dims: tuple[str, str] = ('south_north', 'west_east'),
) -> xr.Dataset:
  """Filters a chunk in its spatial dimensions.

  Args:
    source: The source dataset chunk to interpolate to a new spatial grid.
    scale: Scale (in pixels) used to construct the Gaussian filter.
    mode: Determines how the input array is extended when the filter overlaps a
      border.
    spatial_dims: The dimensions of the source dataset to filter over.

  Returns:
    The spatially filtered chunk.
  """
  # Ordered dimensions of the source dataset can be fetched from a data_var.
  # Dataset dimensions are not guaranteed to be ordered.
  ordered_dims = source[list(source.data_vars)[0]].dims
  # We only filter the spatial dimensions.
  scales = tuple(scale * float(dim in spatial_dims) for dim in ordered_dims)

  gaussian_filter = functools.partial(
      ndimage.gaussian_filter, sigma=scales, mode=mode
  )
  return xr.apply_ufunc(gaussian_filter, source.load())


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  if TIME_START.value or TIME_STOP.value:
    selection = {TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)}
    ds = ds.sel({k: v for k, v in selection.items() if k in ds.dims})
  if VARIABLES.value:
    ds = xr.Dataset(ds.get(VARIABLES.value))
  return ds


def main(argv: list[str]) -> None:

  spatial_dims = tuple(SPATIAL_DIMS.value)
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
  source_dataset = _impose_data_selection(source_dataset)
  source_chunks = {k: source_chunks[k] for k in source_chunks}
  in_working_chunks = source_chunks.copy()

  output_chunks = source_chunks.copy()
  output_chunks['time'] = TIME_CHUNK_SIZE.value

  template = xbeam.make_template(source_dataset)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(
            source_dataset, in_working_chunks, split_vars=True
        )
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _filter_chunk_in_space(
                    v,
                    scale=SCALE.value,
                    mode=MODE.value,
                    spatial_dims=spatial_dims,
                ),
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
