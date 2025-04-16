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

r"""Script to interpolate GenBCSR data to WUS-D3 spatial domains.

This script is used to interpolate GenBCSR data to the western United States
domains covered by the WUS-D3 dynamical downscaling dataset, described in
https://doi.org/10.5194/gmd-17-2265-2024.

The output format is the same as the input format, with the spatial
coordinates interpolated to the WUS-D3 domains. The output spatial dimensions
are denoted as `south_north` and `west_east`. Latitude and longitude coordinates
are a function of these dimensions, as defined by the dataset with path
`out_coords_path`.

"""

import functools
import logging

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from scipy import interpolate
import xarray as xr
import xarray_beam as xbeam


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
IN_COORDS_NAMES = flags.DEFINE_list(
    'in_coords_names',
    ['latitude', 'longitude'],
    help=(
        'Names of the input spatial coordinates, strictly in latitude,'
        ' longitude order.'
    ),
)
OUT_COORDS_PATH = flags.DEFINE_string(
    'out_coords_path', None, help='Output coordinates Zarr dataset path.'
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
METHOD = flags.DEFINE_string(
    'method',
    'cubic',
    help='The interpolation method to use.',
)
MEMBER_SAMPLING_RATE = flags.DEFINE_integer(
    'member_sampling_rate',
    4,
    help='The sampling rate for the member dimension in the output dataset.',
)
TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'time_chunk_size',
    24,
    help='Chunk size for the time dimension of the output.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _interpolate_chunk(
    source_key: xbeam.Key,
    source: xr.Dataset,
    *,
    out_spatial_shape: tuple[int, int],
    source_grid: tuple[np.ndarray, np.ndarray],
    target_grid: tuple[np.ndarray, np.ndarray],
    source_coord_names: tuple[str, str] = ('latitude', 'longitude'),
    method: str = 'cubic'
) -> tuple[xbeam.Key, xr.Dataset]:
  """Interpolate a chunk to a new grid.

  Args:
    source_key: The key for the source data chunk.
    source: The source dataset to interpolate to a new spatial grid.
    out_spatial_shape: A tuple defining the output spatial shape.
    source_grid: A tuple of 1d arrays containing the source grid coordinates,
      where the first array defines the latitude of each data point, and the
      second defines the longitude.
    target_grid: The target grid, with the same format as `source_grid`.
    source_coord_names: The names of the spatial coordinates to be interpolated
      in the source dataset.
    method: The interpolation method.

  Returns:
    The key and the dataset interpolated to the new grid.
  """
  interp_key = source_key.with_offsets(
      south_north=0,
      west_east=0,
      **{coord: None for coord in source_coord_names}
  )
  other_dims = [k for k in source.dims if k not in IN_COORDS_NAMES.value]
  interp_arrays = []
  source = source.transpose(..., 'latitude', 'longitude')
  for varname, data_var in source.data_vars.items():
    shape = data_var.shape[:-2] + out_spatial_shape
    interp_data = interpolate.griddata(
        source_grid, data_var.values.flatten(), target_grid, method=method
    )
    da = xr.DataArray(
        name=varname,
        data=interp_data.reshape(shape),
        dims=[*other_dims, 'south_north', 'west_east'],
    ).transpose(..., 'south_north', 'west_east')
    interp_arrays.append(da)

  return interp_key, xr.merge(interp_arrays)


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  if TIME_START.value is not None or TIME_STOP.value is not None:
    selection = {TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)}
    ds = ds.sel({k: v for k, v in selection.items() if k in ds.dims})
  return ds.isel(member=slice(None, None, MEMBER_SAMPLING_RATE.value))


def main(argv: list[str]) -> None:
  lat_dim, lon_dim = IN_COORDS_NAMES.value
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)

  in_coords = source_dataset.assign_coords(
      longitude=(((source_dataset[lon_dim] + 180) % 360) - 180)
  )
  in_lat, in_long = np.meshgrid(
      in_coords[lat_dim],
      in_coords[lon_dim],
      indexing='ij',
  )
  in_long = in_long.flatten()
  in_lat = in_lat.flatten()
  logging.info('in_lat: %s', in_lat)
  logging.info('in_long: %s', in_long)

  other_dims = [
      k for k in source_dataset.dims if k not in IN_COORDS_NAMES.value
  ]
  out_coords_dataset, _ = xbeam.open_zarr(OUT_COORDS_PATH.value)
  out_lat = out_coords_dataset['XLAT'].values.flatten()
  out_long = out_coords_dataset['XLONG'].values.flatten()
  out_spatial_shape = out_coords_dataset['XLAT'].shape
  logging.info('out_lat: %s', out_lat)
  logging.info('out_long: %s', out_long)

  source_dataset = _impose_data_selection(source_dataset)
  logging.info('source_dataset: %s', source_dataset)
  source_chunks = {
      k: min(source_dataset.sizes[k], source_chunks[k]) for k in source_chunks
  }
  in_working_chunks = {}
  # Interpolation is performed on 2D spatial slices.
  for k in source_chunks:
    if k not in IN_COORDS_NAMES.value:
      in_working_chunks[k] = 1
    else:
      in_working_chunks[k] = -1

  output_chunks = {k: v for k, v in out_coords_dataset.sizes.items()}
  for k in other_dims:
    output_chunks[k] = source_chunks[k]
  output_chunks['time'] = TIME_CHUNK_SIZE.value
  logging.info('output_chunks: %s', output_chunks)

  template = (
      xbeam.make_template(source_dataset)
      .isel(longitude=0, latitude=0, drop=True)
      .expand_dims(out_coords_dataset.sizes)
  ).transpose(..., 'south_north', 'west_east')
  logging.info('template: %s', template)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks, split_vars=True)
        | xbeam.Rechunk(
            source_dataset.sizes,  # pytype: disable=wrong-arg-types
            source_chunks,
            in_working_chunks,
            itemsize=4,
            max_mem=2**30,
        )
        | 'Interpolate'
        >> beam.MapTuple(
            functools.partial(
                _interpolate_chunk,
                out_spatial_shape=out_spatial_shape,
                source_grid=(in_lat, in_long),
                target_grid=(out_lat, out_long),
                method=METHOD.value,
                source_coord_names=(lat_dim, lon_dim),
            )
        )
        | xbeam.ConsolidateChunks(output_chunks)
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            output_chunks,
            num_threads=8,
        )
    )


if __name__ == '__main__':
  app.run(main)
