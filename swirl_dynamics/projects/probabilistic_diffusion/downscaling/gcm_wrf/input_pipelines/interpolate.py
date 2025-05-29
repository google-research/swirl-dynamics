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

r"""Script to interpolate data between WUS spatial domains.

Example usage:
```
INPUT_EXPERIMENT=<parent_dir>/canesm5_r1i1p2f1_ssp370_bc
COORDS_PATH=<dir>/wrf-coordinates

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/input_pipelines/interpolate.py \
  --input_path=${INPUT_EXPERIMENT}/hourly_d01_only_prates.zarr \
  --output_path=${INPUT_EXPERIMENT}/hourly_d01_only_prates_cubic_interpolated_to_d02.zarr \
  --in_coords_path=${COORDS_PATH}/wrfinput_d01.zarr \
  --out_coords_path=${COORDS_PATH}/wrfinput_d02.zarr
```

"""

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from scipy import interpolate
import xarray as xr
import xarray_beam as xbeam


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
IN_COORDS_PATH = flags.DEFINE_string(
    'in_coords_path', None, help='Source coordinates Zarr dataset path.'
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
TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'time_chunk_size',
    24,
    help='Chunk size for the time dimension of the output.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _interpolate_chunk(
    source: xr.Dataset,
    *,
    out_spatial_shape: tuple[int, int],
    source_grid: tuple[np.ndarray, np.ndarray],
    target_grid: tuple[np.ndarray, np.ndarray],
    method: str = 'cubic',
) -> xr.Dataset:
  """Interpolate a chunk to a new grid.

  Args:
    source: The source dataset to interpolate to a new spatial grid.
    out_spatial_shape: A tuple defining the output spatial shape.
    source_grid: A tuple of 1d arrays containing the source grid coordinates,
      where the first array defines the latitude of each data point, and the
      second defines the longitude.
    target_grid: The target grid, with the same formar as `source_grid`.
    method: The interpolation method.

  Returns:
    The dataset interpolated to the new grid.
  """
  interp_arrays = []
  for varname, data_var in source.data_vars.items():
    shape = data_var.shape[:-2] + out_spatial_shape
    interp_data = interpolate.griddata(
        source_grid, data_var.values.flatten(), target_grid, method=method
    )
    da = xr.DataArray(
        name=varname,
        data=interp_data.reshape(shape),
        dims=['time', 'south_north', 'west_east'],
    ).transpose('time', 'south_north', 'west_east')
    interp_arrays.append(da)

  return xr.merge(interp_arrays)


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  if TIME_START.value is None or TIME_STOP.value is None:
    return ds
  selection = {TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)}
  return ds.sel({k: v for k, v in selection.items() if k in ds.dims})


def main(argv: list[str]) -> None:

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
  in_coords_dataset, _ = xbeam.open_zarr(IN_COORDS_PATH.value)
  in_lat = in_coords_dataset['XLAT'].values.flatten()
  in_long = in_coords_dataset['XLONG'].values.flatten()

  out_coords_dataset, _ = xbeam.open_zarr(OUT_COORDS_PATH.value)
  out_lat = out_coords_dataset['XLAT'].values.flatten()
  out_long = out_coords_dataset['XLONG'].values.flatten()
  out_spatial_shape = out_coords_dataset['XLAT'].shape

  source_dataset = _impose_data_selection(source_dataset)
  source_chunks = {
      k: min(source_dataset.dims[k], source_chunks[k]) for k in source_chunks
  }
  in_working_chunks = source_chunks
  in_working_chunks['time'] = 1

  output_chunks = {k: v for k, v in out_coords_dataset.dims.items()}
  output_chunks['time'] = TIME_CHUNK_SIZE.value

  template = (
      xbeam.make_template(source_dataset)
      .isel(south_north=0, west_east=0, drop=True)
      .expand_dims(out_coords_dataset.dims)
  ).transpose('time', 'south_north', 'west_east')

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(
            source_dataset, in_working_chunks
        )  # split_vars=False
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _interpolate_chunk(
                    v,
                    out_spatial_shape=out_spatial_shape,
                    source_grid=(in_lat, in_long),
                    target_grid=(out_lat, out_long),
                    method=METHOD.value,
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
