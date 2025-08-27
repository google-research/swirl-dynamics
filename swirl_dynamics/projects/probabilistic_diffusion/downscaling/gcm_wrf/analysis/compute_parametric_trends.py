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

r"""Script to compute long-term parametric trends of climate data.

The script computes the coefficients of a polynomial fit of degree `degree`
over time for each variable in the input dataset. The coefficients are stored in
a dataset with the same spatial dimensions as the input dataset, but with an
additional `degree` dimension. The `degree` dimension is ordered from `degree`
to 0, where 0 corresponds to the constant term and `degree` corresponds to
the highest degree term.

Example usage:

```
INPUT_PATH=<input_zarr_path>
OUTPUT_PATH=<output_zarr_path>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/compute_parametric_trends.py \
  --input_path=${INPUT_PATH} \
  --output_path=${OUTPUT_PATH} \
  --degree=3
```

"""

import functools

from absl import app
from absl import flags
import apache_beam as beam
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import flag_utils
import xarray as xr
import xarray_beam as xbeam


DEGREE = flags.DEFINE_integer('degree', 3, 'Degree of polynomial to fit.')
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize', 4, help='Itemsize for rechunking.'
)
WORKING_CHUNKS = flag_utils.DEFINE_chunks(
    'working_chunks',
    'south_north=2,west_east=2',
    help=(
        'Chunk sizes overriding input chunks to use for computation, '
        'e.g., "south_north=10,west_east=10".'
    ),
)


def compute_chunk_trend(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    degree: int,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Compute long-term parametric trend of a chunk over time."""
  offsets = {'time': None, 'degree': 0}
  # Cast to float64 to avoid precision issues.
  coeff_chunk = obs_chunk.astype('float64').polyfit(
      dim='time', deg=degree, skipna=True
  )
  coeff_key = obs_key.with_offsets(**offsets)
  coeff_key = coeff_key.replace(
      vars={f'{var}_polyfit_coefficients' for var in coeff_key.vars}
  )
  return coeff_key, coeff_chunk


def main(argv: list[str]) -> None:
  degree = DEGREE.value
  obs, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  # Drop variables without time dimension.
  vars_to_drop = [v for v in obs.data_vars if 'time' not in obs[v].dims]
  obs = obs.drop_vars(vars_to_drop)
  # Assign missing coords.
  coords = {
      k: list(range(obs.sizes[k]))
      for k in list(obs.dims)
      if k not in list(obs.coords)
  }
  obs = obs.assign_coords(**coords)

  # Chunks for temporal reduce.
  in_working_chunks = {'time': -1}
  # Update with spatial chunks
  in_working_chunks.update(WORKING_CHUNKS.value)

  # Chunks for coefficient dataset.
  out_working_chunks = {
      k: v for k, v in in_working_chunks.items() if k != 'time'
  }
  out_working_chunks['degree'] = -1
  out_working_sizes = {
      k: obs.sizes[k] for k in out_working_chunks.keys() if k in obs.dims
  }
  out_working_sizes['degree'] = degree + 1
  out_chunks = {
      k: input_chunks[k] for k in out_working_chunks.keys() if k in obs.dims
  }
  out_chunks['degree'] = -1

  # Define template with float64 precision
  expand_dims = dict(degree=list(range(degree, -1, -1)))
  template = (
      xbeam.make_template(obs)
      .isel(**{'time': 0}, drop=True)
      .expand_dims(**expand_dims)
      .astype('float64')
  )
  # Substitute vars by coefficients in template
  raw_vars = list(template)
  for var in raw_vars:
    template = template.assign({f'{var}_polyfit_coefficients': template[var]})
    template = template.drop(var)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(obs, input_chunks, split_vars=True)
        | 'RechunkIn'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            obs.sizes,
            input_chunks,
            in_working_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | 'EvaluateTrends'
        >> beam.MapTuple(functools.partial(compute_chunk_trend, degree=degree))
        | xbeam.ConsolidateChunks(out_chunks)
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=template,
            zarr_chunks=out_chunks,
        )
    )


if __name__ == '__main__':
  app.run(main)
