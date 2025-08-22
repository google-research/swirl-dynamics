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

r"""Script to compute the spatial radial spectral density of variables.

Example usage:

```
INFERENCE_PATH=<inference_zarr_path>
OUTPUT_PATH=<output_zarr_path>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5/beam/compute_spatial_radial_spec.py \
  --inference_path=${INFERENCE_PATH} \
  --output_path=${OUTPUT_PATH} \
  --year_start=2001 \
  --year_end=2010 \
  --months=6,7,8 \
  --num_bins=256
```

"""

import functools

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.eval import utils as eval_utils
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils as pipeline_utils
import xarray as xr
import xarray_beam as xbeam

# Command line arguments
INFERENCE_PATH = flags.DEFINE_string(
    'inference_path', None, help='Input Zarr path.'
)
# If a reference path is provided, the script will compute tail dependence on
# the reference dataset instead of the inference dataset.
REFERENCE_PATH = flags.DEFINE_string(
    'reference_path', None, help='Reference Zarr path.'
)
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path.')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize', 4, help='Itemsize for rechunking.'
)
YEAR_START = flags.DEFINE_integer(
    'year_start', 2001, help='Starting year for evaluation.'
)
YEAR_END = flags.DEFINE_integer(
    'year_end', 2010, help='Ending year for evaluation.'
)
MONTHS = flags.DEFINE_list('months', ['8'], help='Months for evaluation.')
# Compute tail dependence on the data only at a specific hour of day to reduce
# the effects of diurnal cycles.
HOUR_OF_DAY = flags.DEFINE_integer(
    'hour_of_day', None, help='Hour of day in UTC time.'
)
NUM_BINS = flags.DEFINE_integer(
    'num_bins', 256, help='Number of bins for radial frequency.'
)


Variable = pipeline_utils.DatasetVariable

SAMPLE_VARIABLES = [
    Variable('2m_temperature', None, '2mT'),
    Variable('10m_magnitude_of_wind', None, '10mW'),
    Variable('2m_specific_humidity', None, '2mQ'),
    Variable('mean_sea_level_pressure', None, 'MSL'),
    # Derived variables. These are computed beforehand.
    Variable('relative_humidity', None, 'RH'),
    Variable('heat_index', None, 'HI'),
]


def eval_spatial_radial_psd_chunk(
    key: xbeam.Key, chunk: xr.Dataset, *, num_bins: int, variables: list[str]
) -> tuple[xbeam.Key, xr.Dataset]:
  """Evaluate spatial radial psd for a chunk of data."""
  assert len(chunk.time.values) == 1, (
      'Chunks are supposed to have one time slice but got'
      f' {len(chunk.time.values)}'
  )

  new_key = key.with_offsets(
      longitude=None, latitude=None, member=None, rad_freq=0
  )
  freqs, bins, k, norm, window = get_constants(chunk, num_bins)
  indices = np.digitize(k, bins)
  psd_mean = {}
  for var in variables:
    data = (
        chunk[var]
        .to_numpy()
        .reshape(-1, len(chunk.longitude), len(chunk.latitude))
    )
    psd_mean[var] = 0
    for sample in data:
      energy = np.abs(np.fft.fftshift(np.fft.fft2(sample * window))) ** 2
      energy = energy / norm
      psd = np.zeros_like(freqs)
      for i, b in enumerate(np.arange(1, num_bins + 1)):
        psd[i] = np.sum(energy * (indices == b))
      psd_mean[var] += psd
    psd_mean[var] /= len(data)

  new_chunk = xr.Dataset(
      data_vars={
          var: (('time', 'rad_freq'), np.expand_dims(psd_mean[var], axis=0))
          for var in variables
      },
      coords={'time': chunk.coords['time'].values, 'rad_freq': freqs},
  )
  return new_key, new_chunk


def average_psd_chunk(
    key: xbeam.Key, chunk: xr.Dataset
) -> tuple[xbeam.Key, xr.Dataset]:
  """Average psd over all chunks."""
  key = key.with_offsets(time=None)
  return key, chunk.mean(dim='time', skipna=True)


def get_constants(
    ds: xr.Dataset, num_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
  """Get radial frequency for a dataset."""
  lon, lat = ds['longitude'].to_numpy(), ds['latitude'].to_numpy()
  # (dx, dy) are in kilomemters, where we assume a spherical Earth with radius
  # 6378km, which leads to 111 km/deg.
  dx, dy = (lon[1] - lon[0]) * 111, (lat[1] - lat[0]) * 111
  nx, ny = len(lon), len(lat)
  freqs_x = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
  freqs_y = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
  kx, ky = np.meshgrid(freqs_x, freqs_y, indexing='ij')
  k = np.sqrt(kx**2 + ky**2)
  bins = np.linspace(np.min(k), np.max(k), num=num_bins + 1)
  freqs = (bins[:-1] + bins[1:]) / 2
  window = np.outer(np.hanning(nx), np.hanning(ny))
  return freqs, bins, k, nx * ny * dx * dy, window


def main(argv: list[str]) -> None:

  inference_ds, input_chunks = xbeam.open_zarr(INFERENCE_PATH.value)

  for var in SAMPLE_VARIABLES:
    if var.rename not in inference_ds:
      raise ValueError(
          f'Variable {var.rename} expected but not found in inference dataset.'
      )

  # If a reference path is provided, the script will compute psd on
  # the reference dataset instead of the inference dataset.
  if REFERENCE_PATH.value is not None:
    inference_ds = eval_utils.get_reference_ds(
        REFERENCE_PATH.value, SAMPLE_VARIABLES, inference_ds
    )
    del input_chunks
    input_chunks = {'longitude': -1, 'latitude': -1, 'time': 48, 'member': -1}

  years = list(range(YEAR_START.value, YEAR_END.value + 1))
  months = [int(m) for m in MONTHS.value]
  inference_ds = eval_utils.select_time(inference_ds, years, months)

  if HOUR_OF_DAY.value is not None:
    hour_mask = inference_ds.time.dt.hour.isin([HOUR_OF_DAY.value])
    inference_ds = inference_ds.sel(time=hour_mask, drop=True)

  variables = [v.rename for v in SAMPLE_VARIABLES]
  variables += ['RH', 'HI']

  # Define template
  rad_freq, *_ = get_constants(inference_ds, NUM_BINS.value)
  template_ds = xr.Dataset(
      data_vars={
          v: (('rad_freq',), np.empty((len(rad_freq),))) for v in variables
      },
      coords={'rad_freq': rad_freq},
  )
  template = xbeam.make_template(template_ds)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Read
    _ = (
        root
        | xbeam.DatasetToChunks(inference_ds, input_chunks, split_vars=False)
        | 'RechunkIn'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            inference_ds.sizes,
            input_chunks,
            {'longitude': -1, 'latitude': -1, 'member': -1, 'time': 1},
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | 'EvaluateSpatialPSD'
        >> beam.MapTuple(
            functools.partial(
                eval_spatial_radial_psd_chunk,
                num_bins=NUM_BINS.value,
                variables=variables,
            )
        )
        | 'RechunkOut'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            {'time': inference_ds.sizes['time'], 'rad_freq': len(rad_freq)},
            {'time': 1, 'rad_freq': -1},
            {'time': -1, 'rad_freq': -1},
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | 'AveragePSD' >> beam.MapTuple(average_psd_chunk)
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=template,
            zarr_chunks={'rad_freq': -1},
        )
    )


if __name__ == '__main__':
  app.run(main)
