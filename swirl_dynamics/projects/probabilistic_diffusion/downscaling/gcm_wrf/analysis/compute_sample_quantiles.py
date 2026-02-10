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

r"""Script to compute time and sample quantiles of climate data.

The script computes the quantiles of a given zarr dataset over all of its
non-spatial dimensions. For typical climate datasets that have dimensions
(time, *extra_dims, *spatial_dims), quantiles are computed over
(time, *extra_dims) after flattening these dimensions. The script also supports
quantile computation over multiple bootstrap samples (i.e., multiple resamples
of the data) to estimate the confidence interval of the quantiles.

Quantiles are computed using the `xr.Dataset.quantile` function. The quantiles
are computed over spatial chunks separately, and then consolidated into a single
dataset.


Example usage:

```
INFERENCE_PATH=<inference_zarr_path>
OUTPUT_PATH=<output_zarr_path>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/compute_sample_quantiles.py \
  --inference_path=${INFERENCE_PATH} \
  --output_path=${OUTPUT_PATH} \
  --bootstrap_samples=100 \
  --quantiles=0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99
```

"""

import functools

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import flag_utils
import xarray as xr
import xarray_beam as xbeam


# Command line arguments
QUANTILES = flags.DEFINE_list('quantiles', [], 'List of quantiles to compute.')
INFERENCE_PATH = flags.DEFINE_string(
    'inference_path', None, help='Input Zarr path'
)
BOOTSTRAP_SAMPLES = flags.DEFINE_integer(
    'bootstrap_samples',
    100,
    help='Number of bootstrap samples generated for each requested quantile.',
)
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
SPATIAL_DIMS = flags.DEFINE_list(
    'spatial_dims',
    ['south_north', 'west_east'],
    help='Spatial dimensions of the input dataset.',
)


def _get_sampling_indices(
    dataset: xr.Dataset,
    dim: str,
    samples: int,
    select_dim_items: int,
    dim_coord: xr.DataArray,
    replace: bool = True,
) -> xr.DataArray:
  """Generates random indices to sample along a dimension.

  Args:
    dataset: The dataset for which selection indices are generated.
    dim: The dimension to select from.
    samples: Number of bootstrap samples.
    select_dim_items: Number of items to select along `dim`. This is the size of
      each bootstrap sample.
    dim_coord: The coordinate of `dim`.
    replace: Whether to sample with replacement.

  Returns:
    An xr.DataArray with selection indices.
  """
  if replace:
    idx = np.random.randint(0, dataset[dim].size, (samples, select_dim_items))
  else:
    # create ordered `samples` of size `select_dim_items`
    idx = np.linspace(
        (np.arange(select_dim_items)),
        (np.arange(select_dim_items)),
        samples,
        dtype='int',
    )
    # shuffle each sample
    for ndx in np.arange(samples):
      np.random.shuffle(idx[ndx])
  idx_da = xr.DataArray(
      idx,
      dims=('bootstrap_sample', dim),
      coords=({'bootstrap_sample': range(samples), dim: dim_coord}),
  )
  return idx_da


def resample_dataset(
    dataset: xr.Dataset,
    num_samples: int,
    dim: str = 'time',
    dim_max: int | None = None,
    replace: bool = True,
) -> xr.Dataset:
  """Resamples a dataset over `dim` `num_samples` times.

  Args:
    dataset: The dataset to resample.
    num_samples: Number of samples to generate.
    dim: Dimension name to resample over. Defaults to `time`.
    dim_max: Size of each sample, defaults to `dataset[dim].size`.
    replace: Whether to sample with replacement.

  Returns:
    The dataset resampled along dimension ``dim``, with the additional
      dimension `bootstrap_sample`.
  """
  if dim not in dataset.coords:
    dataset.coords[dim] = np.arange(0, dataset[dim].size)
    dim_coord_set = True
  else:
    dim_coord_set = False

  select_dim_items = dataset[dim].size
  new_dim = dataset[dim]

  def select_bootstrap_indices_ufunc(x, idx):
    """Selects multi-level indices ``idx`` for all num_samples."""
    # `apply_ufunc` sometimes adds a singleton dimension on the end, so we
    # squeeze it out here. This leverages multi-level indexing from numpy,
    # so we can select a different set of, e.g., ensemble members for each
    # iteration and construct one large DataArray with ``num_samples``
    # as a dimension.
    return np.moveaxis(x.squeeze()[idx.squeeze().transpose()], 0, -1)

  # generate random indices to select from
  idx_da = _get_sampling_indices(
      dataset,
      dim,
      num_samples,
      select_dim_items,
      new_dim,
      replace=replace,
  )
  # bug fix if singleton dimension
  singleton_dims = []
  for d in dataset.dims:
    if dataset.sizes[d] == 1:
      singleton_dims.append(d)
      dataset = dataset.isel({d: [0] * 2})

  dataset = dataset.transpose(dim, ...)
  # If num_samples == 1, we don't resample
  if num_samples == 1:
    dataset_smp = dataset
  else:
    dataset_smp = xr.apply_ufunc(
        select_bootstrap_indices_ufunc,
        dataset,
        idx_da,
        output_dtypes=[float],
    )
  # return only dim_max-sized samples
  if dim_max is not None and dim_max <= dataset[dim].size:
    dataset_smp = dataset_smp.isel({dim: slice(None, dim_max)})
  if dim_coord_set:
    del dataset_smp.coords[dim]
  for d in singleton_dims:
    dataset_smp = dataset_smp.isel({d: [0]})
  return dataset_smp


def evaluate_chunk_quantile(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    agg_dims: tuple[str, ...],
    q: list[float],
    bootstrap_dim: str = 'time',
    bootstrap_samples: int = 100,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Evaluate quantiles of a chunk over specified dimensions."""

  resampled_chunk = resample_dataset(
      obs_chunk, bootstrap_samples, dim=bootstrap_dim
  )
  offsets = {agg_dim: None for agg_dim in agg_dims}
  offsets['quantile'] = 0
  ppf_ds = resampled_chunk.quantile(q, dim=agg_dims)

  if bootstrap_samples > 1:
    transpose_dims = ('bootstrap_sample', 'quantile')
    offsets['bootstrap_sample'] = 0
  else:
    transpose_dims = ('quantile',)

  ppf_ds = ppf_ds.transpose(*transpose_dims, ...)
  ppf_key = obs_key.with_offsets(**offsets)
  return ppf_key, ppf_ds


def main(argv: list[str]) -> None:
  bootstrap_samples = BOOTSTRAP_SAMPLES.value
  quantiles = [float(q) for q in QUANTILES.value]
  inf, input_chunks = xbeam.open_zarr(INFERENCE_PATH.value)
  spatial_dims = tuple(SPATIAL_DIMS.value)

  # Chunks for temporal and sample reduce
  in_working_chunks = {
      k: -1
      for k in input_chunks.keys()
      if k not in spatial_dims
  }
  agg_dims = tuple(in_working_chunks.keys())
  # Update with spatial chunks
  in_working_chunks.update(WORKING_CHUNKS.value)

  # Chunks for ppf dataset
  out_working_chunks = {
      k: v for k, v in in_working_chunks.items() if k not in agg_dims
  }
  out_working_chunks['quantile'] = -1
  out_working_sizes = {
      k: inf.sizes[k] for k in out_working_chunks.keys() if k in inf.dims
  }
  out_working_sizes['quantile'] = len(quantiles)
  if bootstrap_samples > 1:
    out_working_chunks['bootstrap_sample'] = -1
    out_working_sizes['bootstrap_sample'] = bootstrap_samples

  out_chunks = {
      k: input_chunks[k] for k in out_working_chunks.keys() if k in inf.dims
  }
  out_chunks['quantile'] = -1
  if bootstrap_samples > 1:
    out_chunks['bootstrap_sample'] = 1

  # Define template
  if bootstrap_samples > 1:
    transpose_dims = ('bootstrap_sample', 'quantile')
    expand_dims = dict(
        quantile=quantiles, bootstrap_sample=np.arange(bootstrap_samples)
    )
  else:
    transpose_dims = ('quantile',)
    expand_dims = dict(quantile=quantiles)
  template = (
      xbeam.make_template(inf)
      .isel(**{agg_dim: 0 for agg_dim in agg_dims}, drop=True)
      .expand_dims(**expand_dims)
      .transpose(*transpose_dims, ...)
  )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Read
    _ = (
        root
        | xbeam.DatasetToChunks(inf, input_chunks, split_vars=True)
        | 'RechunkIn'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            inf.sizes,
            input_chunks,
            in_working_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | 'EvaluateQuantiles'
        >> beam.MapTuple(
            functools.partial(
                evaluate_chunk_quantile,
                agg_dims=agg_dims,
                q=quantiles,
                bootstrap_samples=bootstrap_samples,
            )
        )
        | 'RechunkOut'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            out_working_sizes,
            out_working_chunks,
            out_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=template,
            zarr_chunks=out_chunks,
        )
    )


if __name__ == '__main__':
  app.run(main)
