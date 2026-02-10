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

r"""Script to compute the upper tail dependence of two variables.

The upper tail dependence is defined following eq. 13 and 16 of Schmidt and
Stadtmuller (2006; https://doi.org/10.1111/j.1467-9469.2005.00483.x). The
upper tail dependence coefficient is defined in terms of a parameter k, which
defines the sample rank threshold above which the dependence is computed.
Given a sample size m, k corresponds to the quantile q = (1 - k / m) of the
marginal empirical distributions of the two variables. Therefore, k and q are
related as follows:

                        q * m = m - k -> k = m * (1 - q)

As shown by Schmidt and Stadtmuller, the coefficient can be estimated over a
range of values of k, for which it remains relatively constant (their Fig. 5).
This script computes the upper tail dependence for all k values within a
specified empirical quantile range [q1, q2]. The asymptotic limit for the
underlying tail dependence holds at the limit of m -> infinity, k -> infinity,
with k/m -> 0.

Given a dataset with variables of shape (sample, time, *spatial_dims), the
script computes the upper tail dependence over the flattened time and sample
dimensions. That is, samples and times are considered i.i.d. The output is a
dataset with shape (quantile, ...), where quantile is the quantile of the sample
rank threshold above which the dependence is computed.

The script also implements a `sign_change` option for the second variable, which
is useful for computing the tail dependence of extrema of different signs in the
chosen variables. For instance, one may want to compute the tail dependence of
high temperature and low specific humidity extremes.

Example usage:

```
INFERENCE_PATH=<inference_zarr_path>
OUTPUT_PATH=<output_zarr_path>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/compute_tail_dependence.py \
  --inference_path=${INFERENCE_PATH} \
  --output_path=${OUTPUT_PATH} \
  --quantile_range=0.9,0.95 \
  --variables=T2,Q2
```

"""

import functools

from absl import app
from absl import flags
import apache_beam as beam
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import flag_utils
import xarray as xr
import xarray_beam as xbeam

_DEFAULT_QUANTILE_RANGE = ['0.9', '0.95']

# Command line arguments
QUANTILE_RANGE = flags.DEFINE_list(
    'quantile_range',
    _DEFAULT_QUANTILE_RANGE,
    help=(
        'Quantiles between which to compute tail dependence at all meaningful'
        ' granularities.'
    ),
)
VARIABLES = flags.DEFINE_list(
    'variables',
    ['T2', 'Q2'],
    help='Variables to compute tail dependence on.',
)
SIGN_CHANGE = flags.DEFINE_bool(
    'sign_change',
    False,
    help=(
        'Whether to change the sign of the second variable before computing'
        ' tail dependence.'
    ),
)
INFERENCE_PATH = flags.DEFINE_string(
    'inference_path', None, help='Input Zarr path.'
)
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path.')
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


def eval_upper_tail_dependence_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    variables: tuple[str, str],
    agg_dims: tuple[str, ...],
    k_lims: tuple[float, float],
    nan_policy: str = 'omit',
    sign_change: bool = False,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Evaluate upper tail dependence for two variables over a chunk of data.

  The upper tail dependence is defined following eq. 13 and 16 of Schmidt and
  Stadtmuller (2006; https://doi.org/10.1111/j.1467-9469.2005.00483.x).

  Args:
    obs_key: They key to the data chunk.
    obs_chunk: The data chunk.
    variables: The variables to compute tail dependence on.
    agg_dims: The dimensions to aggregate samples, rank, and compute the tail
      dependence over.
    k_lims: The limits of the k values to compute tail dependence at.
    nan_policy: The policy to apply to NaN values when ranking the samples,
      passed to `scipy.stats.rankdata`. The default is 'omit', which removes
      NaNs from the ranking.
    sign_change: Whether to change the sign of the second variable before
      computing tail dependence.

  Returns:
    The key and chunk of the upper tail dependence dataset, which has variable
    name `variables[0] + '_' + variables[1] + '_tail_dependence'` with shape
    (quantile, *spatial_dims).
  """
  offsets = {agg_dim: None for agg_dim in agg_dims}
  offsets['quantile'] = 0
  retained_dims = tuple(set(obs_chunk.dims) - set(agg_dims))
  retained_coords = {dim: obs_chunk.coords[dim].values for dim in retained_dims}
  # Stacked dimensions along last axis.
  var1_chunk = obs_chunk[variables[0]].stack({'samples': agg_dims}).values
  var2_chunk = obs_chunk[variables[1]].stack({'samples': agg_dims}).values
  if sign_change:
    var2_chunk = -var2_chunk

  sample_size = var1_chunk.shape[-1]
  ks = jnp.arange(k_lims[0], k_lims[1], -1)

  rank_var1 = stats.rankdata(
      var1_chunk, method='min', axis=-1, nan_policy=nan_policy
  )
  rank_var2 = stats.rankdata(
      var2_chunk, method='min', axis=-1, nan_policy=nan_policy
  )

  def _upper_tail_dependence(k):
    cond1 = jnp.where(rank_var1 > sample_size - k, 1, 0)
    cond2 = jnp.where(rank_var2 > sample_size - k, 1, 0)
    return (1.0 / k) * jnp.sum(cond1 * cond2, axis=-1)

  # Output shape: (k, *spatial_dims). Less memory intensive than jax.vmap.
  tdcs = jax.lax.map(_upper_tail_dependence, ks)

  if sign_change:
    tdc_name = f'{variables[0]}_m{variables[1]}_tail_dependence'
  else:
    tdc_name = f'{variables[0]}_{variables[1]}_tail_dependence'

  tdc_chunk = xr.Dataset(
      data_vars={tdc_name: (['quantile', *retained_dims], tdcs)},
      coords=dict(quantile=1 - ks / float(sample_size), **retained_coords),
  )

  tdc_key = obs_key.with_offsets(**offsets)
  return tdc_key, tdc_chunk


def main(argv: list[str]) -> None:

  quantile_range = tuple([float(q) for q in QUANTILE_RANGE.value])
  if len(quantile_range) != 2:
    raise ValueError(
        f'Quantile range must have length 2, but got {quantile_range}.'
    )
  variables = tuple(VARIABLES.value)
  sign_change = SIGN_CHANGE.value
  spatial_dims = tuple(SPATIAL_DIMS.value)

  inference_ds, input_chunks = xbeam.open_zarr(INFERENCE_PATH.value)
  retained_coords = {dim: inference_ds[dim].values for dim in spatial_dims}

  # Chunks for temporal and sample reduce
  in_working_chunks = {
      k: -1 for k in input_chunks.keys() if k not in spatial_dims
  }
  agg_dims = tuple(in_working_chunks.keys())
  # Update with spatial chunks
  in_working_chunks.update(WORKING_CHUNKS.value)

  sample_size = np.prod([inference_ds.sizes[dim] for dim in agg_dims])
  k_lims = [int(sample_size * (1 - q)) for q in quantile_range]
  quantiles = 1 - jnp.arange(k_lims[0], k_lims[1], -1) / float(sample_size)

  # Chunks for tdc dataset
  out_working_chunks = {
      k: v for k, v in in_working_chunks.items() if k not in agg_dims
  }
  out_working_chunks['quantile'] = -1
  out_working_sizes = {
      k: inference_ds.sizes[k]
      for k in out_working_chunks.keys()
      if k in inference_ds.dims
  }
  out_working_sizes['quantile'] = len(quantiles)

  out_chunks = {
      k: input_chunks[k]
      for k in out_working_chunks.keys()
      if k in inference_ds.dims
  }
  out_chunks['quantile'] = -1

  # Define template
  if sign_change:
    tdc_name = f'{variables[0]}_m{variables[1]}_tail_dependence'
  else:
    tdc_name = f'{variables[0]}_{variables[1]}_tail_dependence'

  tdc_template = np.empty(
      (
          len(quantiles),
          inference_ds.sizes[spatial_dims[0]],
          inference_ds.sizes[spatial_dims[1]],
      ),
  )
  template_ds = xr.Dataset(
      data_vars={tdc_name: (['quantile', *spatial_dims], tdc_template)},
      coords=dict(quantile=quantiles, **retained_coords),
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
            in_working_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | 'EvaluateTailDependence'
        >> beam.MapTuple(
            functools.partial(
                eval_upper_tail_dependence_chunk,
                agg_dims=agg_dims,
                k_lims=k_lims,
                variables=variables,
                sign_change=sign_change,
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
