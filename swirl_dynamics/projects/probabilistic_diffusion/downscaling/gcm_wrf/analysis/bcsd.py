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

r"""Script to perform downscaling with the time-aligned BCSD method.

This script implements the Bias Correction and Spatial Disaggregation (BCSD)
method for downscaling (Wood et al, 2002): https://doi.org/10.1029/2001JD000659.
This implementation follows the time-aligned BCSD method of Thrasher et al (2012):
https://doi.org/10.5194/hess-16-3309-2012. The methodology is modified to
operate on a single domain discretization (the high-resolution grid).

The method assumes we have a low-resolution dataset, its climatology, and the
spatially filtered and unfiltered climatology of a high-resolution dataset. The
input low-resolution dataset and all climatologies are assumed to have been
interpolated to the high-resolution grid for convenience. Nevertheless, the
low-resolution and filtered high-resolution fields are assumed to have a low
effective --and similar-- spatial resolution, since interpolation does not yield
energy at the interpolated lengthscales.

The bias correction step is performed by computing the percentiles of the input
dataset with respect to its climatology, and mapping them to the full field
values corresponding to the same percentiles in the filtered high-resolution
climatology.

The spatial disaggregation consists, for most variables, of subtracting the
climatological mean of the filtered high-resolution field from the
bias-corrected low-resolution field, and then adding the climatological mean of
the unfiltered high-resolution field. For some variables, such as precipitation,
we use a multiplicative correction instead: we multiply the bias-corrected
low-resolution field by the ratio of the filtered and unfiltered climatological
means. The variables to be corrected multiplicatively are specified by the
`multiplicative_vars` flag.

Although the temporal disaggregation step of Wood et al. (2002) is omitted in
our implementation (Thrasher et al., 2012), we describe it here for completeness.
Let the low-resolution data correspond to time averages over periods `T`. In the
temporal disaggregation step, for each sample resulting from the spatial
disaggregation step, we sample a high-resolution data sequence over period `T`
and the same time of the year as the spatially
disaggregated sample. The high--resolution sample sequence is drawn at random
from the dataset used to construct the high-resolution climatology. Once the
sample sequence is drawn, we scale it such that the time average over `T`
matches the spatially disaggregated sample. Finally, we use this scaled sequence
as the final BCSD sample.

Example usage:

```
# Path to input dataset climatology.
INPUT_DATA_CLIM=<input_data_clim>

# Path to spatially filtered target climatology.
FILTERED_TARGET_CLIM=<filtered_target_clim>

# Path to unfiltered target climatology.
TARGET_CLIM=<target_clim>

# Path to dataset to be downscaled.
INPUT_DATA=<parent_dir>/low_resolution_dataset.zarr

# Time range to be downscaled.
TIME_START=2094
TIME_STOP=2097

# Path to the output dataset.
OUTPUT_PATH=<parent_dir>/bcsd_downscaled_dataset_${TIME_START}_${TIME_STOP}.zarr

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/bcsd.py \
  --input_data=${INPUT_DATA} \
  --input_data_clim=${INPUT_DATA_CLIM} \
  --filtered_target_clim=${FILTERED_TARGET_CLIM} \
  --target_clim=${TARGET_CLIM} \
  --time_start=${TIME_START} \
  --time_stop=${TIME_STOP} \
  --output_path=${OUTPUT_PATH} \
  --multiplicative_vars=RAIN_24h
```

"""

import typing

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray as xr
import xarray_beam as xbeam


INPUT_DATA = flags.DEFINE_string(
    'input_data',
    None,
    help='Zarr path pointing to the input data to be processed.',
)
INPUT_DATA_CLIM = flags.DEFINE_string(
    'input_data_clim',
    None,
    help='Zarr path pointing to the climatology of the input data.',
)
FILTERED_TARGET_CLIM = flags.DEFINE_string(
    'filtered_target_clim',
    None,
    help=(
        'Zarr path pointing to the climatology of the spatially filtered target'
        ' data.'
    ),
)
TARGET_CLIM = flags.DEFINE_string(
    'target_clim',
    None,
    help='Zarr path pointing to the climatology of the unfiltered target data.',
)
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
METHOD = flags.DEFINE_enum(
    'method',
    'gaussian',
    ['gaussian', 'quantile_interpolation'],
    'Method to use for quantile mapping.',
)
MULTIPLICATIVE_VARS = flags.DEFINE_list(
    'multiplicative_vars',
    [],
    'List of variables to process using multiplicative correction.',
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
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _get_climatology_mean(
    climatology: xr.Dataset, variables: list[str], **sel_kwargs
) -> xr.Dataset:
  """Returns the climatological mean of the given variables.

  The climatology dataset is assumed to have been produced through
  the weatherbench2 compute_climatology.py script, and statistics
  `mean`, and `std`. The convention is that the climatological means do not
  have a suffix, and standard deviations have a `_std` suffix.

  Args:
    climatology: The climatology dataset.
    variables: The variables to extract from the climatology.
    **sel_kwargs: Additional selection criteria for the variables.

  Returns:
    The climatological mean of the given variables.
  """
  # try:
  climatology_mean = climatology[variables]
  return typing.cast(xr.Dataset, climatology_mean.sel(**sel_kwargs).compute())


def _get_climatology_std(
    climatology: xr.Dataset, variables: list[str], **sel_kwargs
) -> xr.Dataset:
  """Returns the climatological standard deviation of the given variables.

  The climatology dataset is assumed to have been produced through
  the weatherbench2 compute_climatology.py script, and statistics
  `mean`, and `std`. The convention is that the climatological means do not
  have a suffix, and standard deviations have a `_std` suffix.

  Args:
    climatology: The climatology dataset.
    variables: The variables to extract from the climatology.
    **sel_kwargs: Additional selection criteria for the variables.

  Returns:
    The climatological standard deviation of the given variables.
  """
  clim_std_dict = {key + '_std': key for key in variables}  # pytype: disable=unsupported-operands
  # try:
  climatology_std = climatology[list(clim_std_dict.keys())].rename(
      clim_std_dict
  )
  return typing.cast(xr.Dataset, climatology_std.sel(**sel_kwargs).compute())


def _bcsd_on_chunks(
    source: xr.Dataset,
    original_clim: xr.Dataset,
    filtered_clim: xr.Dataset,
    target_clim: xr.Dataset,
    method: str = 'gaussian',
) -> xr.Dataset:
  """Process an input data chunk with the BCSD method.

  Args:
    source: The source data chunk to be processed with the BCSD method.
    original_clim: The climatology of the source data.
    filtered_clim: The climatology of the spatially filtered target data.
    target_clim: The climatology of the unfiltered target data.
    method: The method to use for quantile mapping.

  Returns:
    The combined chunks as an xarray Dataset.
  """
  sel = dict(
      dayofyear=source['time'].dt.dayofyear,
      hour=source['time'].dt.hour,
      drop=True,
  )
  variables = [str(key) for key in source.keys()]
  if method == 'gaussian':
    clim_mean = _get_climatology_mean(original_clim, variables, **sel)
    clim_std = _get_climatology_std(original_clim, variables, **sel)

    # Standardize with respect to the original climatology.
    source_standard = (source - clim_mean) / clim_std

    # Get value of the same quantile in the filtered climatology, keep anom.
    filtered_clim_std = _get_climatology_std(filtered_clim, variables, **sel)
    source_bc_anom = source_standard * filtered_clim_std

    # Add anom to the mean of the unfiltered climatology.
    target_clim_mean = _get_climatology_mean(target_clim, variables, **sel)
    source_bcsd = target_clim_mean + source_bc_anom

    # Use multiplicative correction for precipitation variables
    if MULTIPLICATIVE_VARS.value:
      filtered_clim_mean = _get_climatology_mean(
          filtered_clim, MULTIPLICATIVE_VARS.value, **sel
      )
      for var in MULTIPLICATIVE_VARS.value:
        # Compute the full field corresponding to the filtered climatology
        # quantile. Then, keep the multiplicative anomaly.
        bc_mult_anom = (source_bc_anom[var] + filtered_clim_mean[var]) / (
            filtered_clim_mean[var]
        )
        # Multiply the anomaly by the target climatology.
        var_bcsd = bc_mult_anom * target_clim_mean[var]
        source_bcsd = source_bcsd.drop_vars(var).assign(**{var: var_bcsd})

    return source_bcsd.drop_vars(['hour', 'dayofyear'])
  else:
    raise ValueError(f'BCSD method {method} not yet implemented.')


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  if TIME_START.value is None or TIME_STOP.value is None:
    return ds
  selection = {TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)}
  return ds.sel({k: v for k, v in selection.items() if k in ds.dims})


def main(argv: list[str]) -> None:

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_DATA.value)
  source_dataset = _impose_data_selection(source_dataset)

  original_clim = xr.open_zarr(INPUT_DATA_CLIM.value)
  filtered_clim = xr.open_zarr(FILTERED_TARGET_CLIM.value)
  target_clim = xr.open_zarr(TARGET_CLIM.value)

  source_chunks = {k: source_chunks[k] for k in source_chunks}
  in_working_chunks = source_chunks.copy()
  in_working_chunks['time'] = 1

  output_chunks = source_chunks.copy()
  output_chunks['time'] = TIME_CHUNK_SIZE.value
  unassigned_coords = {
      dim: np.arange(source_dataset.sizes[dim])
      for dim in source_dataset.dims
      if dim not in source_dataset.coords
  }
  template = xbeam.make_template(source_dataset).assign_coords(
      **unassigned_coords
  )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(source_dataset, in_working_chunks)
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _bcsd_on_chunks(
                    v,
                    original_clim,
                    filtered_clim,
                    target_clim,
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
