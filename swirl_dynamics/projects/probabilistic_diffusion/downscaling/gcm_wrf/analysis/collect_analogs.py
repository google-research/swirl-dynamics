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

r"""Script to collect weather analogs from a training dataset.

This script collects regional weather analogs of input samples from a training
dataset, using a mask dataset that defines a set of regions around reference
points. The regional analog collection procedure is thoroughly described in
Sections 3 and the Appendix of the LOCA downscaling description paper
(Pierce et al, 2014; https://doi.org/10.1175/JHM-D-14-0082.1). The procedure is
briefly summarized here, but please refer to the paper for a more detailed
account.

Consider an input sample representing a weather state over a domain in space.
The objective of this script is to find the most similar weather states to the
input sample in a given training dataset. The similarity between the input
sample and each training sample is defined by considering their similarities in
a user-defined atmospheric field, for instance the surface zonal velocity. This
field is denoted the analog variable. The similarity in the analog variables of
both samples is defined in terms of the anomaly of this analog variable with
respect to the climatology. Therefore, we first subtract the climatological mean
of the input dataset from the input sample, and the climatological mean of the
training dataset from each training sample. We then compute the weighted mean
squared error (MSE) between the analog variable anomalies in each state as the
similarity metric. A lower weighted MSE indicates more similar states.

In the computation of this weighted MSE, the weights are given by a binary
spatial mask that regionalizes the computation. In this script, we follow the
LOCA method (Pierce et al, 2014; https://doi.org/10.1175/JHM-D-14-0082.1), and
assume the binary mask defines the grid points where the analog variable has a
positive spatial correlation with respect to a reference point. As distance from
the reference point increases, spatial correlations decrease and the mask
becomes zero, thus regionalizing the computation.

We consider a set of reference points, and compute the weighted MSE in the
analog variable for the mask defined by each of the reference points. Let
`i` index over each of the reference points that define each of the masks. Then
we compute the weighted MSEs between the input sample and each training sample
as:

    analog_error_i = sum_j (mask_ij * (input_sample_analog_var -
      training_sample_analog_var) ** 2) / sum_j (1)

where `j` indexes the grid locations over which the data are defined,
`input_sample_analog_var` is the anomaly in the analog variable of the input
sample, and `training_sample_analog_var` is the anomaly in the analog variable
of the training sample. The set of analog errors `analog_error_i` are computed
between the input sample and each potential analog in the training dataset.
The potential analogs are the samples in the training dataset that have the
same hour of the day as the input sample, and are within a window in days of the
year of the input sample. For instance, if the input sample is from January 1st,
and we consider a 30 day window, then the potential analogs are all training
dataset samples from December 17th to January 15th. This includes all years
in the training dataset.

Finally, when the analog error has been computed for all potential analogs, we
retain the `num_analogs` dates of samples in the training dataset with the
lowest analog error, for each input sample. The analog error is retained in the
output dataset, along with the time of the analog. This enables mixing and
matching analogs from different training datasets a posteriori, by comparing
the analog errors. For LOCA, `num_analogs` is set to 30.

A note on the climatologies used tom compute the anomalies of the input and
training samples. In order to capture potential shifts in climatology,
the input sample anomalies can be computed by subtracting the climatological
mean over a period of time close to the test period (e.g., the 2070-2100
climatology when downscaling samples from 2094). The anomalies of the training
set are always computed with respect to a static climatology over the
training period.

Example usage:

```
TIME_START=2094
TIME_STOP=2097

FORCING_DATASET=<PATH_TO_FORCING_DATASET>
ANALOG_VARIABLE=U10

TRAINING_START_YEAR=2020
TRAINING_END_YEAR=2040

INPUT_BASENAME=hourly_d01_cubic_interpolated_to_d02_with_prates
MASK_PATH=<PATH_TO_MASK_DATASET>
TRAINING_DATASET_PATH=<PATH_TO_TRAINING_DATASET>
INPUT_CLIMATOLOGY_PATH=<PATH_TO_INPUT_CLIMATOLOGY_DATASET>
TRAINING_CLIMATOLOGY_PATH=<PATH_TO_TRAINING_CLIMATOLOGY_DATASET>

INPUT_PATH=${FORCING_DATASET}/${INPUT_BASENAME}.zarr
OUTPUT_PATH=${FORCING_DATASET}/analogs/${ANALOG_VARIABLE}_analogs_${TIME_START}_${TIME_STOP}_${INPUT_BASENAME}_${TRAINING_START_YEAR}_${TRAINING_END_YEAR}.zarr

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/collect_analogs.py \
  --input_path=${INPUT_PATH} \
  --output_path=${OUTPUT_PATH} \
  --time_start=${TIME_START} \
  --time_stop=${TIME_STOP} \
  --training_start_year=${TRAINING_START_YEAR} \
  --training_end_year=${TRAINING_END_YEAR} \
  --training_dataset_path=${TRAINING_DATASET_PATH} \
  --input_climatology_path=${INPUT_CLIMATOLOGY_PATH} \
  --training_climatology_path=${TRAINING_CLIMATOLOGY_PATH} \
  --mask_path=${MASK_PATH} \
  --analog_variable=${ANALOG_VARIABLE} \
  --variables=${ANALOG_VARIABLE} \
  --sampling_period=4
```

"""

import functools

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import beam_utils
import xarray as xr
import xarray_beam as xbeam


_DEFAULT_VARIABLES = ['U10']

# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
TRAINING_DATASET_PATH = flags.DEFINE_string(
    'training_dataset_path',
    None,
    help='Zarr path to dataset probed for analog selection.',
)
MASK_PATH = flags.DEFINE_string(
    'mask_path',
    None,
    help='Zarr path to dataset used for masking analog comparison operations.',
)
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
INPUT_CLIMATOLOGY_PATH = flags.DEFINE_string(
    'input_climatology_path',
    None,
    help='Zarr path pointing to the climatology of the input dataset.',
)
TRAINING_CLIMATOLOGY_PATH = flags.DEFINE_string(
    'training_climatology_path',
    None,
    help='Zarr path pointing to the climatology of the training dataset.',
)
TIME_START = flags.DEFINE_string(
    'time_start', None, help='Inclusive start year of input data considered.'
)
TIME_STOP = flags.DEFINE_string(
    'time_stop', None, help='Inclusive end year of input data considered.'
)
TRAINING_START_YEAR = flags.DEFINE_string(
    'training_start_year',
    None,
    help='Inclusive start year of training dataset to probe for analogs.',
)
TRAINING_END_YEAR = flags.DEFINE_string(
    'training_end_year',
    None,
    help='Inclusive end year of training dataset to probe for analogs.',
)
ANALOG_VARIABLE = flags.DEFINE_string(
    'analog_variable',
    'PSFC',
    help='Variables considered for analog selection.',
)
SAMPLING_PERIOD = flags.DEFINE_integer(
    'sampling_period',
    1,
    help='Period at which to sample the training dataset.',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    _DEFAULT_VARIABLES,
    help='Variables retained in the output dataset.',
)
_NUM_ANALOGS = flags.DEFINE_integer(
    'num_analogs',
    30,
    help='Number of analogs to collect per input sample.',
)
WINDOW = flags.DEFINE_integer(
    'window',
    60,
    help='Window in days used to select analogs.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize', 4, help='Itemsize for rechunking.'
)
SPATIAL_DIMS = flags.DEFINE_list(
    'spatial_dims',
    ['south_north', 'west_east'],
    help='Name of the spatial dimensions of the dataset.',
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)


def _impose_data_selection(ds: xr.Dataset) -> xr.Dataset:
  if TIME_START.value or TIME_STOP.value:
    selection = {'time': slice(TIME_START.value, TIME_STOP.value)}
    ds = ds.sel({k: v for k, v in selection.items() if k in ds.dims})
  if VARIABLES.value:
    ds = xr.Dataset(ds.get(VARIABLES.value))
  if SAMPLING_PERIOD.value > 1:
    ds = ds.isel(time=slice(None, None, SAMPLING_PERIOD.value))
  return ds


def _sel_doy_within_window(
    ds: xr.Dataset, input_doy: int, window: int
) -> xr.DataArray:
  """Selects a time window selection around the input day of year.

  Args:
    ds: Dataset to select over.
    input_doy: Doy of year about which to select dates.
    window: The size of the window used to select dates.

  Returns:
    A boolean array of the same length as the input dataset time coordinate,
    with True values indicating that the corresponding time point is within the
    window.
  """
  sel_left = input_doy - window // 2
  sel_right = input_doy + window // 2

  if sel_left > 0 and sel_right < 366:
    sel_doy = (ds.time.dt.dayofyear > sel_left) * (
        ds.time.dt.dayofyear < sel_right
    )
  elif sel_left < 0:
    sel_left = sel_left + 365
    sel_doy = (ds.time.dt.dayofyear > sel_left) + (
        ds.time.dt.dayofyear < sel_right
    )
  else:
    sel_right = sel_right - 365
    sel_doy = (ds.time.dt.dayofyear > sel_left) + (
        ds.time.dt.dayofyear < sel_right
    )
  return sel_doy


def collect_analogs_chunk(
    input_key: xbeam.Key,
    input_chunk: xr.Dataset,
    *,
    mask_ds: xr.Dataset,
    training_ds: xr.Dataset,
    input_clim_ds: xr.Dataset,
    training_clim_ds: xr.Dataset,
    spatial_dims: tuple[str, str] = ('south_north', 'west_east'),
    pool_id: int,
    num_analogs: int = 30,
    window: int = 60,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Collect weather analogs for a single input chunk from a training dataset.

  Args:
    input_key: The key indexing the input chunk.
    input_chunk: The dataset chunk.
    mask_ds: The dataset containing the masks used to select analogs for each
      reference point. The `ANALOG_VARIABLE` variable must be present.
    training_ds: The dataset used to probe for analogs.
    input_clim_ds: The climatology of the input dataset, used to subtract the
      climatological mean of the input samples.
    training_clim_ds: The climatology of the training dataset, used to subtract
      the climatological mean of the training samples.
    spatial_dims: The name of the two spatial dimensions of the dataset.
    pool_id: The index of the reference point pool.
    num_analogs: The number of analogs to collect per reference point.
    window: The size of the day of year window used to select analogs.

  Returns:
    The key and the detrended dataset chunk.
  """
  # Get reference point, analog variable, and mask, to define error.
  analog_var = ANALOG_VARIABLE.value
  reference_point = [mask_ds.reference_point[pool_id].item()]
  mask_da = mask_ds.isel(reference_point=pool_id, drop=True)[analog_var]

  # Get hour and day of year of the input for candidate selection.
  input_hour = input_chunk.time.dt.hour.item()
  input_doy = input_chunk.time.dt.dayofyear.item()
  input_clim_sel = dict(
      dayofyear=input_doy,
      hour=input_hour,
      drop=True,
  )
  input_clim_mean = beam_utils.get_climatology_mean(
      input_clim_ds, [analog_var], **input_clim_sel
  )
  # Squeeze out time dimension for broadcasting.
  input_squeezed = (
      (input_chunk - input_clim_mean)
      .squeeze(['time'], drop=True)[analog_var]
      .compute()
  )

  # Select pool of potential analogs for each year.
  sel_doy = _sel_doy_within_window(training_ds, input_doy, window)
  time_sel = (training_ds.time.dt.hour == input_hour) * sel_doy
  # Select climatology for the window around the input.
  clim_sel = dict(
      dayofyear=list(set(time_sel.sel(time=time_sel).time.dt.dayofyear.values)),
      hour=input_hour,
      drop=True,
  )
  clim_mean = beam_utils.get_climatology_mean(
      training_clim_ds, [analog_var], **clim_sel
  )[analog_var].compute()

  # Loop over training set years.
  analog_errors = np.ones(num_analogs) * np.inf
  analog_times = np.empty(
      num_analogs,
      dtype='datetime64[ns]',
  )
  for year in set(training_ds.time.dt.year.values):

    time_sel_year = time_sel * (training_ds.time.dt.year == year)
    candidate_da = training_ds.sel(time=time_sel_year)[analog_var]
    candidate_da = candidate_da.groupby('time.dayofyear') - clim_mean
    # Get weighted MSE.
    error_da = candidate_da - input_squeezed
    error_da = (error_da * error_da * mask_da).mean(dim=spatial_dims).compute()
    # Collect current analogs times with lowest error.
    current_analog_errors = error_da[error_da.argsort().values].values[
        :num_analogs
    ]
    current_analog_times = error_da.time[error_da.argsort().values].values[
        :num_analogs
    ]
    # Compare to previous analogs and retain closest to the input.
    analog_ids = np.concat([analog_errors, current_analog_errors]).argsort()
    analog_errors = np.concat([analog_errors, current_analog_errors])[
        analog_ids
    ][:num_analogs]
    analog_times = np.concat([analog_times, current_analog_times])[analog_ids][
        :num_analogs
    ]

  analog_chunk = xr.Dataset(
      data_vars=dict(
          analog_time=(['time', 'reference_point', 'analog'], [[analog_times]]),
          analog_mse=(['time', 'reference_point', 'analog'], [[analog_errors]]),
      ),
      coords=dict(
          time=input_chunk.time,
          reference_point=reference_point,
          analog=np.arange(num_analogs, dtype=np.int64),
      ),
  )

  analog_key = input_key.with_offsets(
      analog=0, reference_point=pool_id, south_north=None, west_east=None
  )
  return analog_key, analog_chunk


def main(argv: list[str]) -> None:
  variables = VARIABLES.value
  logging.info('Variables considered: %s', variables)
  analogs = np.arange(_NUM_ANALOGS.value)
  spatial_dims = tuple(SPATIAL_DIMS.value)
  analog_var = ANALOG_VARIABLE.value

  input_ds, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  input_ds = _impose_data_selection(input_ds)

  train_sel = {
      'time': slice(TRAINING_START_YEAR.value, TRAINING_END_YEAR.value)
  }
  training_ds = xr.Dataset(
      xr.open_zarr(TRAINING_DATASET_PATH.value).get(variables)
  ).sel(**train_sel)

  input_clim_ds = xr.Dataset(
      xr.open_zarr(INPUT_CLIMATOLOGY_PATH.value).get(variables)
  )
  training_clim_ds = xr.Dataset(
      xr.open_zarr(TRAINING_CLIMATOLOGY_PATH.value).get(variables)
  )

  # Precompute binarized analog scoring mask.
  mask_ds = xr.Dataset(xr.open_zarr(MASK_PATH.value).get([analog_var]))
  mask_ds = xr.where(mask_ds > 0.01, 1.0, 0.0)
  analog_pool_points = mask_ds.coords['reference_point']

  output_chunks = dict(time=1, reference_point=1, analog=len(analogs))
  template_ds = xr.Dataset(
      data_vars=dict(
          analog_time=(
              ['time', 'reference_point', 'analog'],
              np.empty(
                  (
                      len(input_ds.time),
                      len(analog_pool_points),
                      len(analogs),
                  ),
                  dtype='datetime64[ns]',
              ),
          ),
          analog_mse=(
              ['time', 'reference_point', 'analog'],
              np.empty(
                  (
                      len(input_ds.time),
                      len(analog_pool_points),
                      len(analogs),
                  ),
                  dtype='float32',
              ),
          ),
      ),
      coords=dict(
          time=input_ds.time,
          reference_point=analog_pool_points,
          analog=analogs,
      ),
  )
  # Convert object-type coordinates to string.
  # Required to avoid: https://github.com/pydata/xarray/issues/3476
  for coord_name, coord in template_ds.coords.items():
    if coord.dtype == 'object':
      template_ds[coord_name] = coord.astype(str)
  template = xbeam.make_template(template_ds)

  collect_kwargs = dict(
      mask_ds=mask_ds,
      training_ds=training_ds,
      input_clim_ds=input_clim_ds,
      training_clim_ds=training_clim_ds,
      spatial_dims=spatial_dims,
      num_analogs=_NUM_ANALOGS.value,
      window=WINDOW.value,
  )
  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    pcoll = (
        root
        | xbeam.DatasetToChunks(input_ds, input_chunks)
        | xbeam.SplitChunks({'time': 1})
    )

    # Branches for each analog pool.
    pcolls = []
    for pool_id, _ in enumerate(analog_pool_points):
      pcoll_tmp = pcoll | f'Analog pool {pool_id}' >> beam.MapTuple(
          functools.partial(
              collect_analogs_chunk, **collect_kwargs, pool_id=pool_id
          )
      )
      pcolls.append(pcoll_tmp)

    _ = (
        pcolls
        | beam.Flatten()
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value, template=template, zarr_chunks=output_chunks
        )
    )


if __name__ == '__main__':
  app.run(main)
