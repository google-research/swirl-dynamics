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

r"""Metric evaluation script for western United States downscaling baselines.

The simplest baseline considered is the input to the model, namely  the coarse
resolution fields interpolated to the high resolution domain. We can do so by
simply setting the `baseline_dataset_path` to the input dataset path, and
setting `samples_per_cond=1` to save compute, since this baseline is
deterministic. This script then reads the already interpolated fields from file
and passes them through an identity function sampler for evaluation.

Example usage:
```
MODEL_DIR=<insert_model_log_dir>
FORCING_DATASET=<parent_dir>/canesm5_r1i1p2f1_ssp370_bc

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/baseline_main.py \
--eval_date_start='2095-01-01' --eval_date_end='2095-12-31' \
--baseline_dataset_path=${FORCING_DATASET}/hourly_d01_cubic_interpolated_to_d02_with_prates.zarr \
--output_dataset=${FORCING_DATASET}/hourly_d02_with_prates.zarr \
--time_downsample=4 \
--logs_dir=${MODEL_DIR} --samples_per_cond=1 --sample_batch_size=1
```

Note that here the `MODEL_DIR` is only used to retrieve the config, but no model
inference is performed.

We can evaluate other baselines by setting the `baseline_dataset_path` to the
path to inference samples from the desired baseline model, and optionally
setting `samples_per_cond=1` to save compute for deterministic baselines.

Example usage:
```
MODEL_DIR=<insert_model_log_dir>
FORCING_DATASET=<parent_dir>/canesm5_r1i1p2f1_ssp370_bc
BASELINE_PATH=<baseline_inference_path>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/baseline_main.py \
--eval_date_start='2095-01-01' --eval_date_end='2095-12-31' \
--baseline_dataset_path=${BASELINE_PATH} \
--output_dataset=${FORCING_DATASET}/hourly_d02_with_prates.zarr \
--time_downsample=4 \
--logs_dir=${MODEL_DIR} --samples_per_cond=1 --sample_batch_size=1
```
"""

import os
from typing import Mapping

from absl import app
from absl import flags
from absl import logging
import gin
import jax
from jax import numpy as jnp
import numpy as np
from orbax import checkpoint
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import config_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import eval_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf.input_pipelines import paired_hourly
from swirl_dynamics.templates import evaluate
import tensorflow as tf
import xarray as xr


# See definitions in eval_lib.py.
_DEFAULT_DERIVED_VARS = ['WINDSPEED10', 'RH']

_BASELINE_DATASET_PATH = flags.DEFINE_string(
    'baseline_dataset_path',
    None,
    'Path to the baseline dataset to be evaluated.',
)
_LOGS_DIR = flags.DEFINE_string('logs_dir', None, 'Checkpoint logs directory.')
_EVAL_DATE_START = flags.DEFINE_string(
    'eval_date_start', '2013-09-01', 'Starting eval date.'
)
_EVAL_DATE_END = flags.DEFINE_string(
    'eval_date_end', '2013-09-01 8:00', 'Ending eval date.'
)
_SAMPLE_BATCH_SIZE = flags.DEFINE_integer(
    'sample_batch_size', 4, 'Inference batch size.'
)
_SAMPLES_PER_COND = flags.DEFINE_integer(
    'samples_per_cond', 8, 'Number of generated samples per conditioning input.'
)
_EXP_ID = flags.DEFINE_integer(
    'exp_id',
    0,
    'Inference experiment identifier, appended to output log directory.',
)
_NUM_AGGREGATION_BATCHES = flags.DEFINE_integer(
    'num_aggregation_batches', 24, 'Number of aggregation batches.'
)
_DUMPING_FREQ = flags.DEFINE_integer(
    'dumping_freq', 0, 'Number of aggregation batches.'
)
_OUTPUT_DATASET = flags.DEFINE_string(
    'output_dataset',
    None,
    'Output dataset used for evaluation, paired with the input dataset.'
    'If None, the training output dataset is used.',
)
_CHECKPOINT_FREQ = flags.DEFINE_integer(
    'checkpoint_freq', 1, 'Checkpointing frequency, in groups of batches.'
)
_NUM_CHECKPOINTS = flags.DEFINE_integer(
    'num_checkpoints', 2, 'Number of checkpoints to keep.'
)
_TIME_DOWNSAMPLE = flags.DEFINE_integer(
    'time_downsample', 4, 'The downsampling factor applied to the data source.'
)
_DERIVED_VARS = flags.DEFINE_list(
    'derived_vars',
    _DEFAULT_DERIVED_VARS,
    'List of derived variables to compute for evaluation.',
)

ArrayMapping = Mapping[str, jax.Array]


def input_as_output(
    num_samples: int,
    rng: jax.Array,
    cond: ArrayMapping | None = None,
    guidance_inputs: ArrayMapping | None = None,
) -> jax.Array:
  """Returns the conditioning input as a batch of output samples.

  Args:
    num_samples: The number of samples to generate in a single batch.
    rng: The base rng for the generation process.
    cond: Explicit conditioning inputs for the denoising function. These should
      be provided **without** batch dimensions (one should be added inside this
      function based on `num_samples`).
    guidance_inputs: Inputs used to construct the guided denoising function.
      These also should in principle not include a batch dimension.

  Returns:
    The input-as-output samples, all of them equal.
  """
  del rng, guidance_inputs
  cond = jax.tree.map(lambda x: jnp.stack([x] * num_samples, axis=0), cond)
  return cond['channel:input']


def main(_):
  logs_dir = _LOGS_DIR.value
  config_path = tf.io.gfile.glob(
      f'{os.path.dirname(logs_dir)}/config_file/*.gin'
  )[0]
  gin.parse_config_file(config_path)
  dataset_config = config_lib.DatasetConfig()
  batch_size = 1  # Required by eval benchmarks
  sample_batch_size = _SAMPLE_BATCH_SIZE.value
  date_range = (_EVAL_DATE_START.value, _EVAL_DATE_END.value)
  samples_per_cond = _SAMPLES_PER_COND.value
  # Dataset selection
  output_dataset = _OUTPUT_DATASET.value or dataset_config.output_dataset
  model_dir = os.path.dirname(output_dataset.rstrip('/'))

  if 'hourly_d01_cubic_interpolated_to_d02' in _BASELINE_DATASET_PATH.value:
    baseline_name = 'interp_baseline'
  elif 'bcsd' in _BASELINE_DATASET_PATH.value:
    baseline_name = 'bcsd_baseline'
  elif 'staresdm' in _BASELINE_DATASET_PATH.value:
    baseline_name = 'staresdm_baseline'
  else:
    raise ValueError('Baseline dataset path not recognized.')

  out_path = os.path.join(
      logs_dir,
      f'{baseline_name}_metrics',
      os.path.basename(model_dir),
      _EVAL_DATE_START.value.replace(' ', '_')
      + '_to_'
      + _EVAL_DATE_END.value.replace(' ', '_'),
      str(_EXP_ID.value),
  )

  # Disable GPU memory alloc
  tf.config.set_visible_devices([], device_type='GPU')

  source_kwargs = {
      'date_range': date_range,
      'input_dataset': _BASELINE_DATASET_PATH.value,
      # In order to evaluate the identity, we use the output variables.
      'input_variables': dataset_config.output_variables,
      'output_dataset': output_dataset,
      'output_variables': dataset_config.output_variables,
      'static_input_dataset': dataset_config.static_input_dataset,
      'static_input_variables': dataset_config.static_input_variables,
      'time_downsample': _TIME_DOWNSAMPLE.value,
      'resample_at_nan': False,
      'resample_seed': 42,
      'crop_input': False,
  }
  source = paired_hourly.DataSource(**source_kwargs)
  num_aggregation_batches = min(_NUM_AGGREGATION_BATCHES.value, len(source))

  output_coords = source.get_output_coords()
  # Add field names and wavenumbers as new coordinate for metrics
  num_wavenumbers = 2 ** (
      min(dataset_config.output_shape[-3:-1]).bit_length() - 1
  )
  resolution_km = 9.0
  wavenum = np.fft.fftshift(np.fft.fftfreq(num_wavenumbers, d=resolution_km))
  kx, ky = np.meshgrid(wavenum, wavenum)
  k = np.sqrt(kx**2 + ky**2)
  lengthscale = 1.0 / np.linspace(np.min(k), np.max(k), num=num_wavenumbers)[1:]

  output_coords = output_coords.merge(
      xr.Dataset(
          coords={
              'fields': dataset_config.output_variables + _DERIVED_VARS.value,
              'lengthscale': lengthscale,
          }
      ).coords
  )
  logging.info('Evaluating metrics for fields: %s', output_coords['fields'])

  test_loader = paired_hourly.create_dataset(
      **source_kwargs,
      input_stats=None,  # We want to pass this through the identity function
      output_stats=None,  # We want to evaluate on physical space
      random_maskout_probability=0.0,
      shuffle=False,
      seed=42,
      batch_size=batch_size,
      drop_remainder=True,
      worker_count=0,
      normalization='global',
  )

  benchmark = eval_lib.PairedDownscalingBenchmark(
      samples_per_cond,
      sample_batch_size,
      num_bins=num_wavenumbers,
      target_resolution=resolution_km,
      field_names=list(output_coords['fields'].values),
      landmask=xr.open_zarr(dataset_config.static_input_dataset)[
          'LANDMASK'
      ].values.astype(bool),
  )
  evaluator = eval_lib.PairedDownscalingEvaluator(
      models={baseline_name: input_as_output},
      benchmark=benchmark,
      rng=jax.random.PRNGKey(432),
  )
  max_eval_batches = (
      len(source) // num_aggregation_batches
  ) * num_aggregation_batches

  ckpt_options = checkpoint.CheckpointManagerOptions(
      save_interval_steps=_CHECKPOINT_FREQ.value,
      max_to_keep=_NUM_CHECKPOINTS.value,
  )

  evaluate.run(
      evaluator=evaluator,
      dataloader=test_loader,
      workdir=out_path,
      num_aggregation_batches=num_aggregation_batches,
      dump_collected_every_n_groups=_DUMPING_FREQ.value,
      results_format='zarr',
      datacoords=output_coords,
      max_eval_batches=max_eval_batches,
      checkpoint_options=ckpt_options,
  )


if __name__ == '__main__':
  app.run(main)
