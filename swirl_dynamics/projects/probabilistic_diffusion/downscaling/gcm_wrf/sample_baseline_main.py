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

r"""Sampling script for western United States downscaling baselines.

Example usage:
```
MODEL_DIR=<insert_model_log_dir>
BASELINE_PATH=<baseline_inference_path>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/xm_launch_sample_baseline.py \
--logs_dir=${MODEL_DIR} --baseline_dataset_path=${BASELINE_PATH} \
--eval_date_start='2095-06-01' --eval_date_end='2095-08-31' \
--time_downsample=24
```
"""

import os

from absl import app
from absl import flags
from absl import logging
import gin
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import config_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import eval_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import inference_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf.input_pipelines import paired_hourly
import tensorflow as tf


# See definitions in eval_lib.py.
_DEFAULT_DERIVED_VARS = ['SAWTI', 'WBGT', 'WINDSPEED10', 'RH']

_BASELINE_DATASET_PATH = flags.DEFINE_string(
    'baseline_dataset_path',
    None,
    'Path to the baseline dataset to be evaluated.',
    required=True,
)
_LOGS_DIR = flags.DEFINE_string('logs_dir', None, 'Checkpoint logs directory.')
_EVAL_DATE_START = flags.DEFINE_string(
    'eval_date_start', '2013-09-01', 'Starting eval date.'
)
_EVAL_DATE_END = flags.DEFINE_string(
    'eval_date_end', '2014-09-01', 'Ending eval date.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'Inference batch size.')
_WRITE_FREQ = flags.DEFINE_integer(
    'write_freq', 4, 'Frequency at which evaluated batches are written to file.'
)
_EXP_ID = flags.DEFINE_integer(
    'exp_id',
    0,
    'Inference experiment identifier, appended to output log directory.',
)
_TIME_DOWNSAMPLE = flags.DEFINE_integer(
    'time_downsample', 4, 'The downsampling factor applied to the data source.'
)
_DERIVED_VARS = flags.DEFINE_list(
    'derived_vars',
    _DEFAULT_DERIVED_VARS,
    'List of derived variables to compute for evaluation.',
)


def main(_):
  logs_dir = _LOGS_DIR.value
  config_path = tf.io.gfile.glob(
      f'{os.path.dirname(logs_dir)}/config_file/*.gin'
  )[0]
  gin.parse_config_file(config_path)
  dataset_config = config_lib.DatasetConfig()
  batch_size = _BATCH_SIZE.value
  date_range = (_EVAL_DATE_START.value, _EVAL_DATE_END.value)
  derived_vars = []
  if _DERIVED_VARS.value:
    derived_vars = [str(var) for var in _DERIVED_VARS.value]
  model_dir = os.path.dirname(
      os.path.dirname(_BASELINE_DATASET_PATH.value.rstrip('/'))
  )

  if 'hourly_d01_cubic_interpolated_to_d02' in _BASELINE_DATASET_PATH.value:
    baseline_name = 'interp_inference'
  elif 'bcsd' in _BASELINE_DATASET_PATH.value:
    baseline_name = 'bcsd_inference'
  elif 'staresdm' in _BASELINE_DATASET_PATH.value:
    baseline_name = 'staresdm_inference'
  else:
    raise ValueError('Baseline dataset path not recognized.')

  out_path = os.path.join(
      logs_dir,
      baseline_name,
      os.path.basename(model_dir),
      _EVAL_DATE_START.value.replace(' ', '_')
      + '_to_'
      + _EVAL_DATE_END.value.replace(' ', '_'),
      str(_EXP_ID.value),
  )

  lead_host = jax.process_index() == 0
  # Disable GPU memory alloc
  tf.config.set_visible_devices([], device_type='GPU')

  source_kwargs = {
      'date_range': date_range,
      'input_dataset': _BASELINE_DATASET_PATH.value,
      # In order to evaluate the identity, we use the output variables.
      'input_variables': dataset_config.output_variables,
      'output_dataset': _BASELINE_DATASET_PATH.value,
      'output_variables': dataset_config.output_variables,
      'static_input_dataset': dataset_config.static_input_dataset,
      'static_input_variables': dataset_config.static_input_variables,
      'time_downsample': _TIME_DOWNSAMPLE.value,
      'resample_at_nan': False,
      'resample_seed': 42,
      'crop_input': False,
  }
  source = paired_hourly.DataSource(**source_kwargs)
  source_times = source.get_dates()

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

  # Construct the pass-through inference function
  generate = lambda x: x['cond']['channel:input']

  inference_ds_list = []
  test_loader_iter = iter(test_loader)
  num_batches = len(source) // batch_size  # Drop remainder
  logging.info('Number of sampling batches: %s', num_batches)
  for batch_id in range(num_batches):

    test_batch = next(test_loader_iter)
    generated_samples = jax.vmap(generate, in_axes=(0,))(test_batch)
    generated_samples = jnp.reshape(
        generated_samples, (-1,) + generated_samples.shape[1:]
    )

    times = source_times[
        range(batch_size * batch_id, batch_size * (batch_id + 1))
    ]

    if derived_vars:
      derived_samples = eval_lib.get_derived_fields(
          generated_samples,
          dataset_config.output_variables + derived_vars,
      )
      generated_samples = jnp.concatenate(
          (generated_samples, derived_samples), axis=-1
      )

    if batch_id == 0:
      if derived_vars:
        logging.info('Derived variables: %s', derived_vars)

      logging.info(
          'Shape of generated samples (including derived fields) is'
          ' (batch_size, south_north, west_east, fields)'
          ' =  %s',
          jax.tree.map(np.shape, generated_samples),
      )

    inference_ds_list.append(
        inference_lib.batch_to_dataset(
            generated_samples,
            field_names=dataset_config.output_variables + derived_vars,
            times=times,
        )
    )

    if batch_id % _WRITE_FREQ.value == 0:
      logging.info('Inference for batch %s finished.', batch_id)
      logging.info(
          'Length of inference_ds_list wrote to file: %s',
          len(inference_ds_list),
      )
      if lead_host:
        inference_lib.concat_to_zarr(
            inference_ds_list,
            out_path,
            'inference',
            append=batch_id > 0,
            append_dim='time',
        )
      inference_ds_list = []


if __name__ == '__main__':
  app.run(main)
