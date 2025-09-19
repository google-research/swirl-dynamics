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

r"""Inference script for western United States downscaling.

Example usage:

```
MODEL_DIR=<insert_model_log_dir>
FORCING_DATASET=<insert_forcing_dataset_dir>

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/inference_main.py \
--input_dataset=${FORCING_DATASET}/hourly_d01_cubic_interpolated_to_d02_with_prates.zarr \
--output_dataset=${FORCING_DATASET}/hourly_d02_with_prates.zarr \
--logs_dir=${MODEL_DIR} --time_downsample=24 --samples_per_cond=32 --batch_size=1 \
--num_sde_steps=256 --cfg_strength=0.2 \
--eval_date_start='2095-06-01' --eval_date_end='2095-08-31' --write_inputs_and_targets=True
```

"""
import dataclasses
import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import config_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import data_utils
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import eval_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import inference_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import models
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import trainers
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf.input_pipelines import paired_hourly
import tensorflow as tf


# See definitions in eval_lib.py.
_DEFAULT_DERIVED_VARS = None  # ['SAWTI', 'WBGT', 'WINDSPEED10', 'RH', 'FFWI']

_LOGS_DIR = flags.DEFINE_string('logs_dir', None, 'Checkpoint logs directory.')
_EVAL_DATE_START = flags.DEFINE_string(
    'eval_date_start', '2013-09-01', 'Starting eval date.'
)
_EVAL_DATE_END = flags.DEFINE_string(
    'eval_date_end', '2014-09-01', 'Ending eval date.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'Inference batch size.')
_SAMPLES_PER_COND = flags.DEFINE_integer(
    'samples_per_cond',
    32,
    'Number of generated samples per conditioning input.',
)
_WRITE_FREQ = flags.DEFINE_integer(
    'write_freq', 4, 'Frequency at which evaluated batches are written to file.'
)
_WRITE_INPUTS_AND_TARGETS = flags.DEFINE_bool(
    'write_inputs_and_targets',
    False,
    'Whether to write inputs and targets to file.',
)
_EXP_ID = flags.DEFINE_integer(
    'exp_id',
    0,
    'Inference experiment identifier, appended to output log directory.',
)
_INPUT_DATASET = flags.DEFINE_string(
    'input_dataset',
    None,
    'Input dataset used for evaluation.'
    'If None, the training input dataset is used.',
)
_OUTPUT_DATASET = flags.DEFINE_string(
    'output_dataset',
    None,
    'Output dataset used for evaluation, paired with the input dataset.'
    'If None, the training output dataset is used.',
)
_TIME_DOWNSAMPLE = flags.DEFINE_integer(
    'time_downsample', 4, 'The downsampling factor applied to the data source.'
)
_DERIVED_VARS = flags.DEFINE_list(
    'derived_vars',
    _DEFAULT_DERIVED_VARS,
    'List of derived variables to compute for evaluation.',
)
_NUM_SDE_STEPS = flags.DEFINE_integer(
    'num_sde_steps',
    None,
    'Number of SDE steps to use for sampling.',
)
_CFG_STRENGTH = flags.DEFINE_float(
    'cfg_strength',
    None,
    'Classifier-free guidance strength to use for sampling.',
)
_RANDOM_MASKOUT_PROBABILITY = flags.DEFINE_float(
    'random_maskout_probability',
    0.0,
    'Probability of randomly masking out input variables.',
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
  samples_per_cond = _SAMPLES_PER_COND.value
  derived_vars = []
  if _DERIVED_VARS.value:
    derived_vars = [str(var) for var in _DERIVED_VARS.value]
  # Dataset selection
  input_dataset = _INPUT_DATASET.value or dataset_config.input_dataset
  output_dataset = _OUTPUT_DATASET.value or dataset_config.output_dataset
  # Verification of correct dataset pairs
  model_dir = os.path.dirname(input_dataset.rstrip('/'))
  if model_dir != os.path.dirname(output_dataset.rstrip('/')):
    raise ValueError('input_dataset and output_dataset must be paired.')
  # Retrieve whether model output is diff with respect to input
  is_residual = 'diff_d02_d01' in dataset_config.output_dataset
  if is_residual and 'diff_d02_d01' in output_dataset:
    raise ValueError(
        'Residual downscaling models should be evaluated against '
        'the full high resolution data.'
    )

  out_path = os.path.join(
      logs_dir,
      'inference',
      os.path.basename(model_dir),
      _EVAL_DATE_START.value.replace(' ', '_')
      + '_to_'
      + _EVAL_DATE_END.value.replace(' ', '_'),
      str(_EXP_ID.value),
  )

  lead_host = jax.process_index() == 0
  # Disable GPU memory alloc
  tf.config.set_visible_devices([], device_type='GPU')

  trainer_cls = gin.get_configurable(trainers.Trainer)
  denoising_model = trainer_cls().model
  denoising_kwargs = {}
  if _CFG_STRENGTH.value is not None:
    denoising_kwargs['cg_strength'] = _CFG_STRENGTH.value
  if _NUM_SDE_STEPS.value is not None:
    denoising_kwargs['num_sde_steps'] = _NUM_SDE_STEPS.value
  if denoising_kwargs:
    denoising_model = dataclasses.replace(
        denoising_model,
        cg_strength=_CFG_STRENGTH.value,
        num_sde_steps=_NUM_SDE_STEPS.value,
    )

  # Restore train state from checkpoint. By default, the move recently saved
  # checkpoint is restored. Alternatively, one can directly use
  # `trainer.train_state` if continuing from the training section above.
  trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
      f'{logs_dir}/checkpoints', step=None
  )
  # Construct the inference function
  denoise_fn = dfn.DenoisingTrainer.inference_fn_from_state_dict(
      trained_state, use_ema=True, denoiser=denoising_model.denoiser
  )

  # Recover normalization stats
  in_stat_kwargs = {}
  if dataset_config.normalization == 'local':
    read_stats_fn = data_utils.read_stats
    if dataset_config.crop_input:
      in_stat_kwargs['crop_dict'] = paired_hourly.D2_WITHIN_D1
  else:
    read_stats_fn = data_utils.read_global_stats

  output_mean = read_stats_fn(
      dataset_config.output_stats, dataset_config.output_variables, 'mean'
  )
  output_std = read_stats_fn(
      dataset_config.output_stats, dataset_config.output_variables, 'std'
  )

  sampler = models.SdeSampler(
      noise_dist=denoising_model.noise_dist,
      input_shape=dataset_config.output_shape,
      integrator=solver_lib.EulerMaruyama(),
      tspan=dfn_lib.edm_noise_decay(
          denoising_model.diffusion_scheme,
          # num_steps increases leads to more cost, and more acccuracy.
          num_steps=denoising_model.num_sde_steps,
      ),
      scheme=denoising_model.diffusion_scheme,
      denoise_fn=denoise_fn,
      guidance_transforms=(
          # guidance_strength improves some metrics at the cost of diversity.
          dfn_lib.ClassifierFreeHybrid(
              guidance_strength=denoising_model.cg_strength
          ),
      ),
      apply_denoise_at_end=True,
      return_full_paths=False,
      rescale_mean=output_mean,
      rescale_std=output_std,
  )

  input_mean = read_stats_fn(
      dataset_config.input_stats,
      dataset_config.input_variables,
      'mean',
      **in_stat_kwargs,
  )
  input_std = read_stats_fn(
      dataset_config.input_stats,
      dataset_config.input_variables,
      'std',
      **in_stat_kwargs,
  )

  if is_residual:
    input_indices = [
        dataset_config.input_variables.index(varname)
        for varname in dataset_config.output_variables
    ]
    sampling_fn = functools.partial(
        sampler.generate_denormalize_and_add_input,
        input_indices=input_indices,
        input_mean=input_mean,
        input_std=input_std,
    )
  else:
    sampling_fn = sampler.generate_and_denormalize

  generate = functools.partial(
      eval_lib.batch_inference, sampling_fn, samples_per_cond
  )

  source_kwargs = {
      'date_range': date_range,
      'input_dataset': input_dataset,
      'input_variables': dataset_config.input_variables,
      'output_dataset': output_dataset,
      'output_variables': dataset_config.output_variables,
      'static_input_dataset': dataset_config.static_input_dataset,
      'static_input_variables': dataset_config.static_input_variables,
      'time_downsample': _TIME_DOWNSAMPLE.value,
      'resample_at_nan': False,
      'resample_seed': 42,
      'crop_input': dataset_config.crop_input,
      'use_temporal_inputs': dataset_config.use_temporal_inputs,
      'forcing_dataset': dataset_config.forcing_dataset,
  }

  source = paired_hourly.DataSource(**source_kwargs)
  source_times = source.get_dates()

  test_loader = paired_hourly.create_dataset(
      **source_kwargs,
      input_stats=dataset_config.input_stats,
      output_stats=None,  # We want to recover targets without normalization.
      random_maskout_probability=_RANDOM_MASKOUT_PROBABILITY.value,
      shuffle=False,
      seed=42,
      batch_size=batch_size,
      drop_remainder=True,
      worker_count=0,
      normalization=dataset_config.normalization,
  )

  inference_ds_list = []
  input_ds_list = []
  targets_ds_list = []
  test_loader_iter = iter(test_loader)
  num_batches = len(source) // batch_size  # Drop remainder
  for batch_id in range(num_batches):

    test_batch = next(test_loader_iter)
    generated_samples = jax.vmap(generate, in_axes=(0, 0))(
        test_batch, jax.random.split(jax.random.PRNGKey(456), batch_size)
    )
    generated_samples = jnp.reshape(
        generated_samples, (-1,) + generated_samples.shape[2:]
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
          'Shape of conditioning inputs: %s',
          jax.tree.map(np.shape, test_batch['cond']),
      )
      logging.info(
          'Shape of generated samples (including derived fields) is'
          ' (batch_size, samples_per_cond, south_north, west_east, fields)'
          ' =  %s',
          jax.tree.map(np.shape, generated_samples),
      )

    inference_ds_list.append(
        inference_lib.samples_to_dataset(
            generated_samples,
            field_names=dataset_config.output_variables + derived_vars,
            times=times,
        )
    )

    if _WRITE_INPUTS_AND_TARGETS.value:

      inputs = test_batch['cond']['channel:input'] * input_std + input_mean
      targets = test_batch['x']

      if derived_vars:
        derived_inputs = eval_lib.get_derived_fields(
            inputs,
            dataset_config.input_variables + derived_vars,
        )
        inputs = jnp.concatenate((inputs, derived_inputs), axis=-1)

        derived_targets = eval_lib.get_derived_fields(
            targets,
            dataset_config.output_variables + derived_vars,
        )
        targets = jnp.concatenate((targets, derived_targets), axis=-1)

      input_ds_list.append(
          inference_lib.batch_to_dataset(
              inputs,
              field_names=dataset_config.input_variables + derived_vars,
              times=times,
          )
      )
      targets_ds_list.append(
          inference_lib.batch_to_dataset(
              targets,
              field_names=dataset_config.output_variables + derived_vars,
              times=times,
          )
      )

    if batch_id % _WRITE_FREQ.value == 0 or batch_id == num_batches - 1:
      logging.info('Inference for batch %s finished.', batch_id)
      if lead_host:
        inference_lib.concat_to_zarr(
            inference_ds_list,
            out_path,
            'inference',
            append=batch_id > 0,
            append_dim='time',
        )
      inference_ds_list = []

      if _WRITE_INPUTS_AND_TARGETS.value:
        if lead_host:
          inference_lib.concat_to_zarr(
              input_ds_list,
              out_path,
              'input',
              append=batch_id > 0,
              append_dim='time',
          )
          inference_lib.concat_to_zarr(
              targets_ds_list,
              out_path,
              'target',
              append=batch_id > 0,
              append_dim='time',
          )
        input_ds_list = []
        targets_ds_list = []


if __name__ == '__main__':
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  handler = app.run
  handler(main)
