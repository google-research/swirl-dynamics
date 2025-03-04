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

r"""Metric evaluation script for western United States downscaling.

Example usage:

```
MODEL_DIR=<insert_model_log_dir>
FORCING_DATASET=<parent_dir>/canesm5_r1i1p2f1_ssp370_bc

python swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/evaluate_main.py \
--eval_date_start='2095-03-01' --eval_date_end='2095-05-31' \
--input_dataset=${FORCING_DATASET}/hourly_d01_cubic_interpolated_to_d02_with_prates.zarr \
--output_dataset=${FORCING_DATASET}/hourly_d02_with_prates.zarr \
--time_downsample=4 \
--logs_dir=${MODEL_DIR} --samples_per_cond=32 --sample_batch_size=32
```
"""

import functools
import os

from absl import app
from absl import flags
from absl import logging
from clu import platform
import gin
import jax
import numpy as np
from orbax import checkpoint
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import config_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import data_utils
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import eval_lib
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf import models
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.gcm_wrf.input_pipelines import paired_hourly
from swirl_dynamics.templates import evaluate
import tensorflow as tf
import xarray as xr


# See definitions in eval_lib.py.
_DEFAULT_DERIVED_VARS = ['WINDSPEED10', 'RH']

FLAGS = flags.FLAGS

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
    'num_aggregation_batches', 12, 'Number of aggregation batches.'
)
_DUMPING_FREQ = flags.DEFINE_integer(
    'dumping_freq', 0, 'Number of aggregation batches.'
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
  # Flags --jax_backend_target and --jax_xla_backend are available through JAX.
  if FLAGS.jax_backend_target:
    logging.info('Using JAX backend target %s', FLAGS.jax_backend_target)
    jax_xla_backend = (
        'None' if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
    )
    logging.info('Using JAX XLA backend %s', jax_xla_backend)
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX devices: %r', jax.devices())
  platform.work_unit().set_task_status(
      f'process_index: {jax.process_index()}, '
      f'process_count: {jax.process_count()}'
  )

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
      'eval_metrics',
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
      'forcing_dataset': dataset_config.forcing_dataset,
      'use_temporal_inputs': dataset_config.use_temporal_inputs,
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
      input_stats=dataset_config.input_stats,
      output_stats=None,  # We want to evaluate on physical space
      random_maskout_probability=_RANDOM_MASKOUT_PROBABILITY.value,
      shuffle=False,
      seed=42,
      batch_size=batch_size,
      drop_remainder=True,
      worker_count=0,
      normalization=dataset_config.normalization,
  )

  denoising_model_cls = gin.get_configurable(models.DenoisingModel)
  denoising_model = denoising_model_cls()

  # Restore train state from checkpoint. By default, the most recently saved
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
  if dataset_config.normalization == 'local':
    read_stats_fn = data_utils.read_stats
  else:
    read_stats_fn = data_utils.read_global_stats
  output_mean = read_stats_fn(
      dataset_config.output_stats, dataset_config.output_variables, 'mean'
  )
  output_std = read_stats_fn(
      dataset_config.output_stats, dataset_config.output_variables, 'std'
  )

  guidance_strength = _CFG_STRENGTH.value or denoising_model.cg_strength
  num_sde_steps = _NUM_SDE_STEPS.value or denoising_model.num_sde_steps

  sampler = models.SdeSampler(
      input_shape=dataset_config.output_shape,
      integrator=solver_lib.EulerMaruyama(),
      tspan=dfn_lib.edm_noise_decay(
          denoising_model.diffusion_scheme,
          # num_steps increases leads to more cost, and more acccuracy.
          num_steps=num_sde_steps,
      ),
      scheme=denoising_model.diffusion_scheme,
      denoise_fn=denoise_fn,
      guidance_transforms=(
          # guidance_strength improves some metrics at the cost of diversity.
          dfn_lib.ClassifierFreeHybrid(guidance_strength=guidance_strength),
      ),
      apply_denoise_at_end=True,
      return_full_paths=False,
      rescale_mean=output_mean,
      rescale_std=output_std,
  )
  if is_residual:
    input_indices = [
        dataset_config.input_variables.index(varname)
        for varname in dataset_config.output_variables
    ]
    input_mean = read_stats_fn(
        dataset_config.input_stats, dataset_config.output_variables, 'mean'
    )
    input_std = read_stats_fn(
        dataset_config.input_stats, dataset_config.output_variables, 'std'
    )
    sampling_fn = functools.partial(
        sampler.generate_denormalize_and_add_input,
        input_indices,
        input_mean,
        input_std,
    )
  else:
    sampling_fn = sampler.generate_and_denormalize

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
      models={'model': sampling_fn},
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
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  handler = app.run
  handler(main)
