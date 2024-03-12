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

r"""Default Hyperparameter configuration.

"""

import ml_collections



def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = 'ks_1d'
  # Train params
  config.train_steps = 500_000
  config.seed = 42
  config.lr = 1e-4
  config.metric_aggregation_steps = 50
  config.save_interval_steps = 50_000
  config.max_checkpoints_to_keep = 10
  # Data params
  config.use_tfds = True
  # config.batch_size = 4096
  config.batch_size = 1024
  config.num_time_steps = 11
  config.time_stride = 1
  config.dataset_path = DATA_PATH
  config.dataset_name = DATA_NAME
  config.spatial_downsample_factor = 1
  config.normalize = False
  config.add_noise = False
  config.use_sobolev_norm = False
  config.order_sobolev_norm = 0
  config.noise_level = 0.0
  config.num_time_steps_eval = 600
  config.batch_size_eval = 512
  config.tf_lookup_batch_size = 8192
  config.tf_lookup_num_parallel_calls = -1
  config.tf_interleaved_shuffle = False

  # Model params
  # ######## Dilated Convolutions ########
  config.num_lookback_steps = 1
  config.integrator = 'OneStepDirect'
  config.model = 'PeriodicConvNetModel'
  config.latent_dim = 48
  config.num_levels = 4
  config.num_processors = 4
  config.encoder_kernel_size = (5,)
  config.decoder_kernel_size = (5,)
  config.processor_kernel_size = (5,)
  config.padding = 'CIRCULAR'
  config.is_input_residual = True

  # Trainer params
  config.num_rollout_steps = 1
  config.train_steps_per_cycle = 50_000
  config.time_steps_increase_per_cycle = 1
  config.use_curriculum = True  # Sweepable
  config.use_pushfwd = True  # Sweepable
  config.measure_dist_downsample = 1
  config.measure_dist_lambda = 0.0  # Sweepable
  config.measure_dist_k_lambda = 0.0  # Sweepable
  # Using distributed training.
  config.measure_dist_type = 'MMD_DIST'  # Sweepable
  config.use_distributed = True
  return config


def skip(
    use_curriculum: bool,
    use_pushfwd: bool,
    measure_dist_lambda: float,
    measure_dist_k_lambda: float,
    measure_dist_type: str,
) -> bool:
  """Helper method for avoiding unwanted runs in sweep."""

  if not use_curriculum and use_pushfwd:
    return True
  if (
      measure_dist_type == 'SD'
      and measure_dist_lambda == 0.0
      and measure_dist_k_lambda == 0.0
  ):
    return True
  return False


# use option --sweep=False in the command line to avoid sweeping
def sweep(add):
  """Define param sweep."""
  for seed in [42]:
    for measure_dist_type in ['MMD', 'SD']:
      for batch_size in [128]:
        for lr in [1e-4]:
          for use_curriculum in [True]:
            for use_pushfwd in [False, True]:
              for measure_dist_lambda in [0.0, 1.0, 100.0]:
                for measure_dist_k_lambda in [0.0, 1.0, 100.0]:
                  if use_curriculum:
                    train_steps_per_cycle = 50_000
                    time_steps_increase_per_cycle = 1
                  else:
                    train_steps_per_cycle = 0
                    time_steps_increase_per_cycle = 0
                  if skip(
                      use_curriculum,
                      use_pushfwd,
                      measure_dist_lambda,
                      measure_dist_k_lambda,
                      measure_dist_type,
                  ):
                    continue
                  add(
                      seed=seed,
                      batch_size=batch_size,
                      lr=lr,
                      measure_dist_type=measure_dist_type,
                      train_steps_per_cycle=train_steps_per_cycle,
                      time_steps_increase_per_cycle=time_steps_increase_per_cycle,
                      use_curriculum=use_curriculum,
                      use_pushfwd=use_pushfwd,
                      measure_dist_lambda=measure_dist_lambda,
                      measure_dist_k_lambda=measure_dist_k_lambda,
                  )

