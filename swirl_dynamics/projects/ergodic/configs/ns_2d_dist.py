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

r"""Default Hyperparameter configuration for Navier Stokes 2D.

"""

import ml_collections

# pylint: disable=line-too-long
DATA_PATH = '/datasets/hdf5/pde/2d/ns/ns_trajectories_from_caltech.hdf5'
# DATA_PATH = '/datasets/hdf5/pde/2d/ns/attractor_spectral_grid_256_spatial_downsample_4_dt_0.001_v0_3_warmup_40.0_t_final_200.0_nu_0.001_n_samples_2000_ntraj_train_128_ntraj_eval_32_ntraj_test_32_drag_0.1_wave_number_4_random_seeds_combined_4.hdf5'
# pylint: enable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = 'ns_2d'
  # Train params
  config.train_steps = 360_000
  config.seed = 42
  config.lr = 5e-5
  config.metric_aggregation_steps = 50
  config.save_interval_steps = 50_000
  config.max_checkpoints_to_keep = 10
  # Data params
  config.batch_size = 64
  config.num_time_steps = 10
  config.time_stride = 1
  config.dataset_path = DATA_PATH
  config.spatial_downsample_factor = 1
  config.normalize = False
  config.add_noise = False
  config.noise_level = 0.0
  config.use_sobolev_norm = False
  config.order_sobolev_norm = 0

  # Model params
  config.num_lookback_steps = 1
  config.integrator = 'OneStepDirect'
  config.model = 'PeriodicConvNetModel'
  config.latent_dim = 128
  config.num_levels = 2
  config.num_processors = 4
  config.encoder_kernel_size = (3, 3)
  config.decoder_kernel_size = (3, 3)
  config.processor_kernel_size = (3, 3)
  config.padding = 'CIRCULAR'
  config.is_input_residual = True
  ########### FNO ################
  # config.num_lookback_steps = 2
  # config.integrator = 'MultiStepDirect'
  # config.model = 'FNO'
  # config.out_channels = 1
  # config.hidden_channels = 64
  # config.num_modes = (20, 20)
  # config.lifting_channels = 256
  # config.projection_channels = 256
  # config.num_blocks = 4
  # config.layers_per_block = 2
  # config.block_skip_type = 'identity'
  # config.fft_norm = 'forward'
  # config.separable = False
  # Update num_time_steps based on num_lookback_steps setting
  config.num_time_steps += config.num_lookback_steps - 1
  # Trainer params
  config.num_rollout_steps = 1
  config.train_steps_per_cycle = 0
  config.time_steps_increase_per_cycle = 1
  config.use_curriculum = False  # Sweepable
  config.use_pushfwd = False  # Sweepable
  config.measure_dist_downsample = 1
  config.measure_dist_lambda = 0.0  # Sweepable
  config.measure_dist_k_lambda = 10.0  # Sweepable
  config.measure_dist_type = 'MMD_DIST'  # Sweepable
  config.use_distributed = True
  return config


# TODO(yairschiff): Refactor sweeps and experiment definition to use gin.
def sweep(add):
  """Define param sweep."""
  for seed in [42]:
    for measure_dist_type in ['MMD', 'SD']:
      for measure_dist_k_lambda in [100.0, 1000.0]:
        for measure_dist_lambda in [0.0]:
          for measure_dist_downsample in [1, 2]:
            if measure_dist_k_lambda == measure_dist_lambda == 0.0:
              if measure_dist_type == 'SD' or measure_dist_downsample > 1:
                continue  # Avoid re-running baseline exp multiple times
            add(
                seed=seed,
                measure_dist_type=measure_dist_type,
                measure_dist_lambda=measure_dist_lambda,
                measure_dist_k_lambda=measure_dist_k_lambda,
                measure_dist_downsample=measure_dist_downsample,
            )
