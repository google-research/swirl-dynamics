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
DATA_PATH = '/datasets/hdf5/pde/2d/ns/attractor_spectral_grid_256_spatial_downsample_4_dt_0.001_v0_3_warmup_40.0_t_final_200.0_nu_0.001_n_samples_2000_ntraj_train_256_ntraj_eval_32_ntraj_test_32_drag_0.1_wave_number_4_random_seeds_combined_4.hdf5'
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
  config.save_interval_steps = 36_000
  config.max_checkpoints_to_keep = 10
  config.use_sobolev_norm = True
  config.order_sobolev_norm = 1

  # Data params
  config.batch_size = 1024
  # num_time_steps is length of ground truth trajectories in each batch from
  # dataloader (this differs from num_time_steps in Trainer.preprocess_batch
  # functions that corresponds to the len of ground truth trajectories to
  # actually pass to model, which can vary during training).
  config.num_time_steps = 11
  config.time_stride = 1  # factor for downsampling time dim of ground truth
  config.dataset_path = DATA_PATH
  config.spatial_downsample_factor = 1
  config.normalize = True
  config.add_noise = False
  config.noise_level = 0.0
  config.tf_lookup_batch_size = 512
  config.tf_lookup_num_parallel_calls = -1
  config.tf_interleaved_shuffle = False

  # Model params
  ########### PeriodicConvNetModel ################
  config.model = 'PeriodicConvNetModel'
  config.latent_dim = 96
  config.num_levels = 4
  config.num_processors = 4
  config.encoder_kernel_size = (3, 3)
  config.decoder_kernel_size = (3, 3)
  config.processor_kernel_size = (3, 3)
  config.padding = 'CIRCULAR'
  config.is_input_residual = True

  config.num_lookback_steps = 1
  # Update num_time_steps and integrator based on num_lookback_steps setting
  if config.num_lookback_steps > 1:
    config.integrator = 'MultiStepDirect'
  else:
    config.integrator = 'OneStepDirect'
  # Trainer params
  config.num_rollout_steps = 1
  config.train_steps_per_cycle = 0
  config.time_steps_increase_per_cycle = 1
  config.use_curriculum = False  # Sweepable
  config.use_pushfwd = False  # Sweepable
  config.measure_dist_type = 'MMD'  # Sweepable
  config.measure_dist_downsample = 1
  config.measure_dist_lambda = 0.0  # Sweepable
  config.measure_dist_k_lambda = 1.0  # Sweepable
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


# TODO: Refactor sweeps and experiment definition to use gin.
# use option --sweep=False in the command line to avoid sweeping
def sweep(add):
  """Define param sweep."""
  for seed in [42]:
    for normalize in [False, True]:
      for measure_dist_type in ['MMD', 'SD']:
        for batch_size in [50]:
          for lr in [5e-4]:
            for use_curriculum in [False]:
              for use_pushfwd in [False]:
                for measure_dist_lambda in [0.0, 1.0]:
                  for measure_dist_k_lambda in [0.0, 1.0, 100.0]:
                    if use_curriculum:
                      train_steps_per_cycle = 72_000
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
                        normalize=normalize,
                        lr=lr,
                        measure_dist_type=measure_dist_type,
                        train_steps_per_cycle=train_steps_per_cycle,
                        time_steps_increase_per_cycle=time_steps_increase_per_cycle,
                        use_curriculum=use_curriculum,
                        use_pushfwd=use_pushfwd,
                        measure_dist_lambda=measure_dist_lambda,
                        measure_dist_k_lambda=measure_dist_k_lambda,
                    )
