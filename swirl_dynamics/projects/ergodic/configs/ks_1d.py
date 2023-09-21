# Copyright 2023 The swirl_dynamics Authors.
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

# pylint: disable=line-too-long
DATA_PATH = '/datasets/gcs_staging/hdf5/pde/1d/ks_trajectories.hdf5'
# pylint: enable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = 'ks_1d'
  # Train params
  config.train_steps = 50_000
  config.seed = 42
  config.lr = 1e-4
  config.metric_aggregation_steps = 50
  config.save_interval_steps = 5_000
  config.max_checkpoints_to_keep = 10
  # Data params
  config.batch_size = 128
  # num_time_steps is length of ground truth trajectories in each batch from
  # dataloader (this differs from num_time_steps in Trainer.preprocess_batch
  # functions that corresponds to the len of ground truth trajectories to
  # actually pass to model, which can vary during training).
  config.num_time_steps = 11
  config.time_stride = 1  # factor for downsampling time dim of ground truth
  config.dataset_path = DATA_PATH
  config.spatial_downsample_factor = 1
  config.normalize = False
  config.add_noise = False
  config.use_sobolev_norm = False
  config.order_sobolev_norm = 1
  config.noise_level = 0.0

  # TODO(yairschiff): Split different models into separate configs
  # Model params
  ########### PeriodicConvNetModel ################
  # config.model = 'PeriodicConvNetModel'
  # config.latent_dim = 48
  # config.num_levels = 4
  # config.num_processors = 4
  # config.encoder_kernel_size = (5,)
  # config.decoder_kernel_size = (5,)
  # config.processor_kernel_size = (5,)
  # config.padding = 'CIRCULAR'
  # config.is_input_residual = True
  ########### FNO ################
  config.model = 'Fno'
  config.out_channels = 1
  config.hidden_channels = 64
  config.num_modes = (24,)
  config.lifting_channels = 256
  config.projection_channels = 256
  config.num_blocks = 4
  config.layers_per_block = 2
  config.block_skip_type = 'identity'
  config.fft_norm = 'forward'
  config.separable = False

  config.num_lookback_steps = 1
  # Update num_time_steps and integrator based on num_lookback_steps setting
  config.num_time_steps += config.num_lookback_steps - 1
  if config.num_lookback_steps > 1:
    config.integrator = 'MultiStepDirect'
  else:
    config.integrator = 'OneStepDirect'
  # Trainer params
  config.num_rollout_steps = 1
  config.train_steps_per_cycle = 5_000
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


# TODO(yairschiff): Refactor sweeps and experiment definition to use gin.
# use option --sweep=False in the command line to avoid sweeping
def sweep(add):
  """Define param sweep."""
  for seed in [42]:
    for normalize in [False, True]:
      for batch_size in [128]:
        for lr in [1e-4]:
          for use_curriculum in [False, True]:
            for use_pushfwd in [False, True]:
              for measure_dist_type in ['MMD', 'SD']:
                for measure_dist_lambda in [0.0, 1.0]:
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


# def sweep(add):
#   """Define param sweep."""
#   for seed in [21, 42, 84]:
#     for measure_dist_type in ['MMD', 'SD']:
#       for batch_size in [32, 64, 128, 256]:
#         for lr in [1e-3, 1e-4, 1e-5]:
#           # Skipping 1-step objective
#           for use_curriculum in [True]:  # [False, True]:
#             # Running grid search on just Pfwd
#             for use_pushfwd in [True]:  # [False, True]:
#               # Skipping all x0 regs
#               for regularize_measure_dist in [False]:  # [False, True]:
#                 for regularize_measure_dist_k in [False, True]:
#                   for measure_dist_lambda in [0.0, 1.0, 100.0]:
#                     if use_curriculum:
#                       train_steps_per_cycle = 50_000
#                       time_steps_increase_per_cycle = 1
#                     else:
#                       train_steps_per_cycle = 0
#                       time_steps_increase_per_cycle = 0
#                     if skip(
#                         use_curriculum,
#                         use_pushfwd,
#                         regularize_measure_dist,
#                         regularize_measure_dist_k,
#                         measure_dist_lambda,
#                         measure_dist_type
#                     ):
#                       continue
#                     add(
#                         num_time_steps=61,
#                         train_steps=3_000_000,
#                         save_interval_steps=250_000,
#                         max_checkpoints_to_keep=3_000_000//250_000,
#                         seed=seed,
#                         batch_size=batch_size,
#                         lr=lr,
#                         measure_dist_type=measure_dist_type,
#                         train_steps_per_cycle=train_steps_per_cycle,
#                         time_steps_increase_per_cycle=time_steps_increase_per_cycle,
#                         use_curriculum=use_curriculum,
#                         use_pushfwd=use_pushfwd,
#                         regularize_measure_dist=regularize_measure_dist,
#                         regularize_measure_dist_k=regularize_measure_dist_k,
#                         measure_dist_lambda=measure_dist_lambda,
#                     )
