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

# pylint: disable=line-too-long
DATA_PATH = '/datasets/gcs_staging/hdf5/pde/1d/ks_trajectories.hdf5'
# pylint: enable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = 'ks_1d'
  # Train params
  config.train_steps = 300_000
  config.seed = 42
  config.lr = 1e-4
  config.use_lr_scheduler = True
  config.metric_aggregation_steps = 50
  config.save_interval_steps = 30_000
  config.max_checkpoints_to_keep = 10
  # Data params
  config.batch_size = 128
  # num_time_steps is length of ground truth trajectories in each batch from
  # dataloader (this differs from num_time_steps in Trainer.preprocess_batch
  # functions that corresponds to the len of ground truth trajectories to
  # actually pass to model, which can vary during training).
  config.num_time_steps = 6
  config.time_stride = 1  # factor for downsampling time dim of ground truth
  config.dataset_path = DATA_PATH
  config.spatial_downsample_factor = 1
  config.normalize = False
  config.add_noise = False
  config.use_sobolev_norm = False
  config.order_sobolev_norm = 1
  config.noise_level = 0.0
  config.mmd_bandwidth = (0.2, 0.5, 0.9, 1.3, 2., 5.,
                          9., 13., 20., 50., 90., 130.)
  config.tf_lookup_batch_size = 4096
  config.tf_lookup_num_parallel_calls = -1
  config.tf_interleaved_shuffle = False
  config.use_hdf5_reshaped = True

  # Model params
  config.model = 'PeriodicConvNetModel'  # 'Fno'
  # TODO: Split CNN and FNO into separate configs
  ########### PeriodicConvNetModel ################
  config.latent_dim = 48
  config.num_levels = 4
  config.num_processors = 4
  config.encoder_kernel_size = (5,)
  config.decoder_kernel_size = (5,)
  config.processor_kernel_size = (5,)
  config.padding = 'CIRCULAR'
  config.is_input_residual = True
  ########### FNO ################
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
  config.num_time_steps += config.get_ref('num_lookback_steps') - 1
  if config.get_ref('num_lookback_steps') > 1:
    config.integrator = 'MultiStepDirect'
  else:
    config.integrator = 'OneStepDirect'
  # Trainer params
  config.rollout_weighting = 'geometric'
  config.rollout_weighting_r = 0.9
  config.rollout_weighting_clip = 10e-4
  config.num_rollout_steps = 1
  config.train_steps_per_cycle = 60_000
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
  if measure_dist_lambda > 0.0 and measure_dist_k_lambda == 0.0:
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
  # pylint: disable=line-too-long
  for seed in [1, 11, 21, 31,]:
    for normalize in [True]:
      for model in ['PeriodicConvNetModel']:
        for batch_size in [32]:  # [32, 64, 128, 256]:
          for lr in [5e-4]:
            for use_curriculum in [True, False]:
              for use_pushfwd in [True, False]:
                for measure_dist_type in ['MMD']:
                  for measure_dist_lambda in [0.0]:
                    for measure_dist_k_lambda in [0.0, 1.0, 10.0, 100.0]:
                      for mmd_bandwidth in [
                          (0.025, 0.05,),
                          (0.025, 0.05, 0.1, 0.2,),
                          (0.025, 0.05, 0.1, 0.2, 0.4, 0.8),
                          (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
                          (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8),
                          (0.4, 0.8, 1.6, 3.2, 6.4, 12.8),
                          (1.6, 3.2, 6.4, 12.8),
                          (6.4, 12.8),
                      ]:
                        if use_curriculum:
                          train_steps_per_cycle = 60_000
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
                            normalize=normalize,
                            model=model,
                            batch_size=batch_size,
                            lr=lr,
                            mmd_bandwidth=mmd_bandwidth,
                            measure_dist_type=measure_dist_type,
                            train_steps_per_cycle=train_steps_per_cycle,
                            time_steps_increase_per_cycle=time_steps_increase_per_cycle,
                            use_curriculum=use_curriculum,
                            use_pushfwd=use_pushfwd,
                            measure_dist_lambda=measure_dist_lambda,
                            measure_dist_k_lambda=measure_dist_k_lambda,
                        )
