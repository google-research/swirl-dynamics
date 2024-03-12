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
DATA_PATH = '/datasets/hdf5/ode/lorenz63_seed_42_ntraj_train_5000_ntraj_eval_10000_ntraj_test_10000_nsteps_100000_nwarmup_100000_dt_0.001_dt_downsample_10.hdf5'
# pylint: enable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = 'lorenz63'
  # Train params
  config.train_steps = 500_000
  config.seed = 42
  config.lr = 1e-4
  config.use_lr_scheduler = False
  config.metric_aggregation_steps = 50
  config.save_interval_steps = 50_000
  config.max_checkpoints_to_keep = 10
  config.use_sobolev_norm = False
  config.order_sobolev_norm = 0
  config.mmd_bandwidth = (0.2, 0.5, 0.9, 1.3, 2., 5.,
                          9., 13., 20., 50., 90., 130.)
  config.tf_lookup_batch_size = 4096
  config.tf_lookup_num_parallel_calls = -1
  config.tf_interleaved_shuffle = False
  config.use_hdf5_reshaped = True

  # Data params
  config.batch_size = 2048
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
  config.noise_level = 0.0
  # Model params
  config.model = 'MLP'
  config.mlp_sizes = (32, 32, 3)
  config.num_lookback_steps = 1
  config.integrator = 'ExplicitEuler'
  # Update num_time_steps based on num_lookback_steps setting
  config.num_time_steps += config.num_lookback_steps - 1
  # Trainer params
  config.rollout_weighting = 'geometric'
  config.rollout_weighting_r = 0.1
  config.rollout_weighting_clip = 10e-8
  config.num_rollout_steps = 1
  config.train_steps_per_cycle = 50_000
  config.time_steps_increase_per_cycle = 0
  config.use_curriculum = False  # Sweepable
  config.use_pushfwd = False  # Sweepable
  config.measure_dist_type = 'MMD'  # Sweepable
  config.measure_dist_downsample = 1
  config.measure_dist_lambda = 0.0  # Sweepable
  config.measure_dist_k_lambda = 0.0  # Sweepable

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
def sweep(add):
  """Define param sweep."""
  # pylint: disable=line-too-long
  for seed in [1, 11, 21, 31]:
    for batch_size in [2048]:
      for lr in [1e-4]:
        for time_stride in [40]:
          for normalize in [True]:
            for measure_dist_type in  ['MMD']:
              for use_curriculum in [False, True]:
                for use_pushfwd in [False, True]:
                  for measure_dist_lambda in [0.0, 1.0]:
                    for measure_dist_k_lambda in [0.0, 1.0, 10.0, 100.0, 1000.0]:
                      for mmd_bandwidth in [
                          (0.0125, 0.025,),
                          (0.0125, 0.025, 0.025, 0.05,),
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
                            time_stride=time_stride,
                            normalize=normalize,
                            mmd_bandwidth=mmd_bandwidth,
                            measure_dist_type=measure_dist_type,
                            train_steps_per_cycle=train_steps_per_cycle,
                            time_steps_increase_per_cycle=time_steps_increase_per_cycle,
                            use_curriculum=use_curriculum,
                            use_pushfwd=use_pushfwd,
                            measure_dist_lambda=measure_dist_lambda,
                            measure_dist_k_lambda=measure_dist_k_lambda,
                        )
  # pylint: enable=line-too-long
