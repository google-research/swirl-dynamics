# Copyright 2026 The swirl_dynamics Authors.
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

"""Configuration of the 1-D Kuramoto-Sivashinsky debiasing example.

Domain A holds snapshots of a low-order finite-volume solver and domain B
snapshots of a pseudo-spectral solver sub-sampled to the same resolution (see
`dataloaders.py`). The networks are the library CycleGAN networks with the
debiasing choices discussed in the package README: no normalisation layers
(amplitude-aware discriminators), biases on, an identity output head instead
of `tanh`, circular padding, the paper's loss weights and optimiser.

Usage:
  main_train --config=configs/config_ks_1d.py --workdir=/tmp/cycle_gan_ks
    --config.dataset_path=/tmp/KS_finite_volumes_vs_pseudo_spectral.hdf5
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns the configuration of the KS experiment."""
  config = ml_collections.ConfigDict()
  config.experiment_name = "cycle_gan_ks_1d"

  # Data. `dataset_path` is a local copy of `dataloaders.KS_DATASET_URL`; the
  # empty string selects a small deterministic synthetic dataset, which is
  # meant for smoke tests only.
  config.dataset_path = ""
  config.sample_length = 48
  config.downsample_factor = 4
  config.num_test_trajectories = 16
  config.normalize = True
  config.batch_size = 32
  config.eval_batch_size = 32
  config.seed = 42

  # Networks (keyword arguments of the library `Generator`/`Discriminator`;
  # the identity output head is set in code).
  config.generator = ml_collections.ConfigDict(
      dict(
          output_nc=1,
          ngf=16,
          n_res_blocks=2,
          n_res_blocks_level=0,
          dropout_rate=0.0,
          upsample_mode="deconv",
          n_downsample_layers=2,
          n_upsample_layers=2,
          use_skips=True,
          use_global_skip=True,
          padding="CIRCULAR",
          padding_transpose="CIRCULAR",
          norm_type="none",
      )
  )
  config.discriminator = ml_collections.ConfigDict(
      dict(
          base_features=16,
          n_layers=2,
          kernel_size=(5,),
          padding="CIRCULAR",
          use_bias=True,
          norm_type="none",
      )
  )

  # Losses (paper defaults; see `models.CycleGANModelConfig`).
  config.gan_mode = "lsgan"
  config.lambda_a = 10.0
  config.lambda_b = 10.0
  config.lambda_id = 0.5
  config.lambda_gp = 10.0
  config.use_crossed_dis_loss = False

  # Optimisation: Adam(2e-4, beta1=0.5) with a linear decay to zero over the
  # second half of training, a 50-sample replay pool, no EMA.
  config.num_train_steps = 20_000
  config.learning_rate = 2e-4
  config.beta1 = 0.5
  config.decay_start_fraction = 0.5
  config.n_critic = 1
  config.pool_size = 50
  config.ema_decay = ml_collections.config_dict.placeholder(float)

  # Bookkeeping.
  config.metric_aggregation_steps = 100
  config.eval_every_steps = 1_000
  config.num_batches_per_eval = 4
  config.save_interval_steps = 1_000
  config.max_checkpoints_to_keep = 3
  return config
