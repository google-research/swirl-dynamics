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

"""Configuration for the 1-D class-conditional MeanFlow demo.

Trains a compact `RescaledMeanFlowUNet` with `MergeCategoricalEmbCond` from
scratch via `flow_map.ConditionalMeanFlowModel` and `flow_map.MeanFlowTrainer`
on synthetic 1-D class-conditional bimodal waveforms, comparing 1-, 2-, and
4-NFE backward generation.
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns the default configuration for the 1-D MeanFlow demo."""
  config = ml_collections.ConfigDict()
  config.experiment_name = "mean_flow_1d"
  config.seed = 42

  # Synthetic 1-D class-conditional dataset.
  config.grid_size = 32
  config.num_channels = 1
  config.num_classes = 3
  config.num_train_samples = 384
  config.num_eval_samples = 64
  config.batch_size = 32

  # Compact RescaledMeanFlowUNet backbone.
  config.num_channels_unet = (16, 32)
  config.downsample_ratio = (2, 2)
  config.num_blocks = 1
  config.noise_embed_dim = 16
  config.features_embedding = 16

  # MeanFlow objective & time sampling.
  config.logit_normal_mean = -0.4
  config.logit_normal_std = 1.0
  config.diagonal_sampling_rate = 0.25
  config.norm_power = 1.0
  config.norm_regularization = 1e-2

  # Optimization & EMA.
  config.learning_rate = 3e-3
  config.ema_decay = 0.98
  config.num_train_steps = 80

  # Multi-step backward sampling evaluation.
  config.eval_num_steps = (1, 2, 4)

  return config
