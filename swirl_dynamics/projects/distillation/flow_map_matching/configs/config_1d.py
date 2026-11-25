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

"""Configuration for the 1-D flow-map distillation and self-distillation demo.

Trains a lightweight `lib.flow_matching` teacher velocity field on a synthetic
1-D periodic waveform dataset, distills it with `LagrangianFlowMapModel`, and
compares against teacher-free `LagrangianSelfDistilledFlowMapModel` across
1-, 2-, and 4-NFE generation.
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns the default configuration for the 1-D flow-map matching demo."""
  config = ml_collections.ConfigDict()
  config.experiment_name = "flow_map_matching_1d"
  config.seed = 42

  # Synthetic 1-D periodic dataset.
  config.grid_size = 32
  config.num_channels = 1
  config.num_train_samples = 384
  config.num_eval_samples = 64
  config.batch_size = 32

  # Compact 1-D UNet widths for teacher, distilled map, and LSD map.
  config.num_channels_unet = (16, 32)
  config.downsample_ratio = (2, 2)
  config.num_blocks = 1
  config.noise_embed_dim = 16

  # Training & EMA hyperparameters.
  config.teacher_steps = 60
  config.distill_steps = 60
  config.lsd_steps = 60
  config.lsd_weight_v = 0.5
  config.learning_rate = 3e-3
  config.ema_decay = 0.98

  # Evaluation step counts.
  config.teacher_ode_steps = 32
  config.eval_num_steps = (1, 2, 4)

  return config
