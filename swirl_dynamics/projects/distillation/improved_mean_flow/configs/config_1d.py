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

"""Configuration for the 1-D class-conditional Improved Mean Flow demo.

Trains a compact `InContextMeanFlowTransformer` with an auxiliary `v`-head and
flexible CFG scale/interval conditioning (`cfg_mode="interval"`) on synthetic
1-D class-conditional periodic waveforms.
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns the default configuration for the 1-D iMF demo."""
  config = ml_collections.ConfigDict()
  config.experiment_name = "improved_mean_flow_1d"
  config.seed = 42

  # Synthetic 1-D class-conditional periodic dataset.
  config.grid_size = 32
  config.num_channels = 1
  config.num_classes = 3
  config.num_train_samples = 384
  config.num_eval_samples = 64
  config.batch_size = 32

  # InContextMeanFlowTransformer architecture (compact CPU-friendly width).
  config.hidden_size = 32
  config.depth = 3
  config.num_heads = 4
  config.patch_size = 4
  config.aux_head_depth = 1
  config.residual_scale_init = 0.0
  config.zero_init_output = True

  # Improved Mean Flow objective & flexible CFG settings.
  config.diagonal_sampling_rate = 0.5
  config.norm_power = 1.0
  config.norm_regularization = 1e-2
  config.aux_v_loss_weight = 1.0
  config.cfg_mode = "interval"
  config.class_dropout_rate = 0.1
  config.cfg_scale_min = 1.0
  config.cfg_scale_max = 6.0
  config.cfg_scale_beta = 1.0
  config.cfg_t_min_range = (0.0, 0.5)
  config.cfg_t_max_range = (0.5, 1.0)

  # Optimization & EMA.
  config.learning_rate = 3e-3
  config.ema_decay = 0.99
  config.num_train_steps = 80

  # Evaluation sweeps.
  config.eval_cfg_scales = (1.0, 2.0, 4.0)
  config.eval_cfg_intervals = ((0.0, 1.0), (0.2, 0.8))
  config.eval_num_steps = (1, 2, 4)

  return config
