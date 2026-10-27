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

"""Configuration of the 1-D Kuramoto-Sivashinsky optimal transport debiasing.

Domain A holds snapshots of a low-order finite-volume solver and domain B
snapshots of a pseudo-spectral solver sub-sampled to the same resolution (see
`cycle_gan/dataloaders.py`). An entropic optimal transport map (Pooladian and
Niles-Weed, 2021) or centered debiased map (Pooladian, Cuturi, and Niles-Weed,
2022) is fitted between empirical training snapshots and evaluated on held-out
snapshots.

Usage:
  main_ot_debiasing --config=configs/config_ks_1d.py --workdir=/tmp/ot_ks
    --config.dataset_path=/tmp/KS_finite_volumes_vs_pseudo_spectral.hdf5
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns the configuration of the KS optimal transport experiment."""
  config = ml_collections.ConfigDict()
  config.experiment_name = "ot_ks_1d"

  # Data. `dataset_path` is a local copy of `dataloaders.KS_DATASET_URL`
  # (`KS_finite_volumes_vs_pseudo_spectral.hdf5`); the empty string selects a
  # deterministic synthetic KS dataset for smoke tests.
  config.dataset_path = ""
  config.sample_length = 48
  config.downsample_factor = 4
  config.num_test_trajectories = 16
  config.max_train_samples = 4096
  config.seed = 42

  # Entropic optimal transport solver configuration. When `epsilon <= 0.0`,
  # `experiment.resolve_epsilon` selects `epsilon_scale * median(C)`.
  config.enable_x64 = True
  config.epsilon = 0.0
  config.epsilon_scale = 0.01
  config.num_iterations = 1000
  config.inner_iterations = 10
  config.threshold = 1e-3
  config.criterion = "marginal"
  config.use_debiased_map = False

  # Evaluation parameters.
  config.eval_batch_size = 256
  config.sinkhorn_divergence_max_samples = 512
  return config
