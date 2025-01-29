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

"""Trainer for denoising model with sampling CRPS evaluation."""

import functools

import clu.metrics as clu_metrics
from swirl_dynamics.projects.probabilistic_diffusion import trainers as dfn_trainers
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5 import models


class DenoisingTrainer(
    dfn_trainers.DistributedDenoisingTrainer[
        models.DenoisingModel, dfn_trainers.TrainState
    ]
):
  """Multi-device trainer for the downscaling denoising model."""

  @functools.cached_property
  def EvalMetrics(self):
    denoising_metrics = {
        f"eval_denoise_lvl{i}": clu_metrics.Average.from_output(f"sigma_lvl{i}")
        for i in range(self.model.num_eval_noise_levels)
    }
    sampling_metrics = {
        "eval_mean_crps": clu_metrics.Average.from_output("mean_crps"),
        "eval_plot_data": clu_metrics.CollectingMetric.from_outputs(
            ("example_sample", "example_input", "example_obs")
        ),
    }
    likelihood_metrics = {
        "eval_ll_per_dim": clu_metrics.Average.from_output(
            "sample_log_likelihood_per_dim"
        ),
        "eval_ll_per_dim_std": clu_metrics.Std.from_output(
            "sample_log_likelihood_per_dim"
        ),
    }
    return clu_metrics.Collection.create(
        **denoising_metrics, **sampling_metrics, **likelihood_metrics
    )
