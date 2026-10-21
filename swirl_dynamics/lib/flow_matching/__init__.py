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

"""Flow matching and stochastic interpolant library."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

from swirl_dynamics.lib.flow_matching.interpolants import (
    Interpolant,
    LinearInterpolant,
    LinearInterpolantSinusoidalNoise,
    RectifiedFlow,
    StochasticInterpolant,
    TrigonometricInterpolant,
)
from swirl_dynamics.lib.flow_matching.losses import (
    StochasticInterpolantLossFn,
    denoising_loss,
    score_loss,
    velocity_loss,
)
from swirl_dynamics.lib.flow_matching.mean_flow import (
    ConditionalMeanFlowModel,
    ConditionalSymmetricMeanFlowModel,
    MeanFlowModel,
    logit_normal_dist,
    time_sampler_mean_flow,
)
from swirl_dynamics.lib.flow_matching.models import (
    ConditionalStochasticInterpolantModel,
    RescaledUnet,
    RescaledUnet3d,
    StochasticInterpolantCRPSModel,
    StochasticInterpolantFlowScoreModel,
    StochasticInterpolantModel,
    TimeRescaled,
    cond_sample_from_shape,
)

__all__ = [
    "ConditionalMeanFlowModel",
    "ConditionalStochasticInterpolantModel",
    "ConditionalSymmetricMeanFlowModel",
    "Interpolant",
    "LinearInterpolant",
    "LinearInterpolantSinusoidalNoise",
    "MeanFlowModel",
    "RectifiedFlow",
    "RescaledUnet",
    "RescaledUnet3d",
    "StochasticInterpolant",
    "StochasticInterpolantCRPSModel",
    "StochasticInterpolantFlowScoreModel",
    "StochasticInterpolantLossFn",
    "StochasticInterpolantModel",
    "TimeRescaled",
    "TrigonometricInterpolant",
    "cond_sample_from_shape",
    "denoising_loss",
    "logit_normal_dist",
    "score_loss",
    "time_sampler_mean_flow",
    "velocity_loss",
]
