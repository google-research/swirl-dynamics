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

"""Backwards-compatible re-export shim for graduated MeanFlow models."""

from collections.abc import Callable
from typing import Any

from clu import metrics as clu_metrics
import jax
from swirl_dynamics.lib.flow_map import backbones as _backbones
from swirl_dynamics.lib.flow_map import mean_flow as _mean_flow
from swirl_dynamics.lib.flow_map import samplers as _samplers

Array = _mean_flow.Array
ArrayLike = jax.typing.ArrayLike
ArrayShape = _mean_flow.ArrayShape
CondDict = _mean_flow.CondDict
ConditionalMeanFlowModel = _mean_flow.ConditionalMeanFlowModel
ConditionalSymmetricMeanFlowModel = _mean_flow.ConditionalSymmetricMeanFlowModel
DTypeLike = _samplers.DTypeLike
MeanFlowModel = _mean_flow.MeanFlowModel
Metrics = clu_metrics.Collection
PyTree = Any
RescaledMeanFlowUNet = _backbones.RescaledMeanFlowUNet
SamplingFn = Callable[[ArrayLike, ArrayShape, Array], Array]
ShapeDict = _mean_flow.ShapeDict
VariableDict = _mean_flow.VariableDict
cond_sample_from_shape = _mean_flow.cond_sample_from_shape
logit_normal_dist = _samplers.logit_normal_dist
time_sampler_mean_flow = _samplers.time_sampler_mean_flow

__all__ = [
    "Array",
    "ArrayLike",
    "ArrayShape",
    "CondDict",
    "ConditionalMeanFlowModel",
    "ConditionalSymmetricMeanFlowModel",
    "DTypeLike",
    "MeanFlowModel",
    "Metrics",
    "PyTree",
    "RescaledMeanFlowUNet",
    "SamplingFn",
    "ShapeDict",
    "VariableDict",
    "cond_sample_from_shape",
    "logit_normal_dist",
    "time_sampler_mean_flow",
]
