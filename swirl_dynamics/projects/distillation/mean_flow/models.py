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

r"""Backwards-compatible re-export shim for graduated MeanFlow models."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

from collections.abc import Callable

from clu import metrics as clu_metrics
import jax
from swirl_dynamics.lib.flow_matching.mean_flow import (
    Array,
    ArrayShape,
    CondDict,
    ConditionalMeanFlowModel,
    ConditionalSymmetricMeanFlowModel,
    DTypeLike,
    MeanFlowModel,
    PyTree,
    ShapeDict,
    VariableDict,
    cond_sample_from_shape,
    logit_normal_dist,
    time_sampler_mean_flow,
)
from swirl_dynamics.projects.distillation.flow_map_matching.models import RescaledMeanFlowUNet

ArrayLike = jax.typing.ArrayLike
Metrics = clu_metrics.Collection
SamplingFn = Callable[[ArrayLike, ArrayShape, Array], Array]

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
