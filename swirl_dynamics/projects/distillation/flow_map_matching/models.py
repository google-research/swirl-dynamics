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

"""Backwards-compatible re-export shim for graduated flow-map models."""

from collections.abc import Callable
from typing import Any

from clu import metrics as clu_metrics
import jax
from swirl_dynamics.lib.flow_map import backbones as _backbones
from swirl_dynamics.lib.flow_map import flow_map as _flow_map

Array = _flow_map.Array
ArrayLike = jax.typing.ArrayLike
ArrayShape = _flow_map.ArrayShape
CondDict = _flow_map.CondDict
ConditionalLagrangianFlowMapModel = _flow_map.ConditionalLagrangianFlowMapModel
ConditionalLagrangianSelfDistilledFlowMapModel = (
    _flow_map.ConditionalLagrangianSelfDistilledFlowMapModel
)
LagrangianFlowMapModel = _flow_map.LagrangianFlowMapModel
LagrangianSelfDistilledFlowMapModel = (
    _flow_map.LagrangianSelfDistilledFlowMapModel
)
Metrics = clu_metrics.Collection
PyTree = Any
RescaledFlowMapUNet = _backbones.RescaledFlowMapUNet
RescaledMeanFlowUNet = _backbones.RescaledMeanFlowUNet
SamplingFn = Callable[[ArrayLike, ArrayShape, Array], Array]
ShapeDict = _flow_map.ShapeDict
VariableDict = _flow_map.VariableDict
cond_sample_from_shape = _flow_map.cond_sample_from_shape

__all__ = [
    "Array",
    "ArrayLike",
    "ArrayShape",
    "CondDict",
    "ConditionalLagrangianFlowMapModel",
    "ConditionalLagrangianSelfDistilledFlowMapModel",
    "LagrangianFlowMapModel",
    "LagrangianSelfDistilledFlowMapModel",
    "Metrics",
    "PyTree",
    "RescaledFlowMapUNet",
    "RescaledMeanFlowUNet",
    "SamplingFn",
    "ShapeDict",
    "VariableDict",
    "cond_sample_from_shape",
]
