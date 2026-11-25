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

"""Backwards-compatible re-export shim for graduated flow-map backbones."""

from collections.abc import Mapping
from typing import Any

from clu import metrics as clu_metrics
from swirl_dynamics.lib.flow_map import backbones as _backbones
from swirl_dynamics.lib.flow_map import samplers as _samplers

Array = _backbones.Array
ArrayShape = _backbones.ArrayShape
CondDict = _backbones.CondDict
DTypeLike = _backbones.DTypeLike
FlowMapUNet = _backbones.FlowMapUNet
FourierEmbedding = _backbones.FourierEmbedding
MergeChannelCond = _backbones.MergeChannelCond
Metrics = clu_metrics.Collection
PrecisionLike = _backbones.PrecisionLike
PyTree = Any
ShapeDict = Mapping[str, Any]
logit_normal_dist = _samplers.logit_normal_dist
time_sampler_mean_flow = _samplers.time_sampler_flow_map

__all__ = [
    "Array",
    "ArrayShape",
    "CondDict",
    "DTypeLike",
    "FlowMapUNet",
    "FourierEmbedding",
    "MergeChannelCond",
    "Metrics",
    "PrecisionLike",
    "PyTree",
    "ShapeDict",
    "logit_normal_dist",
    "time_sampler_mean_flow",
]
