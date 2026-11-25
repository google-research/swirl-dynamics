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

"""Two-time flow-map matching and mean-flow library."""

from swirl_dynamics.lib.flow_map import backbones
from swirl_dynamics.lib.flow_map import flow_map
from swirl_dynamics.lib.flow_map import mean_flow
from swirl_dynamics.lib.flow_map import samplers
from swirl_dynamics.lib.flow_map import trainers

CondDict = samplers.CondDict
ConditionalLagrangianFlowMapModel = flow_map.ConditionalLagrangianFlowMapModel
ConditionalLagrangianSelfDistilledFlowMapModel = (
    flow_map.ConditionalLagrangianSelfDistilledFlowMapModel
)
ConditionalLagrangianSelfDistilledFlowMapTrainer = (
    trainers.ConditionalLagrangianSelfDistilledFlowMapTrainer
)
ConditionalMeanFlowModel = mean_flow.ConditionalMeanFlowModel
ConditionalSymmetricMeanFlowModel = mean_flow.ConditionalSymmetricMeanFlowModel
DistributedConditionalLagrangianSelfDistilledFlowMapTrainer = (
    trainers.DistributedConditionalLagrangianSelfDistilledFlowMapTrainer
)
DistributedLagrangianFlowMapTrainer = (
    trainers.DistributedLagrangianFlowMapTrainer
)
DistributedLagrangianSelfDistilledFlowMapTrainer = (
    trainers.DistributedLagrangianSelfDistilledFlowMapTrainer
)
DistributedMeanFlowTrainer = trainers.DistributedMeanFlowTrainer
EmaFlowMapTrainer = trainers.EmaFlowMapTrainer
FlowMapTrainState = trainers.FlowMapTrainState
FlowMapUNet = backbones.FlowMapUNet
FourierEmbedding = backbones.FourierEmbedding
LagrangianFlowMapModel = flow_map.LagrangianFlowMapModel
LagrangianFlowMapTrainer = trainers.LagrangianFlowMapTrainer
LagrangianSelfDistilledFlowMapModel = (
    flow_map.LagrangianSelfDistilledFlowMapModel
)
LagrangianSelfDistilledFlowMapTrainer = (
    trainers.LagrangianSelfDistilledFlowMapTrainer
)
MeanFlowModel = mean_flow.MeanFlowModel
MeanFlowTrainState = trainers.MeanFlowTrainState
MeanFlowTrainer = trainers.MeanFlowTrainer
MergeCategoricalEmbCond = backbones.MergeCategoricalEmbCond
MergeGuidanceEmbCond = backbones.MergeGuidanceEmbCond
RescaledFlowMapUNet = backbones.RescaledFlowMapUNet
RescaledMeanFlowUNet = backbones.RescaledMeanFlowUNet
ShapeDict = flow_map.ShapeDict
adaptive_weighted_loss = samplers.adaptive_weighted_loss
compute_compound_velocity = samplers.compute_compound_velocity
compute_guided_velocity_target = samplers.compute_guided_velocity_target
cond_sample_from_shape = flow_map.cond_sample_from_shape
logit_normal_dist = samplers.logit_normal_dist
sample_backward_mean_flow = samplers.sample_backward_mean_flow
sample_cfg_interval = samplers.sample_cfg_interval
sample_cfg_scale = samplers.sample_cfg_scale
sample_forward_flow_map = samplers.sample_forward_flow_map
time_sampler_flow_map = samplers.time_sampler_flow_map
time_sampler_mean_flow = samplers.time_sampler_mean_flow
validate_flow_map_inputs = backbones.validate_flow_map_inputs

__all__ = [
    "CondDict",
    "ConditionalLagrangianFlowMapModel",
    "ConditionalLagrangianSelfDistilledFlowMapModel",
    "ConditionalLagrangianSelfDistilledFlowMapTrainer",
    "ConditionalMeanFlowModel",
    "ConditionalSymmetricMeanFlowModel",
    "DistributedConditionalLagrangianSelfDistilledFlowMapTrainer",
    "DistributedLagrangianFlowMapTrainer",
    "DistributedLagrangianSelfDistilledFlowMapTrainer",
    "DistributedMeanFlowTrainer",
    "EmaFlowMapTrainer",
    "FlowMapTrainState",
    "FlowMapUNet",
    "FourierEmbedding",
    "LagrangianFlowMapModel",
    "LagrangianFlowMapTrainer",
    "LagrangianSelfDistilledFlowMapModel",
    "LagrangianSelfDistilledFlowMapTrainer",
    "MeanFlowModel",
    "MeanFlowTrainState",
    "MeanFlowTrainer",
    "MergeCategoricalEmbCond",
    "MergeGuidanceEmbCond",
    "RescaledFlowMapUNet",
    "RescaledMeanFlowUNet",
    "ShapeDict",
    "adaptive_weighted_loss",
    "compute_compound_velocity",
    "compute_guided_velocity_target",
    "cond_sample_from_shape",
    "logit_normal_dist",
    "sample_backward_mean_flow",
    "sample_cfg_interval",
    "sample_cfg_scale",
    "sample_forward_flow_map",
    "time_sampler_flow_map",
    "time_sampler_mean_flow",
    "validate_flow_map_inputs",
]
