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

"""Backwards-compatible re-export shim for graduated MeanFlow trainers."""

from swirl_dynamics.lib.flow_map import trainers as _trainers

Array = _trainers.Array
Collection = _trainers.Collection
DistributedMeanFlowTrainer = _trainers.DistributedMeanFlowTrainer
MeanFlowTrainState = _trainers.MeanFlowTrainState
MeanFlowTrainer = _trainers.MeanFlowTrainer
TrainState = _trainers.TrainState
VariableDict = _trainers.VariableDict

__all__ = [
    "Array",
    "Collection",
    "DistributedMeanFlowTrainer",
    "MeanFlowTrainState",
    "MeanFlowTrainer",
    "TrainState",
    "VariableDict",
]
