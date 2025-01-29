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

"""PyGrain transforms for the input pipelines."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

import grain.python as pygrain
import numpy as np

FlatFeatures = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Standardize(pygrain.MapTransform):
  """Standardize variables pixel-wise using pre-computed mean and std."""

  input_fields: Sequence[str]
  mean: Mapping[str, np.ndarray]
  std: Mapping[str, np.ndarray]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = (features[field] - self.mean[field]) / self.std[field]
    return features


@dataclasses.dataclass(frozen=True)
class Concatenate(pygrain.MapTransform):
  """Creates a new field by concatenating selected fields."""

  input_fields: Sequence[str]
  output_field: str
  axis: int
  remove_inputs: bool = True

  def map(self, features: FlatFeatures) -> FlatFeatures:
    arrays = [features[f] for f in self.input_fields]
    features[self.output_field] = np.concatenate(arrays, axis=self.axis)
    if self.remove_inputs:
      for f in self.input_fields:
        del features[f]
    return features


@dataclasses.dataclass(frozen=True)
class Rot90(pygrain.MapTransform):
  """Rotate selected field 90 degrees counterclockwise."""

  input_fields: Sequence[str]
  k: int = 1
  axes: tuple[int, int] = (0, 1)

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = np.rot90(features[field], k=self.k, axes=self.axes)
    return features


@dataclasses.dataclass(frozen=True)
class RandomMaskout(pygrain.RandomMapTransform):
  """Randomly mask out selected fields with a given value."""

  input_fields: Sequence[str]
  probability: float = 0.1
  fill_value: float = 0.0

  def random_map(
      self, features: FlatFeatures, rng: np.random.Generator
  ) -> FlatFeatures:
    if rng.choice([True, False], p=[self.probability, 1 - self.probability]):
      for field in self.input_fields:
        features[field] = self.fill_value * np.ones_like(features[field])
    return features


@dataclasses.dataclass(frozen=True)
class AssembleCondDict(pygrain.MapTransform):
  """Assemble fields into a conditional dictionary."""

  cond_fields: Sequence[str]
  # Default means cond is concat'ed in channel (denoiser module assumption).
  prefix: str = "channel:"
  remove_original: bool = True

  def map(self, features: FlatFeatures) -> FlatFeatures:
    if "cond" not in features:
      features["cond"] = {}
    for field in self.cond_fields:
      features["cond"][f"{self.prefix}{field}"] = features[field]
      if self.remove_original:
        del features[field]
    return features
