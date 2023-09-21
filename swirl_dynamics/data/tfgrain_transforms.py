# Copyright 2023 The swirl_dynamics Authors.
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

"""Module for reusable TfGrain transformations."""

from collections.abc import MutableMapping
import dataclasses
import grain.tensorflow as tfgrain
import jax
import tensorflow as tf

Array = jax.Array
FeatureDict = MutableMapping[str, tf.Tensor]
Scalar = float | int


def _check_valid_scale_range(
    scale_range: tuple[Scalar, Scalar], name: str
) -> None:
  """Checks if a given scale range is valid."""
  if scale_range[0] >= scale_range[1]:
    raise ValueError(
        f"Lower bound of {name} ({scale_range[0]}) must be strictly smaller"
        f" than its upper bound ({scale_range[1]})"
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearRescale(tfgrain.MapTransform):
  """Apply linear rescaling to a feature."""

  feature_name: str
  input_range: tuple[Scalar, Scalar]
  output_range: tuple[Scalar, Scalar]

  def __post_init__(self):
    _check_valid_scale_range(self.input_range, name="input_range")
    _check_valid_scale_range(self.output_range, name="output_range")

  def map(self, features: FeatureDict) -> FeatureDict:
    normalized = (features[self.feature_name] - self.input_range[0]) / (
        self.input_range[1] - self.input_range[0]
    )
    rescaled = (
        normalized * (self.output_range[1] - self.output_range[0])
        + self.output_range[0]
    )
    features[self.feature_name] = rescaled
    return features


@dataclasses.dataclass(frozen=True, kw_only=True)
class Normalize(tfgrain.MapTransform):
  """Apply normalization to a feature."""

  feature_name: str
  mean: Array
  std: Array

  def map(self, features: FeatureDict) -> FeatureDict:
    normalized = (features[self.feature_name] - self.mean) / self.std
    features[self.feature_name] = normalized
    return features


@dataclasses.dataclass(frozen=True, kw_only=True)
class RandomSection(tfgrain.RandomMapTransform):
  """Samples a random section in a given trajectory.

  Sampling always happens along the leading dimension. First a start index is
  randomly selected amongst all permissible locations and the sample is taken to
  be the contiguous section immediately after and including the start index,
  with the specified number of steps and stride.

  Attributes:
    feature_names: names of the features to be sampled. All sampled features
      share the same start index (besides number of steps and stride). They must
      have the same dimension in the leading axis.
    num_steps: the number of steps in the sample.
    stride: the stride (i.e. downsample) in the sampled section wrt the original
      features.
  """

  feature_names: tuple[str, ...]
  num_steps: int
  stride: int = 1

  def random_map(self, features: FeatureDict, seed: tf.Tensor) -> FeatureDict:
    total_length = features[self.feature_names[0]].shape.as_list()[0]
    sample_length = self.stride * (self.num_steps - 1) + 1

    for name in self.feature_names[1:]:
      feature_length = features[name].shape.as_list()[0]
      if feature_length != total_length:
        raise ValueError(
            "Features must have the same dimension along axis 0:"
            f" {self.feature_names[0]} ({total_length}) vs."
            f" {name} ({feature_length})"
        )

    if sample_length > total_length:
      raise ValueError(
          f"Not enough steps [{total_length}] "
          f"for desired sample length [{sample_length}] "
          f"= stride [{self.stride}] * (num_steps [{self.num_steps}] - 1) + 1"
      )
    elif sample_length == total_length:
      start_idx, end_idx = 0, total_length
    else:
      start_idx = tf.random.stateless_uniform(
          shape=(),
          seed=seed,
          maxval=total_length - sample_length + 1,
          dtype=tf.int32,
      )
      end_idx = start_idx + sample_length

    for name in self.feature_names:
      features[name] = features[name][start_idx : end_idx : self.stride]
    return features


@dataclasses.dataclass(frozen=True, kw_only=True)
class Split(tfgrain.MapTransform):
  """Splits a tensor feature into multiple sub tensors.

  Attributes:
    feature_name: name of the feature to be split.
    split_sizes: the sizes of each output feature along `axis`. Must sum to the
      dimension of the presplit feature along `axis`.
    split_names: the name of the output features. Must have the same length as
      the `split_sizes`.
    axis: the axis along which splitting happens.
    keep_presplit: whether to keep the presplit feature in the processed batch.
  """

  feature_name: str
  split_sizes: tuple[int, ...]
  split_names: tuple[str, ...]
  axis: int = 0
  keep_presplit: bool = False

  def __post_init__(self) -> None:
    if len(self.split_names) != len(self.split_sizes):
      raise ValueError(
          f"Length of `split_sizes` [{self.split_sizes}] must match "
          f"that of `split_names` [{self.split_names}]"
      )

  def map(self, features: FeatureDict) -> FeatureDict:
    feature = features[self.feature_name]
    splits = tf.split(feature, self.split_sizes, axis=self.axis)
    for split_name, split_value in zip(self.split_names, splits):
      features[split_name] = split_value
    if not self.keep_presplit:
      features.pop(self.feature_name)
    return features
