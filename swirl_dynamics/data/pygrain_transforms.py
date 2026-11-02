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

"""Module for reusable PyGrain transformations.

The transformations operate on dictionaries of NumPy arrays (e.g. the records
produced by `pygrain_sources.ArrayDataSource`) and never modify the input arrays
in place, so they are safe to use with sources that return views of in-memory
data.
"""

from collections.abc import MutableMapping, Sequence
import dataclasses
from typing import Any

import grain.python as pygrain
import numpy as np
import numpy.typing as npt

FeatureDict = MutableMapping[str, Any]
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
class LinearRescale(pygrain.MapTransform):
  """Apply linear rescaling to a feature.

  Attributes:
    feature_name: Name of the feature to be rescaled.
    input_range: Strict `(min, max)` interval mapped onto `output_range`.
    output_range: Strict `(min, max)` target interval after rescaling.
  """

  feature_name: str
  input_range: tuple[Scalar, Scalar]
  output_range: tuple[Scalar, Scalar]

  def __post_init__(self):
    _check_valid_scale_range(self.input_range, name="input_range")
    _check_valid_scale_range(self.output_range, name="output_range")

  def map(self, features: FeatureDict) -> FeatureDict:
    """See base class."""
    feature = np.asarray(features[self.feature_name])
    if np.issubdtype(feature.dtype, np.floating):
      in_low, in_high = np.asarray(self.input_range, dtype=feature.dtype)
      out_low, out_high = np.asarray(self.output_range, dtype=feature.dtype)
    else:
      in_low, in_high = self.input_range
      out_low, out_high = self.output_range
    normalized = (feature - in_low) / (in_high - in_low)
    features[self.feature_name] = (
        normalized * (out_high - out_low) + out_low
    )
    return features


@dataclasses.dataclass(frozen=True, kw_only=True)
class Normalize(pygrain.MapTransform):
  """Apply normalization to a feature.

  Attributes:
    feature_name: name of the feature to be normalized.
    mean: mean subtracted from the feature (broadcast against it). Converted to
      a NumPy array (of the feature dtype, if it is floating) before use.
    std: standard deviation dividing the feature (broadcast against it).
      Converted like `mean`.
  """

  feature_name: str
  mean: npt.ArrayLike
  std: npt.ArrayLike

  def map(self, features: FeatureDict) -> FeatureDict:
    """See base class."""
    feature = np.asarray(features[self.feature_name])
    if np.issubdtype(feature.dtype, np.floating):
      # Like the former tf_grain transform, the statistics are converted to the
      # dtype of the feature (e.g. float32) before normalizing.
      mean = np.asarray(self.mean, dtype=feature.dtype)
      std = np.asarray(self.std, dtype=feature.dtype)
    else:
      mean, std = np.asarray(self.mean), np.asarray(self.std)
    features[self.feature_name] = (feature - mean) / std
    return features


@dataclasses.dataclass(frozen=True, kw_only=True)
class RandomSection(pygrain.RandomMapTransform):
  """Samples a random section in a given trajectory.

  Sampling always happens along the leading dimension. First a start index is
  randomly selected amongst all permissible locations and the sample is taken to
  be the contiguous section immediately after and including the start index,
  with the specified number of steps and stride.

  The random number generator is provided by the PyGrain sampler, which must
  therefore be seeded (e.g. `pygrain.IndexSampler(..., seed=seed)`).

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

  def __post_init__(self):
    if self.num_steps < 1:
      raise ValueError(f"`num_steps` ({self.num_steps}) must be positive.")
    if self.stride < 1:
      raise ValueError(f"`stride` ({self.stride}) must be positive.")

  def random_map(
      self, features: FeatureDict, rng: np.random.Generator
  ) -> FeatureDict:
    """See base class."""
    total_length = features[self.feature_names[0]].shape[0]
    sample_length = self.stride * (self.num_steps - 1) + 1

    for name in self.feature_names[1:]:
      feature_length = features[name].shape[0]
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
    start_idx = int(rng.integers(0, total_length - sample_length + 1))
    end_idx = start_idx + sample_length

    for name in self.feature_names:
      features[name] = features[name][start_idx : end_idx : self.stride]
    return features


@dataclasses.dataclass(frozen=True, kw_only=True)
class Split(pygrain.MapTransform):
  """Splits an array feature into multiple sub arrays.

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
    if self.keep_presplit and self.feature_name in self.split_names:
      raise ValueError(
          f"`feature_name` ({self.feature_name}) cannot appear in `split_names`"
          f" {self.split_names} when `keep_presplit=True`."
      )

  def map(self, features: FeatureDict) -> FeatureDict:
    """See base class."""
    feature = features[self.feature_name]
    if sum(self.split_sizes) != feature.shape[self.axis]:
      raise ValueError(
          f"`split_sizes` {self.split_sizes} must sum to the dimension of"
          f" feature {self.feature_name} along axis {self.axis}"
          f" ({feature.shape[self.axis]})."
      )
    # Unlike `tf.split`, `np.split` expects the indices at which to split.
    splits = np.split(feature, np.cumsum(self.split_sizes)[:-1], axis=self.axis)
    if not self.keep_presplit:
      del features[self.feature_name]
    for split_name, split_value in zip(self.split_names, splits, strict=True):
      features[split_name] = split_value
    return features


@dataclasses.dataclass(frozen=True, kw_only=True)
class SelectAs(pygrain.MapTransform):
  """Select and rename features similar to 'SELECT ... AS ...' in sql.

  Attributes:
    select_features: Names of the features to be chosen.
    as_features: Replacement names for the selected features.

  Note: If the names of the features in the original dictionary are not among
    select_features, they will be removed from the dictionary, unless they are
    an internal feature, i.e., their key starts by '_'. Features can be swapped,
    e.g. `select_features=("u", "x"), as_features=("x", "u")`.
  """

  select_features: Sequence[str] | str
  as_features: Sequence[str] | str

  def __post_init__(self):
    if isinstance(self.select_features, str):
      object.__setattr__(self, "select_features", (self.select_features,))
    if isinstance(self.as_features, str):
      object.__setattr__(self, "as_features", (self.as_features,))
    if len(self.select_features) != len(self.as_features):
      raise ValueError(
          f"Length of `select_features` [{self.select_features}] must match "
          f"that of `as_features` [{self.as_features}]"
      )
    if len(set(self.select_features)) != len(self.select_features):
      raise ValueError(
          f"The names in `select_features` [{self.select_features}] must be"
          " unique."
      )
    if len(set(self.as_features)) != len(self.as_features):
      raise ValueError(
          f"The names in `as_features` [{self.as_features}] must be unique."
      )

  def map(self, features: FeatureDict) -> FeatureDict:
    """See base class."""
    selected = {
        new: features[old]
        for old, new in zip(self.select_features, self.as_features, strict=True)
    }
    # Keep internal features, unless they are overwritten by a selected one.
    internal = {
        name: value
        for name, value in features.items()
        if name.startswith("_") and name not in selected
    }
    return internal | selected
