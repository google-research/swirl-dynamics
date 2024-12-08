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

"""PyGrain transforms for LENS2 and ERA5 datasets."""
from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any, KeysView, Literal

from etils import epath
import grain.python as pygrain
import numpy as np
import scipy.optimize as spopt
import xarray_tensorstore as xrts

FlatFeatures = dict[str, Any]


def read_stats(
    dataset: epath.PathLike,
    variables: Mapping[str, Mapping[str, Any] | None],
    field: Literal["mean", "std"],
) -> dict[str, np.ndarray]:
  """Reads variables from a zarr dataset and returns as a dict of ndarrays."""
  ds = xrts.open_zarr(dataset)
  out = {}
  for var, indexers in variables.items():
    indexers = indexers | {"stats": field} if indexers else {"stats": field}
    stats = ds[var].sel(indexers).to_numpy()
    assert stats.ndim == 2 or stats.ndim == 3
    stats = np.expand_dims(stats, axis=-1) if stats.ndim == 2 else stats
    out[var] = stats
  return out


@dataclasses.dataclass(frozen=True)
class Standardize(pygrain.MapTransform):
  """Standardize variables pixel-wise using pre-computed mean and std."""

  input_fields: Sequence[str] | KeysView[str]
  mean: Mapping[str, np.ndarray]
  std: Mapping[str, np.ndarray]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = (features[field] - self.mean[field]) / self.std[field]
    return features


@dataclasses.dataclass(frozen=True)
class StandardizeNested(pygrain.MapTransform):
  """Standardize variables pixel-wise using pre-computed mean and std.

  This version is written for the data aligned loader.
  """

  main_field: str
  input_fields: Sequence[str]
  mean: Mapping[str, np.ndarray]
  std: Mapping[str, np.ndarray]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[self.main_field][field] = (
          features[self.main_field][field] - self.mean[field]
      ) / self.std[field]
    return features


@dataclasses.dataclass(frozen=True)
class StandardizeNestedWithStats(pygrain.MapTransform):
  """Standardize variables pixel-wise using pre-computed mean and std.

  This version is written for the data aligned loader such that the stats are
  stored in the same field as the data.

  Attributes:
    main_field: The field that contains the data to be standardized. For the the
      data aligned loader this can be either input or output.
    mean_field: The field that contains the mean of each of the input fields.
    std_field: The field that contains the std of each of the input fields.
    input_fields: The fields to standardize.
  """

  main_field: str
  mean_field: str
  std_field: str
  input_fields: Sequence[str]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[self.main_field][field] = (
          features[self.main_field][field] - features[self.mean_field][field]
      ) / features[self.std_field][field]
    return features


@dataclasses.dataclass
class ComputeWindSpeed(pygrain.MapTransform):
  """Compute wind speed from velocity components and standardize."""

  u_field: str
  v_field: str
  mean: Mapping[str, np.ndarray]
  std: Mapping[str, np.ndarray]
  output_field: str
  remove_inputs: bool = True

  def __post_init__(self):
    ws_mean_squared = (
        self.mean[self.u_field] ** 2 + self.mean[self.v_field] ** 2
    )

    ws_var = (self.mean[self.u_field] ** 2) * self.std[self.u_field] ** 2 + (
        self.mean[self.v_field] ** 2
    ) * self.std[self.v_field] ** 2

    ws_var = ws_var / ws_mean_squared
    self.ws_mean = np.sqrt(ws_mean_squared)
    self.ws_std = np.sqrt(ws_var)

  def map(self, features: FlatFeatures) -> FlatFeatures:
    ws = np.sqrt(features[self.u_field] ** 2 + features[self.v_field] ** 2)
    features[self.output_field] = (ws - self.ws_mean) / self.ws_std

    if self.remove_inputs:
      del features[self.u_field], features[self.v_field]
    return features


@dataclasses.dataclass
class ComputeWindSpeedExact(pygrain.MapTransform):
  """Compute wind speed from velocity components and standardize.

  In this case we use the correct mean and std instead of a first order
  approximation.
  """

  u_field: str
  v_field: str
  speed_field: str
  mean: Mapping[str, np.ndarray]
  std: Mapping[str, np.ndarray]
  output_field: str
  remove_inputs: bool = True

  def map(self, features: FlatFeatures) -> FlatFeatures:
    ws = np.sqrt(features[self.u_field] ** 2 + features[self.v_field] ** 2)
    features[self.output_field] = (ws - self.mean[self.speed_field]) / self.std[
        self.speed_field
    ]

    if self.remove_inputs:
      del features[self.u_field], features[self.v_field]
    return features


@dataclasses.dataclass
class ComputeWindSpeedExactNested(pygrain.MapTransform):
  """Compute wind speed from velocity components and standardize.

  In this case we use the correct mean and std instead of a first order
  approximation. This one is designed to work with the alinged data loader,
  which is why it requires an extra field.
  TODO: refactor these functions.
  """

  main_field: str
  u_field: str
  v_field: str
  speed_field: str
  mean: Mapping[str, np.ndarray]
  std: Mapping[str, np.ndarray]
  output_field: str
  remove_inputs: bool = True

  def map(self, features: FlatFeatures) -> FlatFeatures:
    ws = np.sqrt(
        features[self.main_field][self.u_field] ** 2
        + features[self.main_field][self.v_field] ** 2
    )
    features[self.main_field][self.output_field] = (
        ws - self.mean[self.speed_field]
    ) / self.std[self.speed_field]

    if self.remove_inputs:
      del (
          features[self.main_field][self.u_field],
          features[self.main_field][self.v_field],
      )
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
class BatchOT(pygrain.MapTransform):
  """Shuffles the elements of a chunk by solving a linear assigment problem."""

  input_field: str
  output_field: str
  batch_size: int
  cost: Callable[[np.ndarray], np.ndarray] = np.square
  mean_input_field: str | None = None
  std_input_field: str | None = None

  def map(self, features: FlatFeatures) -> FlatFeatures:
    # Reorganizes the elements to align them.
    in_features = features[self.input_field].reshape((self.batch_size, -1))
    out_features = features[self.output_field].reshape((self.batch_size, -1))

    cost_matrix = np.sum(
        self.cost(in_features[:, None, :] - out_features[None, :, :]), axis=-1
    )
    index = spopt.linear_sum_assignment(cost_matrix)
    features[self.input_field] = features[self.input_field][index[0]]
    features[self.output_field] = features[self.output_field][index[1]]

    # We also modify the conditioning fields.
    if self.mean_input_field:
      features[self.mean_input_field] = features[self.mean_input_field][
          index[0]
      ]
    if self.std_input_field:
      features[self.std_input_field] = features[self.std_input_field][index[0]]

    return features


@dataclasses.dataclass(frozen=True)
class RandomShuffleChunk(pygrain.RandomMapTransform):
  """Randomly shuffle a chunk."""

  input_fields: Sequence[str]
  batch_size: int

  def random_map(
      self, features: FlatFeatures, rng: np.random.Generator
  ) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = rng.permutation(features[field], axis=0)
    return features


@dataclasses.dataclass(frozen=True)
class ConcatenateNested(pygrain.MapTransform):
  """Creates a new field by concatenating selected fields.

  This one is designed to work with the data aligned loader, so one needs to
  squeeze from both the input and ouput fields.
  """

  main_field: str
  input_fields: Sequence[str]
  output_field: str
  axis: int
  remove_inputs: bool = True

  def map(self, features: FlatFeatures) -> FlatFeatures:
    arrays = [features[self.main_field][f] for f in self.input_fields]
    features[self.output_field] = np.concatenate(arrays, axis=self.axis)
    if self.remove_inputs:
      for f in self.input_fields:
        del features[self.main_field][f]
      del features[self.main_field]
    return features


@dataclasses.dataclass(frozen=True)
class FilterExtremes(pygrain.FilterTransform):
  """Transformations for elements with norm greater than a threshold.

  Attributes:
    field: The list of the fields to check for the norm.
    extreme_norm: The threshold for which element will be dropped.
  """

  fields: Sequence[str]
  extreme_norm: float = 270

  def filter(self, element) -> bool:
    """Filters a single element with norm higher than self.extreme_norm."""
    for field in self.fields:
      # The elements have dimensions [..., lon, lat, channel].
      if (
          np.linalg.norm(element[field], axis=(-3, -2)) > self.extreme_norm
      ).any():
        return False
    return True
