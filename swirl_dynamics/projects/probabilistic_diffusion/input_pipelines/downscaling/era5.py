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

"""Input pipeline for ERA5."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, SupportsIndex

from etils import epath
import grain.python as pygrain
import numpy as np
import xarray_tensorstore as xrts

FlatFeatures = dict[str, Any]


def read_zarr_variables_as_tuple(
    path: epath.PathLike,
    variables: Sequence[str],
    dataset_sel_indexers: Mapping[str, Any] | None = None,
    dataset_isel_indexers: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, ...]:
  """Reads variables from a zarr dataset and returns as a tuple of ndarrays."""
  ds = xrts.open_zarr(path)
  if dataset_sel_indexers:
    ds = ds.sel(dataset_sel_indexers)

  if dataset_isel_indexers:
    ds = ds.isel(dataset_isel_indexers)
  return tuple(ds[v].to_numpy() for v in variables)


@dataclasses.dataclass(frozen=True)
class Standardize(pygrain.MapTransform):
  """Standardize variables pixel-wise using pre-computed climatology."""

  input_fields: Sequence[str]
  mean: Sequence[np.ndarray]
  std: Sequence[np.ndarray]

  def __post_init__(self):
    assert len(self.input_fields) == len(self.mean) == len(self.std), (
        f"`input_fields` ({len(self.input_fields)}), `mean` ({len(self.mean)}),"
        f" and `std` ({len(self.std)}) must have the same length."
    )

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field, mean, std in zip(self.input_fields, self.mean, self.std):
      features[field] = (features[field] - mean) / std
    return features


@dataclasses.dataclass(frozen=True)
class Stack(pygrain.MapTransform):
  """Creates a new field by stacking selected fields."""

  input_fields: Sequence[str]
  output_field: str
  axis: int
  remove_inputs: bool = True

  def map(self, features: FlatFeatures) -> FlatFeatures:
    arrays = [features[f] for f in self.input_fields]
    features[self.output_field] = np.stack(arrays, axis=self.axis)
    if self.remove_inputs:
      for f in self.input_fields:
        del features[f]
    return features


@dataclasses.dataclass(frozen=True)
class Squeeze(pygrain.MapTransform):
  """Squeeze selected fields along some axis."""

  input_fields: Sequence[str]
  axis: int | Sequence[int] | None = None

  def map(self, features: FlatFeatures) -> FlatFeatures:
    for field in self.input_fields:
      features[field] = np.squeeze(features[field], axis=self.axis)
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
class Slice(pygrain.MapTransform):
  """Slice a selected field."""

  input_field: str
  slices: tuple[slice, ...]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    features[self.input_field] = features[self.input_field][self.slices]
    return features


@dataclasses.dataclass(frozen=True)
class AssembleCondDict(pygrain.MapTransform):
  """Assemble fields into a conditional dictionary."""

  cond_fields: Sequence[str]
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


class CondZarrDataSource:
  """A Zarr-based data source for training generative models.

  This data source takes zarr datasets, one for samples `x` (and optionally one
  for conditions `z`), and creates examples for training unconditional `p(x)`
  (or conditional `p(x|z)`) generative models. The datasets are expected to have
  a *uniformly sampled* time dimension (`np.datetime64` type) in its
  coordinates. It supports loading subsets of variables, dataset-level index
  selection (e.g. keeping only the lowest pressure level for all variables
  used), date range selection and downsampling, and sequence loading.
  """

  def __init__(
      self,
      path: epath.PathLike,
      variables: Sequence[str],  # e.g. ("2m_temperature",)
      isel_indexers: Mapping[str, Any] | None = None,
      cond_path: epath.PathLike | None = None,
      cond_variables: Sequence[str] | None = None,
      cond_isel_indexers: Mapping[str, Any] | None = None,
      date_range: tuple[str, str] = ("1959", "2015"),
      interval: int = 1,
      sequence_offsets: Sequence[int] = (0,),
  ):
    """Data source constructor.

    Args:
      path: The file path of the zarr dataset containing the samples.
      variables: The set of variables to load from the sample dataset.
      isel_indexers: The indexers to apply to the samples, passed to the
        `.isel()` method of the corresponding `xarray.Dataset` object.
      cond_path: The file path of the zarr dataset containing the conditional
        information.
      cond_variables: The set of variables to load from the conditional dataset.
      cond_isel_indexers: The indexers to apply to the conditions, passed to the
        `.isel()` method of the corresponding `xarray.Dataset` object.
      date_range: The data range applied. Must result in the same data length
        between sample and conditional datasets when applicable.
      interval: The downsampling factor in time, applied uniformly. For example,
        setting `interval=24` for an hourly dataset results in effectively
        loading daily data.
      sequence_offsets: The time offsets in the loaded sequence. For example,
        the default value `(0,)` results in loading individual snapshots (the
        singleton time dimension is not removed) and `(-1, 0, 1)` results in
        loading snapshot sequences of length 3, where the snapshots are
        consecutive in (downsampled) time.
    """
    self._date_slice = slice(
        np.datetime64(date_range[0], "D"),
        np.datetime64(date_range[1], "D"),
        interval,
    )
    self._ds = xrts.open_zarr(path).sel(time=self._date_slice)
    self._ds = self._ds.isel(isel_indexers) if isel_indexers else self._ds
    self._data_arrays = {v: self._ds[v] for v in variables}

    if cond_path:
      self._cond_ds = xrts.open_zarr(cond_path, context=None).sel(
          time=self._date_slice
      )
      if cond_isel_indexers:
        self._cond_ds = self._cond_ds.isel(cond_isel_indexers)

      if self._cond_ds.sizes["time"] != self._ds.sizes["time"]:
        raise ValueError(
            "Time lengths of datasets are different:"
            f" {self._cond_ds.sizes['time']} (cond) vs."
            f" {self._ds.sizes['time']} (main)."
        )

      for v in cond_variables:
        self._data_arrays[f"cond_{v}"] = self._cond_ds[v]

    self._seq_offsets = np.asarray(sequence_offsets) - np.min(sequence_offsets)
    self._max_idx = self._ds.sizes["time"] - np.max(self._seq_offsets) - 1
    self._len = self._max_idx + 1

  def __len__(self):
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, np.ndarray]:
    idx = record_key.__index__()
    time_slice = slice(idx, idx + np.max(self._seq_offsets) + 1)
    return {
        key: xrts.read(array.isel({"time": time_slice})).data[self._seq_offsets]
        for key, array in self._data_arrays.items()
    }


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


def create_cond_dataset(
    zarr_path: epath.PathLike,
    variables: Sequence[str],
    cond_zarr_path: epath.PathLike,
    cond_variables: Sequence[str],
    sample_indexers: Mapping[str, Any] | None = None,
    cond_indexers: Mapping[str, Any] | None = None,
    sample_field_rename: str = "x",
    date_range: tuple[str, str] = ("1959", "2015"),
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    standardize_samples: bool = False,
    stats_zarr_path: epath.PathLike | None = None,
    standardize_cond: bool = False,
    cond_stats_zarr_path: epath.PathLike | None = None,
    cond_maskout_prob: float = 0.0,
) -> pygrain.DataLoader:
  """Creates a pygrain data pipeline for training conditional generation."""
  source = CondZarrDataSource(
      path=zarr_path,
      variables=variables,
      isel_indexers=sample_indexers,
      cond_path=cond_zarr_path,
      cond_variables=cond_variables,
      cond_isel_indexers=cond_indexers,
      date_range=date_range,
  )
  standardizations = []
  if standardize_samples:
    standardizations += [
        # Output standardization.
        Standardize(
            input_fields=variables,
            mean=read_zarr_variables_as_tuple(
                stats_zarr_path,
                variables,
                dataset_sel_indexers={"stats": "mean"},
                dataset_isel_indexers=sample_indexers,
            ),
            std=read_zarr_variables_as_tuple(
                stats_zarr_path,
                variables,
                dataset_sel_indexers={"stats": "std"},
                dataset_isel_indexers=sample_indexers,
            ),
        ),
    ]
  # Standardization for condition.
  prefixed_cond_variables = ["cond_" + v for v in cond_variables]
  if standardize_cond:
    standardizations += [
        Standardize(
            input_fields=prefixed_cond_variables,
            mean=read_zarr_variables_as_tuple(
                cond_stats_zarr_path,
                cond_variables,
                dataset_sel_indexers={"stats": "mean"},
                dataset_isel_indexers=cond_indexers,
            ),
            std=read_zarr_variables_as_tuple(
                cond_stats_zarr_path,
                cond_variables,
                dataset_sel_indexers={"stats": "std"},
                dataset_isel_indexers=cond_indexers,
            ),
        )
    ]

  all_variables = list(variables) + prefixed_cond_variables
  transformations = standardizations + [
      Squeeze(input_fields=all_variables, axis=0),  # Squeeze out the time dim.
      Rot90(input_fields=all_variables, k=1, axes=(0, 1)),
      Stack(input_fields=variables, output_field=sample_field_rename, axis=-1),
      Stack(
          input_fields=prefixed_cond_variables, output_field="low_res", axis=-1
      ),
      RandomMaskout(input_fields=("low_res",), probability=cond_maskout_prob),
      AssembleCondDict(cond_fields=("low_res",), prefix="channel:"),
  ]
  data_loader = pygrain.load(
      source=source,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return data_loader


def create_uncond_dataset(
    zarr_path: epath.PathLike,
    variables: Sequence[str],
    sample_indexers: Mapping[str, Any] | None = None,
    sample_field_rename: str = "x",
    date_range: tuple[str, str] = ("1959", "2015"),
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    drop_remainder: bool = True,
    worker_count: int | None = 0,
    standardize_samples: bool = False,
    stats_zarr_path: epath.PathLike | None = None,
) -> pygrain.DataLoader:
  """Creates a pygrain data pipeline for training conditional generation."""
  source = CondZarrDataSource(
      path=zarr_path,
      variables=variables,
      isel_indexers=sample_indexers,
      date_range=date_range,
  )
  standardizations = []
  if standardize_samples:
    standardizations += [
        # output
        Standardize(
            input_fields=variables,
            mean=read_zarr_variables_as_tuple(
                stats_zarr_path,
                variables,
                dataset_sel_indexers={"stats": "mean"},
                dataset_isel_indexers=sample_indexers,
            ),
            std=read_zarr_variables_as_tuple(
                stats_zarr_path,
                variables,
                dataset_sel_indexers={"stats": "std"},
                dataset_isel_indexers=sample_indexers,
            ),
        ),
    ]

  transformations = standardizations + [
      Squeeze(input_fields=variables, axis=0),  # squeeze out the time dim
      Rot90(input_fields=variables, k=1, axes=(0, 1)),
      Stack(input_fields=variables, output_field=sample_field_rename, axis=-1),
  ]
  data_loader = pygrain.load(
      source=source,
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )
  return data_loader
