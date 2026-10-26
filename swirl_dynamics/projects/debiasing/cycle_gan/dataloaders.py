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

"""Unpaired dataloaders for the CycleGAN debiasing examples.

The loaders feed two *independently* sampled streams of snapshots, one per
domain, to the trainer as batches `{"real_data_a": ..., "real_data_b": ...}`.
The reference dataset is the public Kuramoto-Sivashinsky (KS) file

  gs://gresearch/swirl_dynamics/downscaling/KS_finite_volumes_vs_pseudo_spectral.hdf5

which stores trajectories computed with a low-order finite-volume scheme
(`u_fd`, domain A) and with a pseudo-spectral scheme at four times the spatial
resolution (`u_sp`, domain B), together with the grid `x` and the times `t`.
The file is read locally (e.g. after `gsutil cp`); the pseudo-spectral
trajectories are sub-sampled in space to the finite-volume resolution, the
last trajectories are held out for evaluation and every trajectory is split
into snapshots of shape `(length, 1)`.
"""

from collections.abc import Iterator, Mapping
import dataclasses
from typing import SupportsIndex

import grain.python as pygrain
import h5py
import numpy as np

KS_DATASET_URL = (
    "gs://gresearch/swirl_dynamics/downscaling/"
    "KS_finite_volumes_vs_pseudo_spectral.hdf5"
)
LOW_FIDELITY_KEY = "u_fd"
HIGH_FIDELITY_KEY = "u_sp"

NormalizationStats = Mapping[str, np.ndarray]
Batch = dict[str, np.ndarray]


def normalization_stats(dataset: np.ndarray) -> NormalizationStats:
  """Per-location mean and standard deviation over the leading axis."""
  return {"mean": np.mean(dataset, axis=0), "std": np.std(dataset, axis=0)}


class SourceInMemoryNumpy(pygrain.RandomAccessDataSource):
  """Random-access source over the leading axis of an in-memory array."""

  def __init__(self, dataset: np.ndarray):
    """Initializes the source.

    Args:
      dataset: Array whose leading axis indexes the samples.
    """
    self._dataset = dataset

  def __len__(self) -> int:
    return self._dataset.shape[0]

  def __getitem__(self, record_key: SupportsIndex) -> Batch:
    index = record_key.__index__()
    if not 0 <= index < len(self):
      raise IndexError(f"Index {index} out of range [0, {len(self)}).")
    return {"u": self._dataset[index]}


@dataclasses.dataclass(frozen=True)
class _Standardize(pygrain.MapTransform):
  """Subtracts the mean and divides by the standard deviation of `u`."""

  mean: np.ndarray
  std: np.ndarray

  def map(self, features: Batch) -> Batch:
    safe_std = np.where(self.std == 0, 1.0, self.std)
    return {**features, "u": (features["u"] - self.mean) / safe_std}


@dataclasses.dataclass(frozen=True)
class _Rename(pygrain.MapTransform):
  """Renames the field `u` to `output_name`."""

  output_name: str

  def map(self, features: Batch) -> Batch:
    return {self.output_name: features["u"]}


def create_loader_from_ndarray(
    dataset: np.ndarray,
    batch_size: int,
    output_name: str,
    seed: int = 999,
    shuffle: bool = True,
    normalize_stats: NormalizationStats | None = None,
    num_epochs: int | None = None,
    drop_remainder: bool = True,
    worker_count: int = 0,
) -> pygrain.DataLoader:
  """Builds a pygrain loader over the leading axis of an in-memory array.

  Args:
    dataset: Array whose leading axis indexes the samples.
    batch_size: Number of samples per batch.
    output_name: Key of the samples in the yielded batches.
    seed: Seed of the index sampler.
    shuffle: Whether to shuffle the samples.
    normalize_stats: Mapping with keys `"mean"` and `"std"` used to standardise
      the samples; `None` leaves them unchanged.
    num_epochs: Number of passes over the data; `None` repeats forever.
    drop_remainder: Whether to drop the last incomplete batch.
    worker_count: Number of pygrain worker processes (`0` loads in-process).

  Returns:
    A loader yielding `{output_name: array}` batches.
  """
  transformations: list[pygrain.Transformation] = []
  if normalize_stats is not None:
    transformations.append(
        _Standardize(
            mean=np.asarray(normalize_stats["mean"]),
            std=np.asarray(normalize_stats["std"]),
        )
    )
  transformations.append(_Rename(output_name))
  return pygrain.load(
      source=SourceInMemoryNumpy(dataset),
      num_epochs=num_epochs,
      shuffle=shuffle,
      seed=seed,
      shard_options=pygrain.ShardByJaxProcess(drop_remainder=True),
      transformations=transformations,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      worker_count=worker_count,
  )


class UnpairedDataLoader(Iterator[Batch]):
  """Single-use iterator over independently sampled batches of two domains.

  Each item is `{"real_data_a": ..., "real_data_b": ...}` with the two arrays
  drawn from separate shuffled streams, i.e. the batches are sampled from the
  product of the two marginal distributions. The underlying iterators are
  created in `__init__` (`__iter__` returns `self`), so when `num_epochs` is
  finite the loader is exhausted after one traversal.

  Attributes:
    normalize_stats_a: Statistics used to standardise domain A, or `None`.
    normalize_stats_b: Statistics used to standardise domain B, or `None`.
  """

  def __init__(
      self,
      dataset_a: np.ndarray,
      dataset_b: np.ndarray,
      batch_size: int,
      seed: int = 37,
      normalize: bool = True,
      normalize_stats_a: NormalizationStats | None = None,
      normalize_stats_b: NormalizationStats | None = None,
      num_epochs: int | None = None,
      drop_remainder: bool = True,
      worker_count: int = 0,
  ):
    """Initializes the two streams.

    Args:
      dataset_a: Samples of domain A, leading axis indexes the samples.
      dataset_b: Samples of domain B, leading axis indexes the samples.
      batch_size: Number of samples per domain and batch.
      seed: Seed of domain A; domain B uses `seed + 1`.
      normalize: Whether to standardise the samples.
      normalize_stats_a: Statistics for domain A; computed from `dataset_a` when
        `None`. Pass the training statistics to an evaluation loader.
      normalize_stats_b: Same for domain B.
      num_epochs: Number of passes over each dataset; `None` repeats forever.
      drop_remainder: Whether to drop the last incomplete batch.
      worker_count: Number of pygrain worker processes per stream.
    """
    if normalize:
      if normalize_stats_a is None:
        normalize_stats_a = normalization_stats(dataset_a)
      if normalize_stats_b is None:
        normalize_stats_b = normalization_stats(dataset_b)
    else:
      normalize_stats_a = normalize_stats_b = None
    self.normalize_stats_a = normalize_stats_a
    self.normalize_stats_b = normalize_stats_b
    settings = dict(
        batch_size=batch_size,
        num_epochs=num_epochs,
        drop_remainder=drop_remainder,
        worker_count=worker_count,
    )
    self._iter_a = iter(
        create_loader_from_ndarray(
            dataset_a,
            output_name="real_data_a",
            seed=seed,
            normalize_stats=normalize_stats_a,
            **settings,
        )
    )
    self._iter_b = iter(
        create_loader_from_ndarray(
            dataset_b,
            output_name="real_data_b",
            seed=seed + 1,
            normalize_stats=normalize_stats_b,
            **settings,
        )
    )

  def __iter__(self) -> "UnpairedDataLoader":
    return self

  def __next__(self) -> Batch:
    return {**next(self._iter_a), **next(self._iter_b)}


def load_ks_snapshots(
    path: str,
    downsample_factor: int = 4,
    num_test_trajectories: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Loads the KS file and splits it into unpaired training and test snapshots.

  Args:
    path: Local path of the HDF5 file (see the module docstring).
    downsample_factor: Spatial sub-sampling of the pseudo-spectral trajectories,
      which brings them to the finite-volume resolution.
    num_test_trajectories: Number of trailing trajectories held out.

  Returns:
    `(train_a, train_b, test_a, test_b)`: finite-volume (A) and sub-sampled
    pseudo-spectral (B) snapshots of shape `(num_snapshots, length, 1)`.

  Raises:
    ValueError: If the held-out split leaves no training trajectory or the
      resolutions do not match after sub-sampling.
  """
  with h5py.File(path, "r") as file:
    low_fidelity = np.asarray(file[LOW_FIDELITY_KEY])
    high_fidelity = np.asarray(file[HIGH_FIDELITY_KEY])[
        :, :, ::downsample_factor
    ]
  num_trajectories = low_fidelity.shape[0]
  if not 0 < num_test_trajectories < num_trajectories:
    raise ValueError(
        f"num_test_trajectories={num_test_trajectories} must leave at least"
        f" one of the {num_trajectories} trajectories for training."
    )
  if low_fidelity.shape[-1] != high_fidelity.shape[-1]:
    raise ValueError(
        "The sub-sampled high-fidelity resolution"
        f" ({high_fidelity.shape[-1]}) does not match the low-fidelity one"
        f" ({low_fidelity.shape[-1]})."
    )

  def _snapshots(trajectories: np.ndarray) -> np.ndarray:
    return trajectories.reshape((-1, trajectories.shape[-1], 1))

  split = num_trajectories - num_test_trajectories
  return (
      _snapshots(low_fidelity[:split]),
      _snapshots(high_fidelity[:split]),
      _snapshots(low_fidelity[split:]),
      _snapshots(high_fidelity[split:]),
  )


def create_ks_loaders(
    path: str,
    batch_size: int,
    eval_batch_size: int,
    seed: int = 37,
    normalize: bool = True,
    downsample_factor: int = 4,
    num_test_trajectories: int = 16,
    worker_count: int = 0,
) -> tuple[UnpairedDataLoader, UnpairedDataLoader]:
  """Builds the training and evaluation loaders of the KS example.

  The evaluation loader standardises the held-out snapshots with the
  statistics of the training split, so that no evaluation data influences the
  normalisation.

  Args:
    path: Local path of the HDF5 file.
    batch_size: Training batch size per domain.
    eval_batch_size: Evaluation batch size per domain.
    seed: Seed of the training streams; evaluation uses `seed + 2`.
    normalize: Whether to standardise the samples.
    downsample_factor: See `load_ks_snapshots`.
    num_test_trajectories: See `load_ks_snapshots`.
    worker_count: Number of pygrain worker processes per stream.

  Returns:
    The training and the evaluation loader.
  """
  train_a, train_b, test_a, test_b = load_ks_snapshots(
      path, downsample_factor, num_test_trajectories
  )
  train_loader = UnpairedDataLoader(
      train_a,
      train_b,
      batch_size=batch_size,
      seed=seed,
      normalize=normalize,
      worker_count=worker_count,
  )
  eval_loader = UnpairedDataLoader(
      test_a,
      test_b,
      batch_size=eval_batch_size,
      seed=seed + 2,
      normalize=normalize,
      normalize_stats_a=train_loader.normalize_stats_a,
      normalize_stats_b=train_loader.normalize_stats_b,
      worker_count=worker_count,
  )
  return train_loader, eval_loader
