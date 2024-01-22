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

"""Utilities for loading samples from initial/target sets from hdf5 files."""

from collections.abc import Callable
from typing import Any

import grain.tensorflow as tfgrain
import jax
import numpy as np
from swirl_dynamics.data import hdf5_utils
import tensorflow as tf


Array = jax.Array
PyTree = Any
DynamicsFn = Callable[[Array, Array, PyTree], Array]


class UnpairedDataLoader:
  """Unpaired dataloader for loading samples from two distributions."""

  def __init__(
      self,
      batch_size: int,
      dataset_path_a: str,
      dataset_path_b: str,
      seed: int,
      split: str | None = None,
      spatial_downsample_factor_a: int = 1,
      spatial_downsample_factor_b: int = 1,
      normalize: bool = False,
      normalize_stats_a: dict[str, Array] | None = None,
      normalize_stats_b: dict[str, Array] | None = None,
      tf_lookup_batch_size: int = 1024,
      tf_lookup_num_parallel_calls: int = -1,
      tf_interleaved_shuffle: bool = False,
  ):

    loader, normalize_stats_a = create_loader_from_hdf5(
        batch_size=batch_size,
        dataset_path=dataset_path_a,
        seed=seed,
        split=split,
        spatial_downsample_factor=spatial_downsample_factor_a,
        normalize=normalize,
        normalize_stats=normalize_stats_a,
        tf_lookup_batch_size=tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=tf_interleaved_shuffle,)

    self.loader_a = iter(loader)

    loader, normalize_stats_b = create_loader_from_hdf5(
        batch_size=batch_size,
        dataset_path=dataset_path_b,
        seed=seed,
        split=split,
        spatial_downsample_factor=spatial_downsample_factor_b,
        normalize=normalize,
        normalize_stats=normalize_stats_b,
        tf_lookup_batch_size=tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=tf_interleaved_shuffle,)
    self.loader_b = iter(loader)

    self.normalize_stats_a = normalize_stats_a
    self.normalize_stats_b = normalize_stats_b

  def __iter__(self):
    return self

  def __next__(self):

    b = next(self.loader_b)
    a = next(self.loader_a)

    # Return dictionary with a tuple, following the cycleGAN convention.
    return {"x_0": a["u"], "x_1": b["u"]}


def create_loader_from_hdf5(
    batch_size: int,
    dataset_path: str,
    seed: int,
    split: str | None = None,
    spatial_downsample_factor: int = 1,
    normalize: bool = False,
    normalize_stats: dict[str, Array] | None = None,
    tf_lookup_batch_size: int = 1024,
    tf_lookup_num_parallel_calls: int = -1,
    tf_interleaved_shuffle: bool = False,
) -> tuple[tfgrain.TfDataLoader, dict[str, Array] | None]:
  """Load pre-computed trajectories dumped to hdf5 file.

  Args:
    batch_size: Batch size returned by dataloader. If set to -1, use entire
        dataset size as batch_size.
    dataset_path: Absolute path to dataset file.
    seed: Random seed to be used in data sampling.
    split: Data split - train, eval, test, or None.
    spatial_downsample_factor: reduce spatial resolution by factor of x.
    normalize: Flag for adding data normalization (subtact mean divide by std.).
    normalize_stats: Dictionary with mean and std stats to avoid recomputing.
    tf_lookup_batch_size: Number of lookup batches (in cache) for grain.
    tf_lookup_num_parallel_calls: Number of parallel call for lookups in the
        dataset. -1 is set to let grain optimize tha number of calls.
    tf_interleaved_shuffle: Using a more localized shuffle instead of a global
        suffle of the data.

  Returns:
    loader, stats (optional): Tuple of dataloader and dictionary containing
                              mean and std stats (if normalize=True, else dict
                              contains NoneType values).
  """
  # TODO: create the data arrays following a similar convention.
  snapshots = hdf5_utils.read_single_array(
      dataset_path,
      f"{split}/u",
  )

  # If the data is given aggregated by trajectory, we scramble the time stamps.
  if snapshots.ndim == 5:
    # We assume that the data is 2-dimensional + channels.
    num_trajs, num_time, nx, ny, dim = snapshots.shape
    snapshots = snapshots.reshape((num_trajs*num_time, nx, ny, dim))
  elif snapshots.ndim != 4:
    raise ValueError("The dimension of the data should be either a 5- or 4-",
                     "dimensional tensor: two spatial dimension, one chanel ",
                     "dimension and either number of samples, or number of ",
                     "trajectories plus number of time-steps per trajectories.",
                     f" Instead the data is a {snapshots.ndim}-tensor.")

  # Downsample the data spatially, the data is two-dimensional.
  snapshots = snapshots[
      :, ::spatial_downsample_factor, ::spatial_downsample_factor, :
  ]

  return_stats = None

  if normalize:
    if normalize_stats is not None:
      mean = normalize_stats["mean"]
      std = normalize_stats["std"]
    else:
      if split != "train":
        data_for_stats = hdf5_utils.read_single_array(
            dataset_path,
            "train/u",
        )
        if data_for_stats.ndim == 5:
          num_trajs, num_time, nx, ny, dim = data_for_stats.shape
          data_for_stats = data_for_stats.reshape(
              (num_trajs * num_time, nx, ny, dim)
          )
        # Also perform the downsampling.
        data_for_stats = data_for_stats[
            :, ::spatial_downsample_factor, ::spatial_downsample_factor, :
        ]
      else:
        data_for_stats = snapshots

      # This need to be run in CPU. This needs to be done only once.
      mean = np.mean(data_for_stats, axis=0)
      std = np.std(data_for_stats, axis=0)

    # Normalize snapshot so they are distributed appropiately.
    snapshots -= mean
    snapshots /= std

    return_stats = {"mean": mean, "std": std}

  source = tfgrain.TfInMemoryDataSource.from_dataset(
      tf.data.Dataset.from_tensor_slices({
          "u": snapshots,
      })
  )

  # Grain fine-tuning.
  tfgrain.config.update(
      "tf_lookup_num_parallel_calls", tf_lookup_num_parallel_calls
  )
  tfgrain.config.update("tf_interleaved_shuffle", tf_interleaved_shuffle)
  tfgrain.config.update("tf_lookup_batch_size", tf_lookup_batch_size)

  if batch_size == -1:  # Use full dataset as batch
    batch_size = len(source)

  loader = tfgrain.TfDataLoader(
      source=source,
      sampler=tfgrain.TfDefaultIndexSampler(
          num_records=len(source),
          seed=seed,
          num_epochs=None,  # loads indefinitely.
          shuffle=True,
          shard_options=tfgrain.ShardByJaxProcess(drop_remainder=True),
      ),
      transformations=[],
      batch_fn=tfgrain.TfBatch(batch_size=batch_size, drop_remainder=False),
  )
  return loader, return_stats
