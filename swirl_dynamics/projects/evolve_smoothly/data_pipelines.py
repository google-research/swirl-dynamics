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

"""Data pipelines."""

import grain.tensorflow as tfgrain
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.data import tfgrain_transforms as transforms
import tensorflow as tf

_DEFAULT_LINEAR_RESCALE = transforms.LinearRescale(
    feature_name="x", input_range=(0, 64), output_range=(-1, 1)
)


def create_batch_decode_pipeline(
    hdf5_file_path: str,
    *,
    snapshot_field: str = "train/u",
    grid_field: str = "train/x",
    num_snapshots_to_train: int = 5000,
    transformations: tfgrain.Transformations = (_DEFAULT_LINEAR_RESCALE,),
    seed: int = 42,
    batch_size: int = 32,
) -> tfgrain.TfDataLoader:
  """Creates an in-memory dataloader for batch decode training.

  Works on 1d snapshots and grid.

  Args:
    hdf5_file_path: path of hdf5 file that contains the data.
    snapshot_field: the field name of the hdf5 dataset corresponding to the
      snapshots - the expected shape is `(traj, time, grid, 1)`.
    grid_field: the field name of the hdf5 dataset corresponding to the grid -
      the expected shape is `(grid, 1)`.
    num_snapshots_to_train: number of training snapshots (randomly selected
      after collapsing the `traj` and `time` dimensions).
    transformations: data transformations applied to snapshots or grid before
      batching.
    seed: random seed for both selecting training snapshots and dataloader
      shuffling.
    batch_size: number of grid points per batch - batch dimension will be
      `(batch, num_snapshots, 1)`.

  Returns:
    A TfGrain dataloader.
  """
  snapshots, grid = hdf5_utils.read_arrays_as_tuple(
      hdf5_file_path, (snapshot_field, grid_field)
  )
  snapshots = np.reshape(snapshots, (-1,) + snapshots.shape[-2:])
  # select a subset of snapshots to train
  rng = np.random.default_rng(seed)
  train_idx = rng.choice(
      len(snapshots), size=num_snapshots_to_train, replace=False
  )
  source = tfgrain.TfInMemoryDataSource.from_dataset(
      tf.data.Dataset.from_tensor_slices({
          "u": np.swapaxes(snapshots[train_idx], 0, 1),
          "x": grid,
      })
  )
  loader = tfgrain.TfDataLoader(
      source=source,
      sampler=tfgrain.TfDefaultIndexSampler(
          num_records=len(source),
          seed=seed,
          num_epochs=None,
          shuffle=True,
          shard_options=tfgrain.ShardByJaxProcess(drop_remainder=True),
      ),
      transformations=transformations,
      batch_fn=tfgrain.TfBatch(batch_size=batch_size, drop_remainder=False),
  )
  return loader


def create_encode_decode_pipeline(
    hdf5_file_path: str,
    *,
    snapshot_field: str = "train/u",
    grid_field: str = "train/x",
    num_snapshots_to_train: int | None = None,
    transformations: tfgrain.Transformations = (_DEFAULT_LINEAR_RESCALE,),
    seed: int = 42,
    batch_size: int = 32,
) -> tfgrain.TfDataLoader:
  """Creates an in-memory dataloader for encode-decode training.

  Works on 1d snapshots and grid.

  Args:
    hdf5_file_path: path of hdf5 file that contains the data.
    snapshot_field: the field name of the hdf5 dataset corresponding to the
      snapshots - the expected shape is `(traj, time, grid, 1)`.
    grid_field: the field name of the hdf5 dataset corresponding to the grid -
      the expected shape is `(grid, 1)`.
    num_snapshots_to_train: number of training snapshots (randomly selected
      after collapsing the `traj` and `time` dimensions). If `None`, train on
      all snapshots.
    transformations: data transformations applied to snapshots or grid before
      batching.
    seed: random seed for both randomly selecting training snapshots (if
      applicable) and dataloader shuffling.
    batch_size: number of snapshots per batch.

  Returns:
    A TfGrain dataloader.
  """
  snapshots, grid = hdf5_utils.read_arrays_as_tuple(
      hdf5_file_path, (snapshot_field, grid_field)
  )
  snapshots = np.reshape(snapshots, (-1,) + snapshots.shape[-2:])
  # select a subset of snapshots to train if applicable
  if num_snapshots_to_train and num_snapshots_to_train < len(snapshots):
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(
        len(snapshots), size=num_snapshots_to_train, replace=False
    )
  else:
    train_idx = slice(None)
  snapshots = snapshots[train_idx]

  source = tfgrain.TfInMemoryDataSource.from_dataset(
      tf.data.Dataset.from_tensor_slices({
          "u": snapshots,
          "x": np.tile(grid, (len(snapshots), 1, 1)),
      })
  )
  loader = tfgrain.TfDataLoader(
      source=source,
      sampler=tfgrain.TfDefaultIndexSampler(
          num_records=len(source),
          seed=seed,
          num_epochs=None,
          shuffle=True,
          shard_options=tfgrain.ShardByJaxProcess(drop_remainder=True),
      ),
      transformations=transformations,
      batch_fn=tfgrain.TfBatch(batch_size=batch_size, drop_remainder=False),
  )
  return loader


def create_latent_dynamics_pipeline(
    hdf5_file_path: str,
    *,
    snapshot_field: str = "train/u",
    tspan_field: str = "train/t",
    grid_field: str = "train/x",
    num_time_steps: int = 2,
    time_downsample_factor: int = 1,
    transformations: tfgrain.Transformations = (_DEFAULT_LINEAR_RESCALE,),
    seed: int = 42,
    batch_size: int = 32,
) -> tfgrain.TfDataLoader:
  """Creates an in-memory dataloader for latent dynamics training.

  Works on 1d trajectories and grid. We randomly select a section of the raw
  trajectories, with specified length and downsample factor.

  Args:
    hdf5_file_path: path of hdf5 file that contains the data.
    snapshot_field: the field name of the hdf5 dataset corresponding to the
      snapshots - the expected shape is `(traj, time, grid, 1)`.
    tspan_field: the field name of the hdf5 dataset corresponding to the time
      stamps - the expected shape is `(traj, time)`.
    grid_field: the field name of the hdf5 dataset corresponding to the grid -
      the expected shape is `(grid, 1)`.
    num_time_steps: the number of time steps in the output trajectories.
    time_downsample_factor: the downsampling factor in time.
    transformations: data transformations applied to snapshots or grid before
      batching.
    seed: random seed for both randomly selecting training snapshots (if
      applicable) and dataloader shuffling.
    batch_size: number of snapshots per batch.

  Returns:
    A TfGrain dataloader.
  """
  snapshots, tspan, grid = hdf5_utils.read_arrays_as_tuple(
      hdf5_file_path, (snapshot_field, tspan_field, grid_field)
  )
  source = tfgrain.TfInMemoryDataSource.from_dataset(
      tf.data.Dataset.from_tensor_slices({
          "u": snapshots,
          "t": tspan,
          "x": np.tile(grid, (len(snapshots), 1, 1)),
      })
  )
  section_transform = transforms.RandomSection(
      feature_names=("u", "t"),
      num_steps=num_time_steps,
      stride=time_downsample_factor,
  )
  loader = tfgrain.TfDataLoader(
      source=source,
      sampler=tfgrain.TfDefaultIndexSampler(
          num_records=len(source),
          seed=seed,
          num_epochs=None,
          shuffle=True,
          shard_options=tfgrain.ShardByJaxProcess(drop_remainder=True),
      ),
      transformations=(section_transform,) + tuple(transformations),
      batch_fn=tfgrain.TfBatch(batch_size=batch_size, drop_remainder=False),
  )
  return loader
