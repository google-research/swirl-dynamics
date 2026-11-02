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

"""Simple dataloader generating a short video of a fluid simulation."""

import grain.python as pygrain
import jax
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.data import pygrain_sources
from swirl_dynamics.data import pygrain_transforms as transforms

Array = jax.Array

# Number of batches each PyGrain worker process prepares in advance (only used
# when the loader runs with worker processes).
_WORKER_BUFFER_SIZE = 2


def _downsample_spatial(snapshots: np.ndarray, factor: int) -> np.ndarray:
  """Downsamples spatial axes of trajectory snapshots by `factor`."""
  if factor <= 1:
    return snapshots
  if snapshots.ndim == 3:
    return snapshots[:, :, ::factor]
  if snapshots.ndim == 4:
    return snapshots[:, :, ::factor, :]
  if snapshots.ndim == 5:
    return snapshots[:, :, ::factor, ::factor, :]
  raise NotImplementedError(
      f"Number of dimensions {snapshots.ndim} not supported for spatial"
      " downsampling."
  )


def create_loader_from_hdf5(
    num_time_steps: int,
    time_stride: int,
    batch_size: int,
    dataset_path: str,
    seed: int,
    split: str | None = None,
    spatial_downsample_factor: int = 1,
    normalize: bool = False,
    normalize_stats: dict[str, Array] | None = None,
    use_time_normalization: bool = False,
    tf_lookup_batch_size: int = 4,
    tf_lookup_num_parallel_calls: int = -1,
    tf_interleaved_shuffle: bool = False,
) -> tuple[pygrain.DataLoader, dict[str, Array | None]]:
  """Load pre-computed trajectories dumped to hdf5 file.

  If normalize flag is set, method will also return the mean and std used in
  normalization (which are calculated from train split).

  Arguments:
    num_time_steps: Number of time steps to include in each trajectory. If set
      to -1, use entire trajectory lengths.
    time_stride: Stride of trajectory sampling.
    batch_size: Batch size returned by dataloader. If set to -1, use entire
      per-process dataset shard size as batch_size.
    dataset_path: Absolute path to dataset file.
    seed: Random seed to be used in data sampling.
    split: Data split - train, eval, test, or None.
    spatial_downsample_factor: reduce spatial resolution by factor of x.
    normalize: Flag for adding data normalization (subtact mean divide by std.).
    normalize_stats: Dictionary with mean and std stats to avoid recomputing.
    use_time_normalization: Normalization is performed using a short sequence as
      a unit, instead of a single snaphot.
    tf_lookup_batch_size: Unused. It tuned the previous tf.data pipeline and is
      kept for compatibility with existing configs.
    tf_lookup_num_parallel_calls: Number of PyGrain worker processes loading the
      data. -1 (default) loads the data in the main process.
    tf_interleaved_shuffle: Unused. It tuned the previous tf.data pipeline and
      is kept for compatibility with existing configs.

  Returns:
    loader, stats (optional): tuple of dataloader and dictionary containing
                              mean and std stats (if normalize=True, else dict
                              contains NoneType values).
  """
  del tf_lookup_batch_size, tf_interleaved_shuffle  # Unused.
  if time_stride < 1:
    raise ValueError(f"`time_stride` ({time_stride}) must be positive.")
  split_prefix = f"{split}/" if split is not None else ""
  snapshots, tspan = hdf5_utils.read_arrays_as_tuple(
      dataset_path, (f"{split_prefix}u", f"{split_prefix}t")
  )
  snapshots = _downsample_spatial(snapshots, spatial_downsample_factor)
  if num_time_steps == -1:  # Use full (downsampled) trajectories.
    num_time_steps = tspan.shape[1] // time_stride
  if num_time_steps < 1:
    raise ValueError(f"`num_time_steps` ({num_time_steps}) must be positive.")

  if normalize:
    if normalize_stats is not None:
      mean = normalize_stats["mean"]
      std = normalize_stats["std"]
    else:
      if split not in ("train", None):
        data_for_stats = hdf5_utils.read_single_array(dataset_path, "train/u")
        data_for_stats = _downsample_spatial(
            data_for_stats, spatial_downsample_factor
        )
      else:
        data_for_stats = snapshots
      # TODO: For the sake of memory perform this in CPU.
      if use_time_normalization:
        data_for_stats = data_for_stats[:, ::time_stride]
        num_trajs, num_frames, *spatial_dims = data_for_stats.shape
        num_segments = num_frames // num_time_steps
        if num_segments < 1:
          raise ValueError(
              f"Not enough strided steps ({num_frames}) for"
              f" `num_time_steps` ({num_time_steps})."
          )
        data_for_stats = data_for_stats[:, : (num_segments * num_time_steps)]
        data_for_stats = np.reshape(
            data_for_stats,
            (num_trajs, num_segments, num_time_steps, *spatial_dims),
        )

      mean = np.mean(data_for_stats, axis=(0, 1))
      std = np.std(data_for_stats, axis=(0, 1))
  else:
    mean, std = None, None
  source = pygrain_sources.ArrayDataSource(
      u=snapshots,  # states
  )
  # This transform randomly takes a random section from the trajectory with the
  # desired length and stride.
  operations: list[pygrain.Transformation] = [
      transforms.RandomSection(
          feature_names=("u",),
          num_steps=num_time_steps,
          stride=time_stride,
      )
  ]
  if mean is not None and std is not None:
    operations.append(
        transforms.Normalize(
            feature_name="u", mean=np.asarray(mean), std=np.asarray(std)
        )
    )
  operations.append(
      transforms.SelectAs(select_features=("u",), as_features=("x",))
  )
  shard_options = pygrain.ShardByJaxProcess(drop_remainder=True)
  if batch_size == -1:  # Use full per-process shard as batch
    batch_size = len(source) // shard_options.shard_count
  if batch_size < 1:
    raise ValueError(f"`batch_size` ({batch_size}) must be positive.")
  operations.append(pygrain.Batch(batch_size=batch_size, drop_remainder=False))

  # Grain fine-tuning.
  worker_count = 0
  if tf_lookup_num_parallel_calls != -1:
    worker_count = tf_lookup_num_parallel_calls

  loader = pygrain.DataLoader(
      data_source=source,
      sampler=pygrain.IndexSampler(
          num_records=len(source),
          seed=seed,
          num_epochs=None,  # loads indefnitely
          shuffle=True,
          shard_options=shard_options,
      ),
      operations=operations,
      worker_count=worker_count,
      worker_buffer_size=_WORKER_BUFFER_SIZE,
      shard_options=shard_options,
      read_options=pygrain_sources.in_memory_read_options(),
  )
  return loader, {"mean": mean, "std": std}  # pyrefly: ignore[bad-assignment]
