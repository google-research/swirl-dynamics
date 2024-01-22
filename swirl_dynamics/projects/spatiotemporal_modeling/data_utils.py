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

"""Simple dataloader generating a short video of a fluid simulation."""

import grain.tensorflow as tfgrain
import jax
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.data import tfgrain_transforms as transforms
import tensorflow as tf

Array = jax.Array


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
) -> tuple[tfgrain.TfDataLoader, dict[str, Array | None]]:
  """Load pre-computed trajectories dumped to hdf5 file.

  If normalize flag is set, method will also return the mean and std used in
  normalization (which are calculated from train split).

  Arguments:
    num_time_steps: Number of time steps to include in each trajectory. If set
      to -1, use entire trajectory lengths.
    time_stride: Stride of trajectory sampling.
    batch_size: Batch size returned by dataloader. If set to -1, use entire
      dataset size as batch_size.
    dataset_path: Absolute path to dataset file.
    seed: Random seed to be used in data sampling.
    split: Data split - train, eval, test, or None.
    spatial_downsample_factor: reduce spatial resolution by factor of x.
    normalize: Flag for adding data normalization (subtact mean divide by std.).
    normalize_stats: Dictionary with mean and std stats to avoid recomputing.
    use_time_normalization: Normalization is performed using a short sequence
        as a unit, instead of a single snaphot.
    tf_lookup_batch_size: Number of lookup batches (in cache) for grain.
    tf_lookup_num_parallel_calls: Number of parallel call for lookups in the
      dataset. -1 is set to let grain optimize tha number of calls.
    tf_interleaved_shuffle: Using a more localized shuffle instead of a global
      suffle of the data.

  Returns:
    loader, stats (optional): tuple of dataloader and dictionary containing
                              mean and std stats (if normalize=True, else dict
                              contains NoneType values).
  """
  snapshots, tspan = hdf5_utils.read_arrays_as_tuple(
      dataset_path, (f"{split}/u", f"{split}/t")
  )
  if spatial_downsample_factor > 1:
    if snapshots.ndim == 3:
      snapshots = snapshots[:, :, ::spatial_downsample_factor]
    elif snapshots.ndim == 4:
      snapshots = snapshots[:, :, ::spatial_downsample_factor, :]
    elif snapshots.ndim == 5:
      snapshots = snapshots[
          :, :, ::spatial_downsample_factor, ::spatial_downsample_factor, :
      ]
    else:
      raise NotImplementedError(
          f"Number of dimensions {snapshots.ndim} not "
          "supported for spatial downsampling."
      )

  if normalize:
    if normalize_stats is not None:
      mean = normalize_stats["mean"]
      std = normalize_stats["std"]
    else:
      if split != "train":
        data_for_stats = hdf5_utils.read_single_array(dataset_path, "train/u")
      else:
        data_for_stats = snapshots
      # TODO: For the sake of memory perform this in CPU.
      if use_time_normalization:
        num_trajs, num_frames, nx, ny, d = data_for_stats.shape
        num_segments = num_frames // num_time_steps
        data_for_stats = data_for_stats[:, :(num_segments * num_time_steps)]
        data_for_stats = np.reshape(
            data_for_stats, (num_trajs, num_segments, num_time_steps, nx, ny, d)
        )

      mean = np.mean(data_for_stats, axis=(0, 1))
      std = np.std(data_for_stats, axis=(0, 1))
    snapshots -= mean
    snapshots /= std
  else:
    mean, std = None, None
  source = tfgrain.TfInMemoryDataSource.from_dataset(
      tf.data.Dataset.from_tensor_slices({
          "u": snapshots,  # states
      })
  )
  # This transform randomly takes a random section from the trajectory with the
  # desired length and stride
  if num_time_steps == -1:  # Use full (downsampled) trajectories
    num_time_steps = tspan.shape[1] // time_stride
  section_transform = transforms.RandomSection(
      feature_names=("u",),
      num_steps=num_time_steps,
      stride=time_stride,
  )

  rename = transforms.SelectAs(select_features=("u",), as_features=("x",))

  dataset_transforms = (section_transform, rename)

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
          num_epochs=None,  # loads indefnitely
          shuffle=True,
          shard_options=tfgrain.ShardByJaxProcess(drop_remainder=True),
      ),
      transformations=dataset_transforms,
      batch_fn=tfgrain.TfBatch(batch_size=batch_size, drop_remainder=False),
  )
  return loader, {"mean": mean, "std": std}
