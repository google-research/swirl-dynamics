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

"""Modeling utilities.

References:

[1] Li, Zongyi, et al. "Markov neural operators for learning chaotic systems."
  arXiv preprint arXiv:2106.06898 (2021): 25.
"""

from collections.abc import Callable
from typing import Any

import grain.python as pygrain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.data import pygrain_sources
from swirl_dynamics.data import pygrain_transforms as transforms
from swirl_dynamics.lib.solvers import ode


Array = jax.Array
PyTree = Any
DynamicsFn = Callable[[Array, Array, PyTree], Array]

# Number of batches each PyGrain worker process prepares in advance (only used
# when a loader runs with worker processes).
_WORKER_BUFFER_SIZE = 2

# These used to be defined in this module; the aliases keep the old names
# working now that the implementations are shared across projects.
ArrayDataSource = pygrain_sources.ArrayDataSource
RandomSection = transforms.RandomSection


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


def _normalize_snapshots(
    snapshots: np.ndarray,
    dataset_path: str,
    split: str | None,
    spatial_downsample_factor: int,
    normalize_stats: dict[str, Array] | None,
) -> tuple[np.ndarray, Any, Any]:
  """Normalizes `snapshots` using provided or training-split statistics."""
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
    mean = np.mean(data_for_stats, axis=(0, 1))
    std = np.std(data_for_stats, axis=(0, 1))
  mean_cast = np.asarray(mean, dtype=snapshots.dtype)
  std_cast = np.asarray(std, dtype=snapshots.dtype)
  return (snapshots - mean_cast) / std_cast, mean, std


# TODO: Move this method to swirl_dynamics.data.utils
def generate_data_from_known_dynamics(
    integrator: ode.ScanOdeSolver,
    dynamics: DynamicsFn,
    num_steps: int,
    dt: float,
    warmup: int,
    x0: Array,
) -> Array:
  """Generates data from known dynamics for on-the-fly ground truth batches."""
  num_steps += warmup
  tspan = jnp.arange(num_steps) * dt
  return integrator(dynamics, x0, tspan, {})[warmup:]  # pyrefly: ignore[bad-argument-type]


def create_loader_from_hdf5_reshaped(
    num_time_steps: int,
    time_stride: int,
    batch_size: int,
    dataset_path: str,
    seed: int = 0,
    split: str | None = None,
    spatial_downsample_factor: int = 1,
    normalize: bool = False,
    normalize_stats: dict[str, Array] | None = None,
    tf_lookup_batch_size: int = 1024,
    tf_lookup_num_parallel_calls: int = -1,
) -> tuple[pygrain.DataLoader, dict[str, Array | None]]:
  """Load pre-computed trajectories dumped to hdf5 file.

  If normalize flag is set, method will also return the mean and std used in
  normalization (which are calculated from train split).

  Arguments:
    num_time_steps: Number of time steps to include in each trajectory. If set
      to -1, use entire trajectory lengths.
    time_stride: Stride of trajectory sampling (downsampling by that factor).
    batch_size: Batch size returned by dataloader. If set to -1, use entire
      dataset size as batch_size.
    dataset_path: Absolute path to dataset file. We assume that the file
      contains two arrays for each split (train, eval, test, or None), named u
      and t, respectively.
    seed: Random seed to be used in data sampling.
    split: Data split - train, eval, test, or None.
    spatial_downsample_factor: reduce spatial resolution by factor of x.
    normalize: Flag for adding data normalization (subtract mean divide by
      std.).
    normalize_stats: Dictionary with mean and std stats to avoid recomputing.
    tf_lookup_batch_size: Unused. Kept for compatibility with existing configs.
    tf_lookup_num_parallel_calls: Number of PyGrain worker processes loading the
      data. -1 (default) loads the data in the main process.

  Returns:
    loader, stats (optional): tuple of dataloader and dictionary containing
                              mean and std stats (if normalize=True, else dict
                              contains NoneType values).
  """
  del tf_lookup_batch_size
  split_prefix = f"{split}/" if split is not None else ""
  snapshots, tspan = hdf5_utils.read_arrays_as_tuple(
      dataset_path, (f"{split_prefix}u", f"{split_prefix}t")
  )
  snapshots = _downsample_spatial(snapshots, spatial_downsample_factor)

  if normalize:
    snapshots, mean, std = _normalize_snapshots(
        snapshots,
        dataset_path,
        split,
        spatial_downsample_factor,
        normalize_stats,
    )
  else:
    mean, std = None, None

  if num_time_steps == -1:  # Use full (downsampled) trajectories
    num_time_steps = tspan.shape[1] // time_stride

  # Downsampling the number of snapshots in time.
  snapshots = snapshots[:, ::time_stride]
  tspan = tspan[:, ::time_stride]

  # Reshaping the data to have num_time_steps time steps per trajectory.
  num_segments_per_traj = snapshots.shape[1] // num_time_steps
  snapshots = snapshots[:, : (num_segments_per_traj * num_time_steps)]
  snapshots = np.reshape(snapshots, (-1, num_time_steps, *snapshots.shape[2:]))

  tspan = tspan[:, : (num_segments_per_traj * num_time_steps)]
  tspan = np.reshape(tspan, (-1, num_time_steps, *tspan.shape[2:]))

  source = pygrain_sources.ArrayDataSource(u=snapshots, t=tspan)
  worker_count = (
      0 if tf_lookup_num_parallel_calls == -1 else tf_lookup_num_parallel_calls
  )
  if batch_size == -1:  # Use full dataset as batch
    batch_size = len(source)

  shard_options = pygrain.ShardByJaxProcess(drop_remainder=True)
  loader = pygrain.DataLoader(
      data_source=source,
      sampler=pygrain.IndexSampler(
          num_records=len(source),
          shuffle=True,
          seed=seed,
          shard_options=shard_options,
          num_epochs=None,
      ),
      operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
      worker_count=worker_count,
      worker_buffer_size=_WORKER_BUFFER_SIZE,
      shard_options=shard_options,
      read_options=pygrain_sources.in_memory_read_options(),
  )
  return loader, {"mean": mean, "std": std}


# TODO: Move this method to swirl_dynamics.data.utils
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
    tf_lookup_batch_size: int = 1024,
    tf_lookup_num_parallel_calls: int = -1,
) -> tuple[pygrain.DataLoader, dict[str, Array | None]]:
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
    normalize: Flag for adding data normalization (subtract mean divide by
      std.).
    normalize_stats: Dictionary with mean and std stats to avoid recomputing.
    tf_lookup_batch_size: Unused. Kept for compatibility with existing configs.
    tf_lookup_num_parallel_calls: Number of PyGrain worker processes loading the
      data. -1 (default) loads the data in the main process.

  Returns:
    loader, stats (optional): tuple of dataloader and dictionary containing
                              mean and std stats (if normalize=True, else dict
                              contains NoneType values).
  """
  del tf_lookup_batch_size
  split_prefix = f"{split}/" if split is not None else ""
  snapshots, tspan = hdf5_utils.read_arrays_as_tuple(
      dataset_path, (f"{split_prefix}u", f"{split_prefix}t")
  )
  snapshots = _downsample_spatial(snapshots, spatial_downsample_factor)

  if normalize:
    snapshots, mean, std = _normalize_snapshots(
        snapshots,
        dataset_path,
        split,
        spatial_downsample_factor,
        normalize_stats,
    )
  else:
    mean, std = None, None

  source = pygrain_sources.ArrayDataSource(u=snapshots, t=tspan)

  if num_time_steps == -1:  # Use full (downsampled) trajectories.
    num_time_steps = tspan.shape[1] // time_stride
  section_transform = transforms.RandomSection(
      feature_names=("u", "t"),
      num_steps=num_time_steps,
      stride=time_stride,
  )

  worker_count = (
      0 if tf_lookup_num_parallel_calls == -1 else tf_lookup_num_parallel_calls
  )
  if batch_size == -1:  # Use full dataset as batch.
    batch_size = len(source)

  shard_options = pygrain.ShardByJaxProcess(drop_remainder=True)
  loader = pygrain.DataLoader(
      data_source=source,
      sampler=pygrain.IndexSampler(
          num_records=len(source),
          shuffle=True,
          seed=seed,
          shard_options=shard_options,
          num_epochs=None,
      ),
      operations=[
          section_transform,
          pygrain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
      worker_count=worker_count,
      worker_buffer_size=_WORKER_BUFFER_SIZE,
      shard_options=shard_options,
      read_options=pygrain_sources.in_memory_read_options(),
  )
  return loader, {"mean": mean, "std": std}


def create_loader_from_tfds(
    num_time_steps: int,
    time_stride: int,
    batch_size: int,
    seed: int,
    dataset_path: str,
    dataset_name: str,
    split: str | None = None,
    normalize: bool = False,
    tf_lookup_batch_size: int = 8192,
    tf_lookup_num_parallel_calls: int = -1,
    tf_interleaved_shuffle: bool = False,
) -> tuple[pygrain.DataLoader, dict[str, Array | None]]:
  """Load trajectories from TFDS.

  This loader has fewer options than the one from hdf5, in particular, it has
  no normalization. TODO: Add normalization.

  The dataset is read with random access (through `tfds.data_source`), which
  requires it to be stored in the ArrayRecord file format.

  Arguments:
    num_time_steps: Number of time steps to include in each trajectory.
    time_stride: Stride of trajectory sampling.
    batch_size: Batch size returned by dataloader.
    seed: Random seed to be used in data sampling.
    dataset_path: Absolute path to dataset tfds folder.
    dataset_name: Name of the dataset.
    split: Data split chosen from train, eval or test.
    normalize: Flag for adding data normalization (subtract mean, divide by
      std).
    tf_lookup_batch_size: Unused. It tuned the previous tf.data pipeline and is
      kept for compatibility with existing configs.
    tf_lookup_num_parallel_calls: Number of PyGrain worker processes loading the
      data. -1 (default) loads the data in the main process.
    tf_interleaved_shuffle: Unused. It tuned the previous tf.data pipeline and
      is kept for compatibility with existing configs.

  Returns:
    loader, stats (optional): Tuple of dataloader and dictionary containing
                              mean and std stats (if normalize=True, else dict
                              contains NoneType values).
  """
  del tf_lookup_batch_size, tf_interleaved_shuffle  # Unused.
  if normalize:
    raise NotImplementedError("This loader does not support normalization.")

  if num_time_steps <= 0:
    raise ValueError(f"num_time_steps must be > 0, instead of {num_time_steps}")

  if batch_size <= 0:
    raise ValueError(f"batch_size must be > 0, instead of {batch_size}")

  if split is None:
    raise ValueError("A split must be specified to load data from TFDS.")

  # TFDS is only needed by this loader, so it is imported lazily (it is not a
  # dependency of the swirl_dynamics package).
  import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top

  source = tfds.data_source(
      dataset_name, split=split, data_dir=dataset_path, download=False
  )

  section_transform = transforms.RandomSection(
      feature_names=("u", "t"),
      num_steps=num_time_steps,
      stride=time_stride,
  )

  worker_count = (
      0 if tf_lookup_num_parallel_calls == -1 else tf_lookup_num_parallel_calls
  )

  shard_options = pygrain.ShardByJaxProcess(drop_remainder=True)
  loader = pygrain.DataLoader(
      data_source=source,
      sampler=pygrain.IndexSampler(
          num_records=len(source),
          shuffle=True,
          seed=seed,
          shard_options=shard_options,
          num_epochs=None,
      ),
      operations=[
          section_transform,
          pygrain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
      worker_count=worker_count,
      worker_buffer_size=_WORKER_BUFFER_SIZE,
      shard_options=shard_options,
  )

  return loader, {"mean": None, "std": None}


# TODO: find a better place for this function and refactor with
# vmap.
def sobolev_norm(
    u: Array, s: int = 1, dim: int = 2, length: float = 1.0
) -> Array:
  r"""Sobolev Norm computed via the Plancherel equality following [1].

  Arguments:
    u: Input to compute the norm (n_batch, n_frames, n_x, n_y, d).
    s: Order of the Sobolev norm.
    dim: Dimension of the input (either 1 or 2).
    length: The length of the domain (we assume that we have a square.)

  Returns:
    The average H^s squared along trajectories and batch size.

  We compute the Sobolov norm using the Fourier Transform following:
  \| x \|_{H^s}^2 = \int (\sum_{i=0}^s \|k\|^2i)) | \hat{u}(k) |^2 dk.

  In particular, we assemble the multipliers, and we approximate the quadrature
  using a trapezoidal rule.
  """
  n_x = u.shape[-2]
  k_x = jnp.fft.fftfreq(n_x, length / (2 * jnp.pi * n_x))

  # Reusing the same expression for both one and two dimensional.
  axes = (-2,) if dim == 1 else (-3, -2)
  u_fft = jnp.fft.fftn(u, axes=axes)

  # Computing the base multiplier: \| k \|^2.
  if dim == 1:
    multiplier = jnp.square(k_x)
    multiplier = multiplier.reshape(
        len(u.shape[:-2]) * (1,) + (n_x, u.shape[-1])
    )
  elif dim == 2:
    k_x, k_y = jnp.meshgrid(k_x, k_x)
    multiplier = jnp.square(k_x) + jnp.square(k_y)
    multiplier = multiplier.reshape(
        len(u.shape[:-3]) * (1,) + (n_x, n_x, u.shape[-1])
    )
  else:
    raise ValueError(f"Unsupported dim: {dim}")

  # Computing the different in Fourier space | \hat{u}(k) |^2.
  u_fft_squared = jnp.square(jnp.abs(u_fft))

  # Building the set of multipliers following:
  # \left ( \sum_{i=0}^s \| k \|^{2i} \right).
  mult = jnp.sum(
      jnp.power(
          multiplier[..., None],  # add an extra dimension for broadcasting.
          jnp.arange(s + 1).reshape(len(multiplier.shape) * (1,) + (-1,)),
      ),
      axis=-1,
  )

  # Performing the integration using trapezoidal rule.
  norm_squared = jnp.sum(mult * u_fft_squared, axis=axes) / (n_x) ** dim

  # Returns the mean.
  return jnp.mean(norm_squared)


def rmse(upred: Array, utrue: Array, axis: int | tuple[int, ...]) -> Array:
  """Root mean squared error (RMSE)."""
  rmse_t = jnp.sqrt(jnp.nanmean(jnp.square(upred - utrue), axis=axis))
  return jnp.nanmean(rmse_t, axis=0)


def relrmse(upred: Array, utrue: Array, axis: int | tuple[int, ...]) -> Array:
  """Relative root mean squared error (relrmse)."""
  relrmse_t = jnp.sqrt(jnp.sum(jnp.square(upred - utrue), axis=axis))
  relrmse_t /= jnp.sqrt(jnp.sum(jnp.square(utrue), axis=axis))
  return jnp.nanmean(relrmse_t, axis=0)


def plot_error_metrics(
    dt,
    traj_length,
    trajs,
    pred_trajs,
    metrics=("rmse", "rel_rmse"),
):
  """Plot selected error metrics over time."""
  metric_disply = {"rmse": "RMSE", "rel_rmse": "Relative RMSE"}
  metric_fns = {"rmse": rmse, "rel_rmse": relrmse}

  plot_time = jnp.arange(traj_length) * dt
  t_max = plot_time.max()
  errs = {}
  for metric in metrics:
    errs[metric] = metric_fns[metric](pred_trajs, trajs, axis=(2, 3))
  figsize = (3 + 4 * len(metrics), 4)
  fig, ax = plt.subplots(
      1, len(metrics), figsize=figsize, sharey=True, tight_layout=True
  )
  if not hasattr(ax, "__iter__"):
    ax = (ax,)
  for i, metric in enumerate(metrics):
    ax[i].plot(plot_time, errs[metric])
    ax[i].set_xlim([0, t_max])
    ax[i].set_xlabel(r"$t$")
    ax[i].set_title(metric_disply[metric])
    ax[i].set_yscale("log")
  return {"err": fig}


def sample_uniform_spherical_shell(
    n_points: int,
    radii: tuple[float, float],
    shape: tuple[int, ...],
    key: Array,
):
  """Uniform sampling (in angle and radius) from an spherical shell.

  Arguments:
    n_points: Number of points to sample.
    radii: Interior and exterior radii of the spherical shell.
    shape: Shape of the points to sample.
    key: Random key for generating the random numbers.

  Returns:
    A vector of size (n_points,) + shape, within the spherical shell. The
    vector is chosen uniformly in both angle and radius.
  """

  inner_radius, outer_radius = radii

  # Shape to help broadcasting.
  broadcasting_shape = (n_points,) + len(shape) * (1,)
  # Obtain the correct axis for the sum, depending on the shape.
  # Here we suppose that shape comes in the form (nx, ny, d) or (nx, d).
  assert len(shape) < 4 and len(shape) >= 2, (
      "The shape should represent one- or two-dimensional points."
      f" Instead we have shape {shape}"
  )

  axis_sum = (1,) if len(shape) == 2 else (1, 2)

  key_radius, key_vec = jax.random.split(key)

  sampling_radius = jax.random.uniform(
      key_radius, (n_points,), minval=inner_radius, maxval=outer_radius
  )
  vec = jax.random.normal(key_vec, shape=((n_points,) + shape))

  vec_norm = jnp.linalg.norm(vec, axis=axis_sum).reshape(broadcasting_shape)
  vec /= vec_norm

  return vec * sampling_radius.reshape(broadcasting_shape)


def linear_scale_dissipative_target(inputs: Array, scale: float = 1.0):
  """Function to rescale the random input to bias the model.

  Arguments:
    inputs: Input point in state space.
    scale: Real number in [0, 1] that scales down input.

  Returns:
    The rescaled input, scale*input.

  This function is implemented to follow the implementation of the Markov Neural
  Operator (MNO).
  """
  return scale * inputs


def plot_cos_sims(dt: Array, traj_length: int, trajs: Array, pred_trajs: Array):
  """Plot cosine similarities over time."""

  def sum_non_batch_dims(x: Array) -> Array:
    """Helper method to sum array along all dimensions except the 0th."""
    ndim = x.ndim
    return x.sum(axis=tuple(range(1, ndim)))

  def state_cos_sim(x: Array, y: Array) -> Array:
    """Compute cosine similiarity between two batches of states.

      Computes x^Ty / ||x||*||y|| averaged across batch dimension (axis = 0).

    Args:
      x: array of states; shape: batch_size x state_dimension
      y: array of states; shape: batch_size x state_dimension

    Returns:
      cosine similarity averaged along batch dimension.
    """
    x_norm = jnp.expand_dims(
        jnp.sqrt(sum_non_batch_dims((x**2))), axis=tuple(range(1, x.ndim))
    )
    x /= x_norm
    y_norm = jnp.expand_dims(
        jnp.sqrt(sum_non_batch_dims((y**2))), axis=tuple(range(1, y.ndim))
    )
    y /= y_norm
    return sum_non_batch_dims(x * y).mean(axis=0)

  plot_time = jnp.arange(traj_length) * dt
  t_max = plot_time.max()
  fig, ax = plt.subplots(1, 1, figsize=(7, 4), tight_layout=True)

  # Plot 0.9, 0.8 threshold lines.
  ax.plot(
      plot_time,
      jnp.ones(traj_length) * 0.9,
      color="black",
      linestyle="dashed",
      label="0.9 threshold",
  )
  ax.plot(
      plot_time,
      jnp.ones(traj_length) * 0.8,
      color="red",
      linestyle="dashed",
      label="0.8 threshold",
  )

  # Plot correlation lines.
  cosine_sims = jax.vmap(state_cos_sim, in_axes=(1, 1))(
      trajs[:, :traj_length, :], pred_trajs[:, :traj_length, :]
  )
  ax.plot(plot_time, cosine_sims)
  ax.set_xlim([0, t_max])
  ax.set_xlabel(r"$t$")
  ax.set_ylabel("Avg. cosine sim.")
  ax.set_title("Cosine Similiarity over time")
  ax.legend(frameon=False, bbox_to_anchor=(1, 1))
  return {"cosine_sim": fig}
