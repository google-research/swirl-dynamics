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

"""Measure distances.

1) Maximum mean discrepancy (MMD).
2) Sinkhorn divergence (sinkhorn_div).

References:

[1] https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html#implementation-of-mmd  # pylint: disable=line-too-long
[2] Li, Yujia, Kevin Swersky, and Rich Zemel. "Generative moment matching
  networks." International conference on machine learning. PMLR, 2015.
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp
from ott_jax.geometry import pointcloud
from ott_jax.tools import sinkhorn_divergence


Array = jax.Array
MeasureDistFn = Callable[[Array, Array], Array]


def mmd(
    x: Array,
    y: Array,
    bandwidth: tuple[float, ...] = (0.2, 0.5, 0.9, 1.3),
) -> Array:
  """Maximum Mean Discrepancy.

  Emprical maximum mean discrepancy. The lower the result the more evidence that
  distributions are the same.
  Input arrays are reshaped to dimension: `batch_size x -1`, where `-1`
  indicates that all non-batch dimensions are flattened.
  This implementation was adapted from [1].

  Args:
    x: first sample, distribution P
    y: second sample, distribution Q
    bandwidth: Multiscale levels for the bandwidth.

  Returns:
    mmd value.
  """
  # Samples x and y are of size `batch_size x state_space_dim`, e.g. for Lorenz
  # system `state_space_dim` is `3`, for KS it is `xspan x 1`, for NS it is
  # `h x w x 1`.
  # These arrays are then reshaped to be order two with shape
  # `batch_size x state_space_dim_flattened`.
  x = x.reshape((
      x.shape[0],
      -1,
  ))
  y = y.reshape((
      y.shape[0],
      -1,
  ))
  xx, yy, zz = jnp.matmul(x, x.T), jnp.matmul(y, y.T), jnp.matmul(x, y.T)
  rx = jnp.broadcast_to(jnp.expand_dims(jnp.diag(xx), axis=0), xx.shape)
  ry = jnp.broadcast_to(jnp.expand_dims(jnp.diag(yy), axis=0), yy.shape)

  dxx = rx.T + rx - 2.0 * xx
  dyy = ry.T + ry - 2.0 * yy
  dxy = rx.T + ry - 2.0 * zz

  xx, yy, xy = (jnp.zeros_like(xx), jnp.zeros_like(xx), jnp.zeros_like(xx))

  # Multiscale bandwisth.
  for a in bandwidth:
    xx += a**2 * (a**2 + dxx) ** -1
    yy += a**2 * (a**2 + dyy) ** -1
    xy += a**2 * (a**2 + dxy) ** -1

  # TODO: We may want to use jnp.sqrt here; see [2].
  return jnp.mean(xx + yy - 2.0 * xy)


def mmd_distributed(x: Array, y: Array) -> Array:
  """Distributed (multi-device) Maximum Mean Discrepancy.

  Empirical maximum mean discrepancy when using distributed learning.
  The lower the result the more evidence that distributions are the same.

  Args:
    x: first sample, distribution P
    y: second sample, distribution Q

  Returns:
    mmd value.
  """
  # Samples x and y are of size `batch_size x state_space_dim`, e.g. for Lorenz
  # system `state_space_dim` is `3` and for KS it is `512 x 1`.
  # The arrays are sharded along the other devices, so we need to gather them
  # and reshape them properly
  # `(num_device * batch_size, state_space_dim_flattened)`.
  x = jax.lax.all_gather(x, axis_name="device")
  y = jax.lax.all_gather(y, axis_name="device")

  x_shape = x.shape
  y_shape = y.shape

  # Merge the batch per device and device dimensions.
  x = jnp.reshape(x, (x_shape[0] * x_shape[1],) + x_shape[2:])
  y = jnp.reshape(y, (y_shape[0] * y_shape[1],) + y_shape[2:])

  return mmd(x, y)


def sinkhorn_div(x: Array, y: Array) -> Array:
  """Sinkhorn Divergence.

  Emprical sinkhorn divergence. The lower the result the more evidence that
  distributions are the same.
  Input arrays are reshaped to dimension: `batch_size x -1`, where `-1`
  indicates that all non-batch dimensions are flattened.

  Args:
    x: first sample, distribution P
    y: second sample, distribution Q

  Returns:
    sd value.
  """
  # Samples x and y are of size `batch_size x state_space_dim`, e.g. for Lorenz
  # system `state_space_dim` is `3`, for KS it is `xspan x 1`, for NS it is
  # `h x w x 1`.
  # These arrays are then reshaped to be order two with shape
  # `batch_size x state_space_dim_flattened`.
  ot = sinkhorn_divergence.sinkhorn_divergence(
      pointcloud.PointCloud,  # geom,
      x.reshape((x.shape[0], -1,)),  # geom.x,
      y.reshape((y.shape[0], -1,)),  # geom.y,
      static_b=False,
  )
  return jnp.array(ot.divergence)


def spatial_downsampled_dist(
    dist_fn: MeasureDistFn, x: Array, y: Array, spatial_downsample: int = 1
) -> float | Array:
  """Downsampled Divergence.

  Args:
    dist_fn: measure distance function
    x: first sample, distribution P
    y: second sample, distribution Q
    spatial_downsample: factor by which to downsample samples along spatial dim

  Returns:
    dist value.
  """
  # Samples x and y are of size `batch_size x state_space_dim`, e.g. for Lorenz
  # system `state_space_dim` is `3`, for KS it is `xspan x 1`, for NS it is
  # `h x w x 1`.
  if spatial_downsample > 1:
    if x.ndim == 2:
      x = x[:, ::spatial_downsample]
      y = y[:, ::spatial_downsample]
    elif x.ndim == 3:
      x = x[:, ::spatial_downsample, :]
      y = y[:, ::spatial_downsample, :]
    elif x.ndim == 4:
      x = x[:, ::spatial_downsample, ::spatial_downsample, :]
      y = y[:, ::spatial_downsample, ::spatial_downsample, :]
    else:
      raise NotImplementedError(
          f"Number of spatial dimensions {x.ndim - 1} not"
          " supported for spatial downsampling."
      )
  print(x.shape, y.shape)
  return dist_fn(x, y)
