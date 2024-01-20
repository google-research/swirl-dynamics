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

"""Utilities for reshaping from tokens to spatial coordinates."""

from typing import Sequence

import jax
import jax.numpy as jnp

Array = jax.Array
Shape = Sequence[int]


def reshape_2d_to_1d_factorized(x: Array, axis: int) -> Array:
  """Converts 2d inputs to 1d for axial attention."""

  if x.ndim != 4:
    raise ValueError(
        'The input should a 4-tensor with dimensions following '
        '(batch_size, height, width, channel), instead '
        f'the input shape is {x.shape}'
    )
  if axis not in [1, 2]:
    raise ValueError(
        'The input axis should be either 1 (time) or 2 (space), ',
        f'instead the axis provided as {axis}.',
    )

  batch_size, height, width, channel = x.shape
  x = jnp.moveaxis(x, axis, -2)

  if axis == 1:
    x = jnp.reshape(x, (batch_size * width, height, channel))
  elif axis == 2:
    x = jnp.reshape(x, (batch_size * height, width, channel))

  return x


def reshape_3d_to_1d_factorized(x: Array, axis: int) -> Array:
  """Converts 2d inputs to 1d for axial attention."""

  if x.ndim != 5:
    raise ValueError('The input should a 5-tensor with dimensions following '
                     '[batch_size, time, height, width, channel], instead '
                     f'the input shape is {x.shape}')
  if axis not in [1, 2, 3]:
    raise ValueError(
        'The input axis should be either 1 (time), 2 (height), or 3 (width), ',
        f'instead the axis provided as {axis}.',
    )

  batch_size, time, height, width, channel = x.shape

  x = jnp.moveaxis(x, axis, -2)

  if axis == 1:
    x = jnp.reshape(x, (batch_size * height * width, time, channel))
  elif axis == 2:
    x = jnp.reshape(x, (batch_size * time * width, height, channel))
  elif axis == 3:
    x = jnp.reshape(x, (batch_size * time * height, width, channel))

  return x


def reshape_to_2d_factorized(x: Array, axis: int,
                             two_d_shape: Shape) -> Array:
  """Converts 1D inputs back to 2D after axial attention.

  Args:
    x: Array to reshape.
    axis: Axis in which the reshaping will be computed. This coincides with the
        axis in which the axial attention was computed.
    two_d_shape: Original shape of the tensor (batch_size, height, width,
        emb_dim).

  Returns:
    Tensor reshaped following the 2D topology.
  """
  if x.ndim != 3:
    raise ValueError(
        'The input dimention should be a 3-tensor with dimensions',
        ' following [batch_size*height, width, channel], if axis = 2 or',
        ' [batch_size*width, height, channel], if axis = 1, instead',
        f'the shape of the input is {x.shape}',
    )
  if len(two_d_shape) != 4:
    raise ValueError(
        'The target tensor should be a 4-tensor with dimensions following',
        '(batch_size, height, width, channel), instead ',
        f'the shape of the output is chosed to be {two_d_shape}',
    )

  batch_size, height, width, channel = two_d_shape
  if axis == 1:
    if x.shape[0] != batch_size * width:
      raise ValueError(
          f'The modified batch size of the input ({x.shape[0]}) should match ',
          f'with the product batch_size ({batch_size}) x width ({width})',
      )
    x = x.reshape((batch_size, width, height, channel)).transpose(
        (0, 2, 1, 3)
    )
  elif axis == 2:
    if x.shape[0] != batch_size * height:
      raise ValueError(
          f'The modified batch size of the input ({x.shape[0]}) should match ',
          f'with the product batch_size ({batch_size}) x height ({height})',
      )
    x = x.reshape(two_d_shape)

  return x


def reshape_to_3d_factorized(
    x: Array, axis: int,
    three_d_shape: Shape
) -> Array:
  """Converts 1D inputs back to 3D after axial attention.

  Args:
    x: Array to reshape.
    axis: Axis in which the reshaping will be computed. This coincides with the
        axis in which the axial attention was computed.
    three_d_shape: Original shape of the tensor (batch_size, time, height,
        width, emb_dim).

  Returns:
    Tensor reshaped following the 3D topology.
  """
  # Sanity checks.
  if x.ndim != 3:
    raise ValueError(
        'The input dimention should be a 3-tensor with dimensions following',
        '(batch_size, num_tokes, emb_dim), instead ',
        f'the shape of the input is {x.shape}',
    )
  if len(three_d_shape) != 5:
    raise ValueError(
        'The target tensor should be a 5-tensor with dimensions following',
        '(batch_size, time, height, width, channel), instead ',
        f'the shape of the output is chosed to be {three_d_shape}',
    )

  batch_size, time, height, width, channel = three_d_shape

  if axis == 1:
    if x.shape[0] != batch_size * height * width:
      raise ValueError(
          f'The modified batch size of the input ({x.shape[0]}) should match ',
          f'with the product batch_size ({batch_size}) x width ({width}) x ',
          f'height ({height}) when axis = {axis}'
      )
    x = x.reshape((batch_size, height, width, time, channel)).transpose(
        (0, 3, 1, 2, 4))
  elif axis == 2:
    if x.shape[0] != batch_size * time * width:
      raise ValueError(
          f'The modified batch size of the input ({x.shape[0]}) should match ',
          f'with the product batch_size ({batch_size}) x time ({time}) x ',
          f'width ({width}) when axis = {axis}'
      )
    x = x.reshape((batch_size, time, width, height, channel)).transpose(
        (0, 1, 3, 2, 4))
  elif axis == 3:
    if x.shape[0] != batch_size * time * height:
      raise ValueError(
          f'The modified batch size of the input ({x.shape[0]}) should match ',
          f'with the product batch_size ({batch_size}) x time ({time}) x ',
          f'width ({height}) when axis = {axis}',
      )
    x = x.reshape(three_d_shape)

  return x


def reshape_to_time_space(x: Array, temporal_dims: int)-> Array:
  """Reshape the input tensor from tokens to time-space.

  Args:
    x: Three dimensional array of shape (batch_size, num_tokens, emb_dim).
    temporal_dims: Number of time frames after tokenization.

  Returns:
    The input array with the time frames explicit in its shape.
  """

  if x.ndim != 3:
    raise ValueError(
        'The input tensor should be 3-dimensional (including batch dimension)',
        f'instead the shape of the input tensor if {x.shape}')

  elif x.ndim == 3:
    batch_size, num_tokens, emb_dim = x.shape
    if num_tokens % temporal_dims != 0:
      raise ValueError(
          'The number of tokens should be divisible by the number of',
          f'encoded frames, instead we have number of tokens {num_tokens}',
          f' and number of frames {temporal_dims}')

    hw = num_tokens // temporal_dims
    x = jnp.reshape(x, [batch_size, temporal_dims, hw, emb_dim])

  # Last check.
  if x.ndim != 4:
    raise ValueError(
        'The output should be a 4-tensor, instead the output has shape'
        f' {x.shape}.')

  return x
