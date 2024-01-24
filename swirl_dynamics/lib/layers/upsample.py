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

"""Upsampling modules."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


def channel_to_space(inputs: Array, block_shape: Sequence[int]) -> Array:
  """Reshapes data from the channel to spatial dims as a way to upsample.

  As an example, for an input of shape (*batch, x, y, z) and block_shape of
  (a, b), additional spatial dimensions are first formed from the channel
  dimension (always the last one), i.e. reshaped into
  (*batch, x, y, a, b, z//(a*b)). Then the new axes are interleaved with the
  original ones to arrive at shape (*batch, x, a, y, b, z//(a*b)). Finally, the
  new axes are merged with the original axes to yield final shape
  (*batch, x*a, y*b, z//(a*b)).

  Args:
    inputs: The input array to upsample.
    block_shape: The shape of the block that will be formed from the channel
      dimension. The number of elements (i.e. prod(block_shape) must divide the
      number of channels).

  Returns:
    The upsampled array.
  """
  if not inputs.ndim > len(block_shape):
    raise ValueError(
        f"Ndim of `x` ({inputs.ndim}) expected to be higher than the length of"
        f" `block_shape` {len(block_shape)}."
    )

  if inputs.shape[-1] % np.prod(block_shape) != 0:
    raise ValueError(
        f"The number of channels in the input ({inputs.shape[-1]}) must be"
        f" divisible by the block size ({np.prod(block_shape)})."
    )

  # Create additional spatial axes from channel.
  old_shape = inputs.shape
  batch_ndim = inputs.ndim - len(block_shape) - 1
  cout = old_shape[-1] // np.prod(block_shape)
  x = jnp.reshape(inputs, old_shape[:-1] + tuple(block_shape) + (cout,))

  # Interleave old and new spatial axes.
  spatial_axes = np.arange(2 * len(block_shape), dtype=np.int32) + batch_ndim
  new_axes = spatial_axes.reshape(2, -1).ravel(order="F")
  x = jnp.transpose(
      x,
      tuple(range(batch_ndim))
      + tuple(new_axes)
      + (len(new_axes) + batch_ndim,),
  )

  # Merge the interleaved axes.
  new_shape = np.asarray(old_shape[batch_ndim:-1]) * np.asarray(block_shape)
  new_shape = old_shape[:batch_ndim] + tuple(new_shape) + (cout,)
  return jnp.reshape(x, new_shape)
