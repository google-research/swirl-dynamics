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

"""Resizing modules."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.layers import convolutions

Array = jax.Array
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)


class FilteredResize(nn.Module):
  """A resizing op followed by a convolution layer.

  Attributes:
    output_size: The target output spatial dimensions for resizing.
    kernel_size: The kernel size of the convolution layer.
    method: The resizing method (passed to `jax.image.resize`).
    padding: The padding type of the convolutions, one of ['SAME', 'CIRCULAR',
      'LATLON', 'LONLAT].
    initializer: The initializer for the convolution kernels.
    use_local: Whether to use unshared weights in the filtering.
    precision: Level of precision used in the convolutional layer.
    dtype: The data type of the input and output.
    params_dtype: The data type of of the weights.
  """

  output_size: Sequence[int]
  kernel_size: Sequence[int]
  method: str = "cubic"
  padding: str = "CIRCULAR"
  initializer: nn.initializers.Initializer = jax.nn.initializers.normal(
      stddev=0.02
  )
  use_local: bool = False
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Resizes and filters the input with a convolution layer.

    Args:
      inputs: An input tensor of shape `(*batch_dims, *resized_dims, channels)`,
        where `batch_dims` can be or arbitrary length and `resized_dims` has the
        same length as that of `self.kernel_size`.

    Returns:
      The input resized to target shape.
    """
    if not inputs.ndim > len(self.output_size):
      raise ValueError(
          f"Number of dimensions in x ({inputs.ndim}) must be larger than the"
          f" length of `output_size` ({len(self.output_size)})!"
      )

    if self.padding not in ["CIRCULAR", "LATLON", "LONLAT", "SAME"]:
      raise ValueError(
          f"Unsupported padding type: {self.padding} - please use one of"
          " ['SAME', 'CIRCULAR', 'LATLON', 'LONLAT']!"
      )

    batch_ndim = inputs.ndim - len(self.output_size) - 1
    inputs = inputs.astype(self.dtype)
    resized = jax.image.resize(
        inputs,
        shape=(*inputs.shape[:batch_ndim], *self.output_size, inputs.shape[-1]),
        method=self.method,
    )
    # We add another convolution layer to undo any aliasing that could have
    # been introduced by the resizing step.
    out = convolutions.ConvLayer(
        features=inputs.shape[-1],
        kernel_size=self.kernel_size,
        padding=self.padding,
        kernel_init=self.initializer,
        use_local=self.use_local,
        dtype=self.dtype,
        precision=self.precision,
        param_dtype=self.param_dtype,
    )(resized)
    return out
