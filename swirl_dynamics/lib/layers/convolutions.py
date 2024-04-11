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

"""Convolution layers."""

from collections.abc import Sequence
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)


def ConvLayer(
    features: int,
    kernel_size: int | Sequence[int],
    padding: nn.linear.PaddingLike,
    use_local: bool = False,
    precision: PrecisionLike = None,
    dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
    **kwargs,
) -> nn.Module:
  """Factory for different types of convolution layers."""
  if isinstance(padding, str) and padding.lower() in ["lonlat", "latlon"]:
    if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
      raise ValueError(
          f"Kernel size {kernel_size} must be a length-2 tuple for convolution"
          f" type {padding}."
      )
    return LatLonConv(
        features,
        kernel_size,
        use_local=use_local,
        order=padding.lower(),
        precision=precision,
        dtype=dtype,
        param_dtype=param_dtype,
        **kwargs,
    )
  elif use_local:
    return nn.ConvLocal(
        features,
        kernel_size,
        padding=padding,
        precision=precision,
        dtype=dtype,
        param_dtype=param_dtype,
        **kwargs,
    )
  else:
    return nn.Conv(
        features,
        kernel_size,
        padding=padding,
        precision=precision,
        dtype=dtype,
        param_dtype=param_dtype,
        **kwargs,
    )


class LatLonConv(nn.Module):
  """2D convolutional layer adapted to inputs on a lat-lon grid."""

  features: int
  kernel_size: tuple[int, int] = (3, 3)
  order: Literal["latlon", "lonlat"] = "latlon"
  use_bias: bool = True
  kernel_init: nn.initializers.Initializer = nn.initializers.variance_scaling(
      scale=1.0, mode="fan_avg", distribution="uniform"
  )
  strides: tuple[int, int] = (1, 1)
  use_local: bool = False
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies lat-lon convolution.

    Padding is type "edge" in the latitudinal direction and type "circular"
    in the longitunidal direction.

    Args:
      inputs: Input data of at least 4 dimensions (*batch, lat, lon, channels)
        or (*batch, lon, lat, channels). The lat-lon (or lon-lat) axes are
        assumed at position (-3, -2). Multiple batch axes are supported.

    Returns:
      The convolved output.
    """
    if not (self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1):
      raise ValueError(f"Current kernel size {self.kernel_size} must be odd.")

    if inputs.ndim < 4:
      raise ValueError(f"Input must be 4D or higher: {inputs.shape}.")

    if self.order == "latlon":
      lat_axes, lon_axes = (-3, -2)
      lat_pad, lon_pad = self.kernel_size[0] // 2, self.kernel_size[1] // 2
    elif self.order == "lonlat":
      lon_axes, lat_axes = (-3, -2)
      lon_pad, lat_pad = self.kernel_size[1] // 2, self.kernel_size[0] // 2
    else:
      raise ValueError(
          f"Unrecognized order {self.order} - 'latlon' or 'lonlat' expected."
      )

    lon_pads = [(0, 0)] * inputs.ndim
    lon_pads[lon_axes] = (lon_pad, lon_pad)
    padded_inputs = jnp.pad(inputs, lon_pads, mode="wrap")

    lat_pads = [(0, 0)] * inputs.ndim
    lat_pads[lat_axes] = (lat_pad, lat_pad)
    padded_inputs = jnp.pad(padded_inputs, lat_pads, mode="edge")

    conv_fn = nn.ConvLocal if self.use_local else nn.Conv

    return conv_fn(
        self.features,
        kernel_size=self.kernel_size,
        use_bias=self.use_bias,
        strides=self.strides,
        kernel_init=self.kernel_init,
        padding="VALID",
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(padded_inputs)


class DownsampleConv(nn.Module):
  """Downsampling layer through strided convolution."""

  features: int
  ratios: Sequence[int]
  use_bias: bool = True
  kernel_init: nn.initializers.Initializer = nn.initializers.variance_scaling(
      scale=1.0, mode="fan_avg", distribution="uniform"
  )
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies strided convolution for downsampling."""
    if not inputs.ndim > len(self.ratios):
      raise ValueError(
          f"Inputs ({inputs.shape}) must have at least 1 more dimension than"
          f" that of `ratios` ({self.ratios})."
      )

    batch_ndims = inputs.ndim - len(self.ratios) - 1
    if not np.all(
        np.asarray(inputs.shape[batch_ndims:-1]) % np.asarray(self.ratios) == 0
    ):
      raise ValueError(
          f"Input dimensions (spatial) {inputs.shape[batch_ndims:-1]} must"
          f" divide the downsampling ratio {self.ratios}."
      )

    return nn.Conv(
        self.features,
        kernel_size=self.ratios,
        use_bias=self.use_bias,
        strides=self.ratios,
        kernel_init=self.kernel_init,
        padding="VALID",
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(inputs)
