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

"""Library for the Fourier Neural Operator (FNO).

Jax/flax implementation following the ICLR 2021 paper
(https://arxiv.org/abs/2010.08895) and the official github repo
(https://github.com/neuraloperator/neuraloperator).

Currently, only dense contraction is supported. Other types (included in
https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/spectral_convolution.py)
will be added in the future.
"""

from collections.abc import Callable
import enum
import math
import string
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# ********************
# Layers
# ********************

SYMBOLS = string.ascii_lowercase


def _contract_dense(x, weights, separable=False):
  """Dense contraction function."""
  x_subscripts = SYMBOLS[: x.ndim]
  if separable:  # no contraction: hadamard product between x and weights
    w_subscripts = x_subscripts[1:]
    out_subscripts = x_subscripts
  else:
    w_subscripts = x_subscripts[1:] + SYMBOLS[x.ndim]
    out_subscripts = x_subscripts[:-1] + w_subscripts[-1]
  subscripts = f"{x_subscripts},{w_subscripts}->{out_subscripts}"
  return jnp.einsum(subscripts, x, weights)


class ContractFnType(enum.Enum):
  DENSE = "dense"


contract_fn_dict = {
    ContractFnType.DENSE: _contract_dense,
}


class SpectralConv(nn.Module):
  """Generic n-dimensional spectral convolution.

  The dimension is indicated by `len(num_modes)`.

  Attributes:
    in_channels: The number of input channels.
    out_channels: The number of output channels.
    num_modes: The number of modes in the kernel, at most (Nx // 2 + 1); length
      must agree with the number of spatial dimensions.
    domain_size: The spatial dimensions of the inverse fft transform; if not
      specified, the input spatial dimensions are used.
    use_bias: Whether to add a bias after the spectral convolution.
    fft_norm: Arg to `jnp.fft.rfftn` and `jnp.fft.irfftn`; choose from
      `backward`, `ortho` or `forward`.
    contract_fn: The contraction function; only `dense` is supported atm. The
      factorized versions will be added later (has dependency on TensorLy).
    separable: Whether to use the separable version of the contraction.
    weights_dtype: The dtype of the contraction weights. Biases are always in
      dtype `jnp.float32` if enabled.
  """

  in_channels: int
  out_channels: int
  num_modes: tuple[int, ...]
  domain_size: tuple[int, ...] | None = None
  use_bias: bool = True
  fft_norm: Literal["backward", "ortho", "forward"] = "backward"
  contract_fn: ContractFnType = ContractFnType.DENSE
  separable: bool = False
  weights_dtype: jnp.dtype = jnp.complex64

  def setup(self):
    weights_shape = (
        *self.num_modes[:-1],
        self.num_modes[-1] // 2 + 1,
        self.in_channels,
    )
    if self.separable:
      if self.in_channels != self.out_channels:
        raise ValueError(
            "`in_channels` must be equal to `out_channels` for"
            " `separable=True`."
        )
    else:
      weights_shape += (self.out_channels,)

    scale = 1 / (self.in_channels * self.out_channels)
    initializer = nn.initializers.normal(stddev=scale)
    self.weights = self.param(
        "weights",
        initializer,
        weights_shape,
        self.weights_dtype,
    )
    bias_shape = (1,) * len(self.num_modes) + (self.out_channels,)
    self.bias = (
        self.param("bias", initializer, bias_shape, jnp.float32)
        if self.use_bias
        else None
    )

    self._contract_fn = contract_fn_dict[self.contract_fn]

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies generic spectral convolution based on input dimension."""
    _, *spatial_dims, in_channels = x.shape
    if len(spatial_dims) != len(self.num_modes):
      raise ValueError(
          f"Input spatial ndim ({len(spatial_dims)}) is inconsistent with"
          f" the ndim of `self.num_modes` ({len(self.num_modes)})."
      )

    if in_channels != self.in_channels:
      raise ValueError(
          f"Input `in_channels` ({in_channels}) is inconsistent with the"
          f" `self.in_channels` declaration ({self.in_channels})."
      )
    domain_size = self.domain_size or spatial_dims

    fft_axes = tuple(range(1, x.ndim - 1))
    x = jnp.fft.rfftn(x, axes=fft_axes, norm=self.fft_norm)

    # It is easier to slice after `fftshift` since low-frequency modes are
    # contiguous. In contrast, the original implementation extracts corners of
    # pre-shift coeffs one by one.
    x = jnp.fft.fftshift(x, axes=fft_axes[:-1])
    mode_slices = tuple(
        slice(x.shape[i] // 2 - m // 2, x.shape[i] // 2 + math.ceil(m / 2))
        for i, m in zip(fft_axes[:-1], self.num_modes[:-1])
    )
    # Add slices for the batch and the last spatial dimension.
    mode_slices = (
        slice(None),
        *mode_slices,
        slice(None, self.num_modes[-1] // 2 + 1),
    )
    x = jnp.fft.fftshift(x[mode_slices], axes=fft_axes[:-1])

    x = self._contract_fn(x, self.weights, separable=self.separable)
    x = jnp.fft.irfftn(x, s=domain_size, axes=fft_axes, norm=self.fft_norm)

    if self.use_bias:
      x = x + self.bias[None]

    return x


# ********************
# Blocks
# ********************


class FnoResBlock(nn.Module):
  """N-dimensional FNO residual block.

  The dimension is indicated by `len(num_modes)`.

  Attributes:
    out_channels: The number of output channels.
    num_modes: The number of modes used for spectral conv layers; see
      class::`SpectralConv.num_modes`.
    num_layers: The number of spectral conv layers in the block.
    fft_norm: arg to `jnp.fft.rfftn` and `jnp.fft.irfftn` in spectral conv
      layers.
    contract_fn: The type of contraction function to be used for spectral conv;
      see class::`SpectralConv.contract_fn`.
    separable: Whether to use a separable contraction function.
    act_fn: the activation function.
    skip_type: The type of skip connection to use - choose from ["linear",
      "soft-gate", "identity"].
    param_dtype: The dtype of model parameters.
  """

  out_channels: int
  num_modes: tuple[int, ...]
  num_layers: int = 1
  fft_norm: Literal["backward", "ortho", "forward"] = "forward"
  contract_fn: ContractFnType = ContractFnType.DENSE
  separable: bool = False
  act_fn: Callable[[jax.Array], jax.Array] = nn.swish
  skip_type: Literal["linear", "soft-gate", "identity"] = "soft-gate"
  param_dtype: jnp.dtype = jnp.complex64

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    h = x
    for _ in range(self.num_layers):
      h = nn.LayerNorm()(h)
      h = self.act_fn(h)
      h = SpectralConv(
          in_channels=h.shape[-1],
          out_channels=self.out_channels,
          num_modes=self.num_modes,
          fft_norm=self.fft_norm,
          contract_fn=self.contract_fn,
          separable=self.separable,
          weights_dtype=self.param_dtype,
      )(h)

    # Combine residual and skip connections.
    if self.skip_type == "linear":
      x = nn.Dense(features=self.out_channels, use_bias=False)(x)
    elif self.skip_type == "soft-gate":
      if x.shape[-1] != self.out_channels:
        raise ValueError(
            "`soft-gate` requires input and output channels to be the same."
        )
      shape = (1,) * len(self.num_modes) + (self.out_channels,)
      skip_weights = self.param(
          "skip_weights", nn.initializers.ones, shape, jnp.float32
      )
      x = x * skip_weights[None]
    return x + h


# ********************
# Operators
# ********************


class Fno(nn.Module):
  """N-dimensional FNO.

  The dimension is indicated by `len(num_modes)`.

  Attributes:
    out_channels: the number of output channels.
    hidden_channels: the number of hidden channels to use in all of the
      intermediate spectral conv layers.
    num_modes: the number of modes used for spectral conv layers; see
      class::`SpectralConv.num_modes`.
    lifting_channels: the number of hidden channels in the 2-layer MLP before
      applying the residual blocks. If `None`, the input is directly projected
      to `hidden_channels`.
    projection_channels: the number of hidden channels before projecting to the
      final output dimensions.
    num_blocks: the number of residual blocks in the model.
    layers_per_block: the number of spectral conv layers per residual block.
    block_skip_type: the type of skip connection for all blocks; see
      class::`FnoResBlock.skip_type`.
    fft_norm: arg to `jnp.fft.rfftn` and `jnp.fft.irfftn` in spectral conv
      layers.
    contract_fn: the type of contraction function to be used for spectral conv;
      see class::`SpectralConv.contract_fn`.
    separable: whether to use a separable contraction function.
    act_fn: the activation function for the Fno residual blocks.
    param_dtype: the dtype of model parameters.
  """

  out_channels: int
  hidden_channels: int = 64
  num_modes: tuple[int, ...] = (24, 24)  # The default is for 2D.
  lifting_channels: int | None = 256
  projection_channels: int = 256
  num_blocks: int = 4
  layers_per_block: int = 2
  block_skip_type: Literal["linear", "soft-gate", "identity"] = "soft-gate"
  fft_norm: Literal["backward", "ortho", "forward"] = "forward"
  contract_fn: ContractFnType = ContractFnType.DENSE
  separable: bool = False
  act_fn: Callable[[jax.Array], jax.Array] = nn.swish
  param_dtype: jnp.dtype = jnp.complex64

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    if self.lifting_channels:
      x = nn.Dense(features=self.lifting_channels)(x)
      x = self.act_fn(x)
    x = nn.Dense(features=self.hidden_channels)(x)

    # For non-periodic BC, pad zeros here and unpad after res blocks. See
    # https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/padding.py
    for _ in range(self.num_blocks):
      x = FnoResBlock(
          out_channels=self.hidden_channels,
          num_modes=self.num_modes,
          num_layers=self.layers_per_block,
          fft_norm=self.fft_norm,
          contract_fn=self.contract_fn,
          separable=self.separable,
          act_fn=self.act_fn,
          skip_type=self.block_skip_type,
          param_dtype=self.param_dtype,
      )(x)

    x = nn.Dense(features=self.projection_channels)(x)
    x = self.act_fn(x)
    x = nn.Dense(features=self.out_channels)(x)
    return x


class Fno2d(nn.Module):
  """2-dimensional FNO network.

  This network structure and default configs follow
  https://github.com/neuraloperator/markov_neural_operator/blob/main/models/fno_2d.py

  Attributes:
    out_channels: The number of output channels.
    num_modes: The base number of modes for the spectral conv layers (scaled
      with depth); see class::`SpectralConv.num_modes`.
    width: The base number of features in the intermediate layers (scaled with
      depth).
    num_layers: The number of spectral conv layers in the block.
    domain_size: Arg to the spectral conv layers; see
      class::`SpectralConv.domain_size`.
    fft_norm: Arg to `jnp.fft.rfftn` and `jnp.fft.irfftn` in spectral conv
      layers.
    act_fn: The activation function.
    param_dtype: The dtype of model parameters.
  """

  out_channels: int
  num_modes: tuple[int, int] = (20, 20)
  width: int = 128
  domain_size: tuple[int, int] | None = None
  fft_norm: Literal["backward", "ortho", "forward"] = "ortho"
  act_fn: Callable[[jax.Array], jax.Array] = jax.nn.selu
  param_dtype: jnp.dtype = jnp.complex64
  grid_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    batch_sz, *grid_size, _ = x.shape
    grid = self.get_grid(tuple(grid_size), dtype=self.grid_dtype)
    grid = jnp.tile(grid, (batch_sz,) + (1,) * (len(grid_size) + 1))

    # Scaling follows the reference repo in the class description
    widths = np.asarray([2, 3, 4, 4, 5], dtype=np.int32) * self.width // 4
    modes = np.outer(np.asarray([4, 3, 2, 2]), np.asarray(self.num_modes)) // 4
    kernel_sz = (1,) * len(grid_size)

    x = jnp.concatenate([x, grid], axis=-1)
    x = nn.Dense(widths[0])(x)

    for i in range(4):
      h1 = SpectralConv(
          in_channels=x.shape[-1],
          out_channels=widths[i + 1],
          domain_size=self.domain_size,
          num_modes=tuple(modes[i]),
          fft_norm=self.fft_norm,
      )(x)
      h2 = nn.Conv(features=widths[i + 1], kernel_size=kernel_sz)(x)
      x = h1 + h2
      if i < 4:
        x = self.act_fn(x)

    x = self.act_fn(nn.Dense(features=widths[-1] * 2)(x))
    x = self.act_fn(nn.Dense(features=widths[-1] * 2)(x))
    x = nn.Dense(features=self.out_channels)(x)
    return x

  def get_grid(self, grid_size: tuple[int, int], dtype: jnp.dtype) -> jax.Array:
    sz_x, sz_y = grid_size
    grid_x = jnp.expand_dims(
        jnp.linspace(0, 1, sz_x, endpoint=False, dtype=dtype), (1, 2)
    )
    grid_x = jnp.tile(grid_x, (1, sz_y, 1))
    grid_y = jnp.expand_dims(
        jnp.linspace(0, 1, sz_y, endpoint=False, dtype=dtype), (0, 2)
    )
    grid_y = jnp.tile(grid_y, (sz_x, 1, 1))
    return jnp.concatenate([grid_x, grid_y], axis=-1)
