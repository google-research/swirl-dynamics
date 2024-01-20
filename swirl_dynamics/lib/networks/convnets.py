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

"""Convolution-based modules for training dynamical systems.

1) PeriodicConvNetModel
"""
from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array


class DilatedBlock(nn.Module):
  """Implements a Dilated ConvNet block."""

  num_channels: int
  num_levels: int
  kernel_size: tuple[int, ...]
  act_fun: Callable[[jax.Array], jax.Array] | Any = nn.relu
  dtype: Any = jnp.float32
  padding: str = "CIRCULAR"

  @nn.compact
  def __call__(self, inputs: Array) -> Array:

    x = inputs.astype(self.dtype)

    # Ascending and descending order
    dilation_order = list(range(self.num_levels))
    dilation_order += list(range(self.num_levels-1))[::-1]
    for i in dilation_order:
      x = nn.Conv(features=self.num_channels,
                  kernel_size=self.kernel_size,
                  kernel_dilation=2**i,
                  padding=self.padding,
                  dtype=self.dtype)(x)
      x = self.act_fun(x)
    return x


class PeriodicConvNetModel(nn.Module):
  """Periodic ConvNet model.

  Simple convolutional model with skip connections, and dilated blocks. Based on
  the paper: Learned Coarse Models for Efficient Turbulence Simulation.

  Attributes:
    latent_dim: Dimension of the latent space in the processor.
    num_levels: Number of dilated convolutions, the larger the number of levels,
      the bigger receptives field the network will have.
    num_processors: Number of dilated blocks.
    encoder_kernel_size: Size of the kernel in the conv layer for the encoder.
    decoder_kernel_size: Size of the kernel in the conv layer for the decoder.
    processor_kernel_size: Size of the kernel in the conv layers inside the
      dilated convolutional blocks.
    act_fun: Activation function to be after each dilated block.
    norm_layer: Normalization layer to be applied after each dilated block.
    dtype: Type of input/layers.
    padding: Type of padding added to the convolutional layers depending on the
      geometry of underlying problem.
    is_input_residual: Boolean to use a global skip connection, so the
      architecture is similar to a Forward Euler integration rule.
  """
  latent_dim: int = 48
  num_levels: int = 4
  num_processors: int = 4
  encoder_kernel_size: tuple[int, ...] = (5,)
  decoder_kernel_size: tuple[int, ...] = (5,)
  processor_kernel_size: tuple[int, ...] = (5,)
  act_fun: Callable[[jax.Array], jax.Array] | Any = nn.relu
  norm_layer: Callable[[jax.Array], jax.Array] = lambda x: x  # default is ID
  dtype: Any = jnp.float32
  padding: str = "CIRCULAR"
  is_input_residual: bool = True

  @nn.compact
  def __call__(self, inputs: Array) -> Array:

    x = inputs.astype(self.dtype)

    # Encoder to latent dimension (larger than regular dimension).
    latent_x = nn.Conv(features=self.latent_dim,
                       kernel_size=self.encoder_kernel_size,
                       padding=self.padding,
                       dtype=self.dtype)(x)

    for _ in range(self.num_processors):
      y = DilatedBlock(num_channels=self.latent_dim,
                       kernel_size=self.processor_kernel_size,
                       num_levels=self.num_levels,
                       act_fun=self.act_fun)(latent_x)
      y = self.norm_layer(y)
      latent_x += y

    x = nn.Conv(features=inputs.shape[-1],
                kernel_size=self.decoder_kernel_size,
                padding=self.padding,
                dtype=self.dtype)(latent_x)

    # Last skip connection.
    if self.is_input_residual:
      x += inputs

    return x
