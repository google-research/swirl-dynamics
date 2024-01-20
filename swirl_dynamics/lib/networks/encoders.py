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

"""Hyper-networks encoders.

Implementation of the encoders used in [1], in which the input is a snapshot of
the state in an equispaced grid, and the output is the weights/parameters of
the decoder ansatz.

References:
[1] Z. Y. Wan, L. Zepeda-Núñez, A. Boral and F.Sha, Evolve Smoothly, Fit
Consistently: Learning Smooth Latent Dynamics For Advection-Dominated Systems,
submitted to ICLR2023. "https://openreview.net/forum?id=Z4s73sJYQM"
"""
from collections.abc import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array


class ResNetBlock1D(nn.Module):
  """Simple convolutional ResNetBlock in 1D.

  Attributes:
    features: Number of output features.
    kernel_size: Size of the filter in space.
    act_fn: Activation function after each convolutional + batch normalization
      block.
    downsample: Boolean for downsampling the features in space (by a factor 2)
    padding: Type of padding for the convolutional layers. The default is
      "CIRCULAR", which implements periodic boundary conditions.
    dtype: Type of the inputs and parameters. The default is float32.
  """

  features: int
  kernel_size: tuple[int, ...] = (5,)
  act_fn: Callable[[jax.Array], jax.Array] = nn.tanh
  downsample: bool = False
  padding: str = "CIRCULAR"
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jax.Array, is_training: bool = True) -> jax.Array:
    """Application of the ResNet block.

    Args:
      x: Input array.
      is_training: Boolean for batch normalization.

    Returns:
      Array normalized after two convolutions plus an skip connection.
    """

    x = x.astype(self.dtype)

    z = nn.Conv(
        features=self.features,
        kernel_size=self.kernel_size,
        use_bias=False,
        strides=(1,) if not self.downsample else (2,),
        padding=self.padding,
        param_dtype=self.dtype,
    )(x)

    # Normalization and non-linear activation.
    z = nn.BatchNorm()(z, use_running_average=not is_training)
    z = self.act_fn(z)

    z = nn.Conv(
        features=self.features,
        kernel_size=self.kernel_size,
        use_bias=False,
        strides=(1,),
        padding=self.padding,
        param_dtype=self.dtype,
    )(z)

    z = nn.BatchNorm()(z, use_running_average=not is_training)

    # Downsampling the input to match the output dimension.
    if self.downsample:
      x = nn.Conv(
          features=self.features,
          kernel_size=(1,),
          strides=(2,),
          use_bias=False,
          padding=self.padding,
          param_dtype=self.dtype,
      )(x)

    # Adding the skip connection.
    x_out = self.act_fn(z + x)

    return x_out


class EncoderResNet(nn.Module):
  """Convolutional encoder with periodic boundary conditions.

  The encoder takes an one-dimensional signal, and it increasingly downsample it
  in space, by applying several convolutional resnet layers. Each time that it
  is downsampled the channel dimension is doubles preserve the overall
  information in the signal. The signal is downsampled num_levels times, and
  at each level is is processed by num_resnet_blocks number of resnet blocks.

  After the signal has been donwsampled, the signal is reshaped as a vector, and
  a dense network is used to generate the final encoding vector of dimension
  dim_out.

  Attributes:
    filter: Base number of filter for the convolutional networks. Then increase
      by a factor a two by every downsampling by a factor of 2
    dim_out: Dimensionality of the output. This is usually the number of
      training parameters (or weights) in the ansatz.
    num_levels: Number of downsampling levels
    num_resnet_blocks: Number of resnets blocks for each level.
    act: Activation function after each convolutional + batch normalization
      block.
    padding: Type of padding for the convolutional layers. The default is
      "CIRCULAR", which implements periodic boundary conditions.
    dtype: Type of the inputs and parameters. The default is float 32
  """

  filters: int
  dim_out: int
  num_levels: int = 4
  num_resnet_blocks: int = 1
  kernel_size: tuple[int, ...] = (5,)
  act_fn: Callable[[jax.Array], jax.Array] = nn.tanh
  padding: str = "CIRCULAR"
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jax.Array, is_training: bool = True) -> jax.Array:
    assert self.num_levels > 0, (
        "Number of levels should be greater than 0"
        + " instead we have "
        + str(self.num_levels)
    )

    x = x.astype(self.dtype)

    for level_idx in range(self.num_levels):
      # Downsampling the features.
      x = ResNetBlock1D(
          features=self.filters * (2**level_idx),
          kernel_size=self.kernel_size,
          act_fn=self.act_fn,
          downsample=True,
          padding=self.padding,
          dtype=self.dtype,
      )(x, is_training=is_training)

      # Applying the rest of the chain of ResNet blocks.
      for _ in range(self.num_resnet_blocks - 1):
        x = ResNetBlock1D(
            features=self.filters * (2**level_idx),
            act_fn=self.act_fn,
            padding=self.padding,
            dtype=self.dtype,
        )(x, is_training=is_training)

    # Reshaping and applying a dense layer at the end.
    x = nn.Dense(features=self.dim_out, param_dtype=self.dtype)(
        x.reshape(-1, (x.shape[-2] * x.shape[-1]))
    )

    return x
