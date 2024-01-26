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

"""CycleGAN module.

References:
[1]
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190#issuecomment-358546675
# pylint: disable=line-too-long
"""

from collections.abc import Callable
import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets


Pytree = Any
Array = jax.Array
Initializer = nn.initializers.Initializer


class FilteredInterpolation(nn.Module):
  """Filtered interpolation layer to minimize spurious artifacts.

  Attributes:
    height: Target height (in number of pixels) of the output.
    width: Target width (in number of pixels) of the output.
    output_nc: Number of features for the ouput (the same as the input).
    interpolation_method: Specific type of interpolation scheme.
    padding: Type of padding for the convolutions.
    initializer: Initialization function.
    use_local: Whether to use unshared weights in the filtering.
    dtype: Data type of the interpolation and neural networks weights.
  """

  height: int
  width: int
  output_nc: int = 1
  interpolation_method: str = "bicubic"
  padding: str = "CIRCULAR"
  initializer: Initializer = jax.nn.initializers.normal(stddev=0.02)
  use_local: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    x = x.astype(self.dtype)
    x = functools.partial(
        jax.image.resize,
        shape=(
            x.shape[0],
            self.height,
            self.width,
            self.output_nc,
        ),
        method=self.interpolation_method,
    )(x)

    # We add another convolution layers to undo any aliasing that could have
    # been introduced by the interpolation step.
    x = layers.ConvLayer(
        features=self.output_nc,
        kernel_size=(7, 7),
        strides=[1, 1],
        padding=self.padding,  # this is still a large convolutional layer.
        kernel_init=self.initializer,
        use_local=self.use_local,
    )(x)
    return x


class ResnetBlock(nn.Module):
  """Simple ResNet block.

  Attributes:
    features: Number of features for the ouput (the same as the input).
    dropout_layer: Specific type of of dropout layer.
    padding: Type of padding for the convolutions.
    act_fun: Activation function.
    dtype: Type of the inputs and weights.
    initializer: Initialization function.
  """

  features: int
  dropout_layer: Callable[..., Any]
  padding: str = "CIRCULAR"
  kernel_size: tuple[int, int] = (3, 3)
  act_fun: Callable[[Array], Array] | Any = nn.relu
  dtype: jnp.dtype = jnp.float32
  use_bias: bool = True
  initializer: Initializer = jax.nn.initializers.normal(stddev=0.02)
  normalization_layer: Callable[..., Any] = functools.partial(
      nn.GroupNorm, num_groups=None, group_size=1
  )

  @nn.compact
  def __call__(self, x: Array) -> Array:
    x = x.astype(self.dtype)
    assert x.shape[-1] == self.features, (
        "Input and ouput features don't match. Input number of features is",
        f"{x.shape[-1]}, while output is {self.features}",
    )

    # Skip connection.
    x_skip = x

    x = layers.ConvLayer(
        features=self.features,
        kernel_size=self.kernel_size,
        padding=self.padding,
        kernel_init=self.initializer,
        use_bias=self.use_bias,
    )(x)
    x = self.normalization_layer()(x)
    x = self.act_fun(x)

    x = self.dropout_layer()(x)
    x = layers.ConvLayer(
        features=self.features,
        kernel_size=self.kernel_size,
        padding=self.padding,
        kernel_init=self.initializer,
    )(x)

    x = self.normalization_layer()(x)

    return x + x_skip


class Generator(nn.Module):
  """The generator for the CycleGAN A->B.

  See [1] for further details on the upsampling.

  Attributes:
    output_nc: The number of channels in output images.
    ngf: The number of filters in the last conv layer.
    n_res_blocks: The number of ResNet blocks at the core (lowest) level.
    n_res_blocks_level: The number of ResNet blocks at each skip level (non core
      level).
    dropout_rate: The rate for dropout sampling.
    upsample_mode: Modality of upsampling: deconvolution or bilinear
      interpolation.
    n_downsample_layers: Number of dowsampling layers.
    act_fun: Activation functions.
    final_act_fun: Final activation function.
    kernel_size_downsampling: Size of the kernel for the downsamplign layers.
    kernel_size_upsampling: Size of the kernel for the upsampling layers.
    n_downsample_layers: Number of downsampling levels, each level produces a
      downsmapling of factor 2 in each dimension (x, y) while increasing the
      number of channels by a factor 2.
    n_upsample_layers: Number of downsampling levels, each level produces a
      upsampling of factor 2 in each dimension (x, y) while decreasing the
      number of channels by a factor 2.
    use_skips: Using skip connections to have a U-net type network.
    use_global_skip: Using a skip connection between input and output.
    padding: Type of padding used in the convolutional layers.
    padding_transpose: Type of padding used for the tranpose convolutional
      layers when performing the upsampling.
    use_weight_global_skip: Use a weight in the global skip.
    weight_skip: To either weight the skip or the output of the network.
    use_local: Use locally connected convolutional layers, i.e., with unshared
      weights. This only used for the projection to/from tokens.
    interpolated_shape: Shapes in the case we need to change the spatial
      dimensions of the input.
    initializer: Function for randomly initializing the parameters.
  """

  output_nc: int = 1
  ngf: int = 32
  n_res_blocks: int = 6
  n_res_blocks_level: int = 0
  dropout_rate: float = 0.5
  upsample_mode: str = "deconv"
  act_fun: Callable[[Array], Array] = nn.relu
  final_act_fun: Callable[[Array], Array] = nn.activation.tanh
  kernel_size_downsampling: tuple[int, int] = (3, 3)
  kernel_size_upsampling: tuple[int, int] = (3, 3)
  kernel_size_core: tuple[int, int] = (3, 3)
  use_attention: bool = False
  use_position_encoding: bool = False
  num_heads: int = 4
  n_downsample_layers: int = 2
  n_upsample_layers: int = 2
  use_skips: bool = True
  use_global_skip: bool = True
  dtype: jnp.dtype = jnp.float32
  padding: str = "CIRCULAR"  # TODO: Add one adapted for ERA5.
  padding_transpose: str = "CIRCULAR"
  use_weight_global_skip: bool = False
  weight_skip: bool = False
  use_local: bool = False
  interpolated_shape: tuple[int, int] | None = None
  interpolation_method: str = "bicubic"
  initializer: Initializer = jax.nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x: Array, is_training: bool) -> Array:

    # Perfoming sanity check.
    if self.upsample_mode not in ["bilinear", "deconv"]:
      raise NotImplementedError(
          "Generator upsample_mode [%s] is not recognized" % self.upsample_mode
      )

    # Saving for a skip connection.
    input_x = x
    # Kernel dimension for the positional embedding.
    kernel_dim = x.ndim - 2

    batch_size, input_height, input_width, input_channels = x.shape

    # Interpolate the input to the desired shape.
    if self.interpolated_shape:
      x = functools.partial(
          jax.image.resize,
          shape=(
              batch_size,
              *self.interpolated_shape,
              input_channels,
          ),
          method=self.interpolation_method,
      )(x)

    # Projection layer.
    x = layers.ConvLayer(
        features=self.ngf,
        kernel_size=(7, 7),
        strides=[1, 1],
        padding=self.padding,
        kernel_init=self.initializer,
        use_local=self.use_local,
    )(x)
    x = nn.GroupNorm(num_groups=None, group_size=1)(x)
    x = self.act_fun(x)

    # Downsampling layers.
    horizontal_skips = []

    for i in range(self.n_downsample_layers):
      # At each level we increase the channel dimension by a factor 2
      # and decimate the space by a factor 2.
      # save the skip
      horizontal_skips.append(x)

      # Downsampling layer.
      mult = 2 ** (i + 1)
      x = layers.ConvLayer(
          features=self.ngf * mult,
          kernel_size=self.kernel_size_downsampling,
          strides=[2, 2],
          padding=self.padding,
          kernel_init=self.initializer,
      )(x)
      x = nn.GroupNorm(num_groups=None, group_size=1)(x)
      x = self.act_fun(x)

    # Core (lowest) level.
    # Multiplier of the base number of features.
    mult = 2**self.n_downsample_layers
    for i in range(self.n_res_blocks):
      x = ResnetBlock(
          features=self.ngf * mult,
          kernel_size=self.kernel_size_core,
          padding=self.padding,
          dropout_layer=functools.partial(
              nn.Dropout, rate=self.dropout_rate, deterministic=not is_training
          ),
          initializer=self.initializer,
          name=f"resnet_block_core_number_{i}",
      )(x)

      # Use a transformer core.
      # TODO add a conformer model.
      if self.use_attention:
        b, *hw, c = x.shape
        # Adding positional encoding.
        if self.use_position_encoding:
          x = unets.position_embedding(
              ndim=kernel_dim,
              name=f"position_embedding_number_{i}",
          )(x)
        x = unets.AttentionBlock(
            num_heads=self.num_heads,
            name=f"attention_block_number_{i}",
        )(x.reshape(b, -1, c), is_training=is_training)
        x = unets.ResConv1x(
            hidden_layer_size=self.ngf * mult * 2,
            out_channels=self.ngf * mult,
            name=f"reprojection_number_{i}",
        )(x).reshape(b, *hw, c)

    # Upsampling layers.
    # Reversing the skip connections so they are in the correct order.
    horizontal_skips.reverse()

    for i in range(self.n_upsample_layers):
      # Channel multiplier (level dependent).
      mult = 2 ** (self.n_upsample_layers - i)

      # Upsampling by a factor 2 in each spatial dimension.
      if self.upsample_mode == "bilinear":
        # Upsampling using bilinear interpolation.
        x = functools.partial(
            jax.image.resize,
            shape=(
                x.shape[0],
                x.shape[1] * 2,
                x.shape[2] * 2,
                x.shape[3],
            ),
            method="bilinear",
        )(x)
        x = layers.ConvLayer(
            features=(self.ngf * mult) // 2,
            kernel_size=self.kernel_size_upsampling,
            strides=[1, 1],
            padding=self.padding,
            kernel_init=self.initializer,
        )(x)

      elif self.upsample_mode == "deconv":
        # TODO: use channel unrolling for the upsampling.
        x = nn.ConvTranspose(
            features=(self.ngf * mult) // 2,
            kernel_size=self.kernel_size_upsampling,
            strides=[2, 2],
            padding=self.padding_transpose,
            kernel_init=self.initializer,
        )(x)

      x = nn.GroupNorm(num_groups=None, group_size=1)(x)
      x = self.act_fun(x)

      if self.use_skips and i < len(horizontal_skips):
        # If using skips we concatenate the branches before going into the
        # the convolutional layer which will merge the information.

        y = horizontal_skips[i]
        # Further processing each skip branch.
        for j in range(self.n_res_blocks_level):
          y = ResnetBlock(
              features=y.shape[-1],
              kernel_size=self.kernel_size_core,
              padding=self.padding,
              dropout_layer=functools.partial(
                  nn.Dropout,
                  rate=self.dropout_rate,
                  deterministic=not is_training,
              ),
              initializer=self.initializer,
              name=f"resnet_level_{i}_number_{j}",
          )(y)

        x = jnp.concatenate([x, y], axis=-1)
        x = layers.ConvLayer(
            features=(self.ngf * mult) // 2,
            kernel_size=self.kernel_size_upsampling,
            strides=[1, 1],
            padding=self.padding,
            kernel_init=self.initializer,
        )(x)
        x = nn.GroupNorm(num_groups=None, group_size=1)(x)
        x = self.act_fun(x)

    # Last convolution layer to correct the number of output channels.
    x = layers.ConvLayer(
        features=self.output_nc,
        kernel_size=(7, 7),
        strides=[1, 1],
        padding=self.padding,
        kernel_init=self.initializer,
        use_local=self.use_local,
    )(x)

    # Interpolating back to the original size.
    if self.interpolated_shape:
      x = FilteredInterpolation(
          height=input_height,
          width=input_width,
          output_nc=self.output_nc,
          interpolation_method=self.interpolation_method,
          padding=self.padding,
          initializer=self.initializer,
          use_local=self.use_local,
      )(x)

    # Add skip connection between generator input and output.
    # Reference: https://github.com/leehomyc/cyclegan-1
    if (
        self.n_upsample_layers == self.n_downsample_layers
        and self.use_global_skip
    ):
      if self.use_weight_global_skip:
        init_weight_global = jax.nn.initializers.constant(jnp.array([0.001]))
        # Initializing the weight.
        weight_global_skip = self.param(
            "weight_global_skip", init_weight_global, (1,), self.dtype
        )

        # Applying the weight to either the output or the skip.
        if self.weight_skip:
          input_x = weight_global_skip * input_x
        else:
          x = weight_global_skip * x

      x += input_x

    x = self.final_act_fun(x)

    return x


class Discriminator(nn.Module):
  """Discriminator module to predict the class of the input.

  The discriminator would take an image input and predict if it's an original
  or the output from the generator.

  Attributes:
      base_features: Number of filters in the first conv layer
      n_layers: The number of conv layers in the discriminator.
      kernel_size: Size of the convolutional kernel.
      padding: Type of padding. An integer input is translated to a uniform
        padding. A string argument means the type of padding.
      use_bias: Flag for using biases after each convolutional layer.
      use_local: Flag for using locally connected layers (This has more dof).
      initializer: Initialization method.
  """

  base_features: int = 64
  n_layers: int = 3
  kernel_size: tuple[int, int] = (5, 5)
  padding: int | str = 1
  use_bias: bool = False
  use_local: bool = False
  initializer: Initializer = jax.nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x: Array) -> Array:
    x = layers.ConvLayer(
        features=self.base_features,
        kernel_size=self.kernel_size,
        strides=(2, 2),
        padding=self.padding,
        kernel_init=self.initializer,
        use_local=self.use_local,
    )(x)
    x = nn.PReLU(negative_slope_init=0.2)(x)

    # Hierarchical processing.
    for n in range(1, self.n_layers):
      # Gradually increase the number of filters/features as we downsample
      # in space.
      feature_multiplier = min(2**n, 8)
      x = layers.ConvLayer(
          features=self.base_features * feature_multiplier,
          kernel_size=self.kernel_size,
          strides=(2, 2),
          padding=self.padding,
          use_bias=self.use_bias,
          kernel_init=self.initializer,
      )(x)
      x = nn.GroupNorm(num_groups=None, group_size=1)(x)
      x = nn.PReLU(negative_slope_init=0.2)(x)

    feature_multiplier = min(2**self.n_layers, 8)
    x = layers.ConvLayer(
        features=self.base_features * feature_multiplier,
        kernel_size=self.kernel_size,
        strides=(2, 2),
        padding=self.padding,
        use_bias=self.use_bias,
        kernel_init=self.initializer,
    )(x)
    x = nn.GroupNorm(num_groups=None, group_size=1)(x)
    x = nn.PReLU(negative_slope_init=0.2)(x)

    # The output should be just one channel.
    x = layers.ConvLayer(
        features=1,
        kernel_size=self.kernel_size,
        strides=(1, 1),
        padding=self.padding,
        use_local=self.use_local,
    )(x)

    return x
