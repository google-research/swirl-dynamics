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

"""CycleGAN networks (generator, discriminator and filtered interpolation).

The networks accept inputs of shape `(batch, *spatial, channels)` with one or
two spatial dimensions; every kernel or stride given as a 2-tuple is reduced to
the spatial rank of the input (see `_spatial_tuple`), so 2-D callers keep
their exact behaviour and 1-D signals work out of the box.

References:
[1] Upsampling discussion of the reference implementation:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
"""

from collections.abc import Callable, Sequence
import functools
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets

Pytree = Any
Array = jax.Array
DTypeLike = jax.typing.DTypeLike
Initializer = nn.initializers.Initializer


def _check_spatial_ndim(x: Array) -> int:
  """Returns the spatial rank of `x` (`x.ndim - 2`), validating 1-D or 2-D."""
  ndim = x.ndim - 2
  if ndim not in (1, 2):
    raise ValueError(
        f"Expected 1-D or 2-D spatial input (rank 3 or 4), got shape {x.shape}."
    )
  return ndim


def _spatial_tuple(value: int | Sequence[int], ndim: int) -> tuple[int, ...]:
  """Broadcasts a kernel or stride spec to the spatial rank `ndim`.

  Legacy 2-D defaults such as `(k, k)` are accepted for `ndim == 1` and
  reduced to `(k,)`, so existing callers do not have to change their configs.

  Args:
    value: An integer or a sequence of integers.
    ndim: Number of spatial dimensions of the input.

  Returns:
    A tuple of length `ndim`.

  Raises:
    ValueError: If a non-uniform sequence cannot be matched to `ndim`.
  """
  if isinstance(value, int):
    return (value,) * ndim
  value = tuple(value)
  if len(value) == ndim:
    return value
  if ndim == 1 and len(set(value)) == 1:
    return value[:1]
  raise ValueError(
      f"Cannot use spatial spec {value!r} with an input of spatial rank {ndim}."
  )


def _instance_norm(x: Array) -> Array:
  """Normalises each channel of each sample over its spatial axes.

  Args:
    x: Feature map of shape `(batch, *spatial, channels)`.

  Returns:
    The instance-normalised feature map.

  Raises:
    ValueError: If the map has a single spatial element in total, in which
      case the normalised map (and anything computed from it) is identically 0.
  """
  if math.prod(x.shape[1:-1]) == 1:
    raise ValueError(
        "Instance normalisation of a feature map with a single spatial"
        f" element is identically 0 (shape {x.shape}); use fewer layers or a"
        " larger input."
    )
  return nn.GroupNorm(num_groups=None, group_size=1)(x)


_NORM_TYPES = ("instance", "none", "group")


def _apply_norm(x: Array, norm_type: str) -> Array:
  """Per-sample normalisation of a feature map (no cross-sample coupling).

  Args:
    x: Feature map of shape `(batch, *spatial, channels)`.
    norm_type: `"instance"` (one channel per group, the historical default),
      `"none"` (identity) or `"group"` (up to 8 groups of at least 4 channels).

  Returns:
    The normalised feature map.

  Raises:
    ValueError: For an unknown `norm_type`.
  """
  if norm_type == "none":
    return x
  if norm_type == "instance":
    return _instance_norm(x)
  if norm_type == "group":
    num_groups = max(1, min(8, x.shape[-1] // 4))
    while x.shape[-1] % num_groups:
      num_groups -= 1
    if math.prod(x.shape[1:-1]) * (x.shape[-1] // num_groups) == 1:
      raise ValueError(
          "Group normalisation of a feature map with a single element per"
          f" group is identically 0 (shape {x.shape}); use fewer layers or a"
          " larger input."
      )
    return nn.GroupNorm(num_groups=num_groups)(x)
  raise ValueError(
      f"Unsupported norm_type {norm_type!r}; expected one of {_NORM_TYPES}."
  )


class FilteredInterpolation(nn.Module):
  """Filtered interpolation layer to minimize spurious artifacts.

  Attributes:
    height: Deprecated 2-D alias of `target_shape`; target height (in number of
      pixels).
    width: Deprecated 2-D alias of `target_shape`; target width (in number of
      pixels).
    output_nc: Number of features for the ouput (the same as the input).
    interpolation_method: Specific type of interpolation scheme.
    padding: Type of padding for the convolutions.
    initializer: Initialization function.
    use_local: Whether to use unshared weights in the filtering.
    dtype: Data type of the interpolation and neural networks weights.
    target_shape: Target spatial shape of the output, one entry per spatial
      dimension. Required unless `height` and `width` are given.
  """

  height: int | None = None
  width: int | None = None
  output_nc: int = 1
  interpolation_method: str = "bicubic"
  padding: str = "CIRCULAR"
  initializer: Initializer = jax.nn.initializers.normal(stddev=0.02)
  use_local: bool = False
  dtype: DTypeLike = jnp.float32
  target_shape: tuple[int, ...] | None = None

  def _resolved_target_shape(self) -> tuple[int, ...]:
    """Returns `target_shape`, or `(height, width)` for legacy callers."""
    if self.target_shape is not None:
      return tuple(self.target_shape)
    if self.height is None or self.width is None:
      raise ValueError(
          "FilteredInterpolation needs `target_shape` (or both `height` and"
          " `width`)."
      )
    return (self.height, self.width)

  @nn.compact
  def __call__(self, x: Array) -> Array:
    target_shape = self._resolved_target_shape()
    ndim = len(target_shape)
    if x.ndim != ndim + 2:
      raise ValueError(
          f"Input of shape {x.shape} does not match the target spatial shape"
          f" {target_shape}."
      )
    x = x.astype(self.dtype)
    # Only the spatial axes are interpolated; the convolution below maps the
    # channels to `output_nc`.
    x = functools.partial(
        jax.image.resize,
        shape=(x.shape[0], *target_shape, x.shape[-1]),
        method=self.interpolation_method,
    )(x)

    # We add another convolution layers to undo any aliasing that could have
    # been introduced by the interpolation step.
    x = layers.ConvLayer(
        features=self.output_nc,
        kernel_size=(7,) * ndim,
        strides=(1,) * ndim,
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
    kernel_size: Size of the convolutional kernels (reduced to the spatial rank
      of the input).
    act_fun: Activation function.
    dtype: Type of the inputs and weights.
    use_bias: Whether the first convolution has a bias.
    initializer: Initialization function.
    normalization_layer: Factory of the normalisation layers used when
      `norm_type == "instance"` (ignored for other `norm_type` values).
    norm_type: Normalisation after each convolution, see `_apply_norm`.
  """

  features: int
  dropout_layer: Callable[..., Any]
  padding: str = "CIRCULAR"
  kernel_size: int | Sequence[int] = (3, 3)
  act_fun: Callable[[Array], Array] | Any = nn.relu
  dtype: DTypeLike = jnp.float32
  use_bias: bool = True
  initializer: Initializer = jax.nn.initializers.normal(stddev=0.02)
  normalization_layer: Callable[..., Any] = functools.partial(
      nn.GroupNorm, num_groups=None, group_size=1
  )
  norm_type: str = "instance"

  def _normalize(self, x: Array) -> Array:
    if self.norm_type == "instance":
      return self.normalization_layer()(x)
    return _apply_norm(x, self.norm_type)

  @nn.compact
  def __call__(self, x: Array) -> Array:
    x = x.astype(self.dtype)
    if x.shape[-1] != self.features:
      raise ValueError(
          f"Input features ({x.shape[-1]}) must match the block features"
          f" ({self.features})."
      )
    kernel_size = _spatial_tuple(self.kernel_size, x.ndim - 2)

    # Skip connection.
    x_skip = x

    x = layers.ConvLayer(
        features=self.features,
        kernel_size=kernel_size,
        padding=self.padding,
        kernel_init=self.initializer,
        use_bias=self.use_bias,
    )(x)
    x = self._normalize(x)
    x = self.act_fun(x)

    x = self.dropout_layer()(x)
    x = layers.ConvLayer(
        features=self.features,
        kernel_size=kernel_size,
        padding=self.padding,
        kernel_init=self.initializer,
        use_bias=self.use_bias,
    )(x)

    x = self._normalize(x)

    return x + x_skip


class Generator(nn.Module):
  """The generator for the CycleGAN A->B.

  See [1] for further details on the upsampling. Inputs have shape
  `(batch, *spatial, channels)` with one or two spatial dimensions.

  Attributes:
    output_nc: The number of channels in output images.
    ngf: The number of filters in the last conv layer.
    n_res_blocks: The number of ResNet blocks at the core (lowest) level.
    n_res_blocks_level: The number of ResNet blocks at each skip level (non core
      level).
    dropout_rate: The rate for dropout sampling.
    upsample_mode: Modality of upsampling: deconvolution or bilinear
      interpolation.
    act_fun: Activation functions.
    final_act_fun: Final activation function.
    kernel_size_downsampling: Size of the kernel for the downsampling layers
      (reduced to the spatial rank of the input).
    kernel_size_upsampling: Size of the kernel for the upsampling layers.
    kernel_size_core: Size of the kernel of the core ResNet blocks.
    use_attention: Whether to add an attention block after each core ResNet
      block.
    use_position_encoding: Whether to add a positional encoding before the
      attention blocks.
    num_heads: Number of attention heads.
    n_downsample_layers: Number of downsampling levels, each level produces a
      downsmapling of factor 2 in each spatial dimension while increasing the
      number of channels by a factor 2.
    n_upsample_layers: Number of upsampling levels, each level produces an
      upsampling of factor 2 in each spatial dimension while decreasing the
      number of channels by a factor 2.
    use_skips: Using skip connections to have a U-net type network.
    use_global_skip: Using a skip connection between input and output. Requires
      as many input channels as `output_nc`.
    dtype: Data type of the parameters.
    padding: Type of padding used in the convolutional layers.
    padding_transpose: Type of padding used for the tranpose convolutional
      layers when performing the upsampling.
    use_weight_global_skip: Use a weight in the global skip.
    weight_skip: To either weight the skip or the output of the network.
    use_local: Use locally connected convolutional layers, i.e., with unshared
      weights. This only used for the projection to/from tokens.
    interpolated_shape: Spatial shape on which the network operates when it
      differs from the input shape; the output is interpolated back.
    interpolation_method: Interpolation scheme of `interpolated_shape`.
    initializer: Function for randomly initializing the parameters.
    norm_type: Normalisation after every convolution of the trunk and inside the
      ResNet blocks: `"instance"` (default), `"none"` or `"group"`. With
      instance normalisation the first normalisation removes the per-channel
      mean and scale of the stem output, so the trunk is insensitive to the
      amplitude and the mean of its input up to the normalisation epsilon and
      padding effects (the global skip passes both through unchanged); use
      `"none"` for amplitude-aware generators, e.g. on physical fields.
  """

  output_nc: int = 1
  ngf: int = 32
  n_res_blocks: int = 6
  n_res_blocks_level: int = 0
  dropout_rate: float = 0.5
  upsample_mode: str = "deconv"
  act_fun: Callable[[Array], Array] = nn.relu
  final_act_fun: Callable[[Array], Array] = nn.activation.tanh
  kernel_size_downsampling: int | Sequence[int] = (3, 3)
  kernel_size_upsampling: int | Sequence[int] = (3, 3)
  kernel_size_core: int | Sequence[int] = (3, 3)
  use_attention: bool = False
  use_position_encoding: bool = False
  num_heads: int = 4
  n_downsample_layers: int = 2
  n_upsample_layers: int = 2
  use_skips: bool = True
  use_global_skip: bool = True
  dtype: DTypeLike = jnp.float32
  padding: str = "CIRCULAR"  # TODO: Add one adapted for ERA5.
  padding_transpose: str = "CIRCULAR"
  use_weight_global_skip: bool = False
  weight_skip: bool = False
  use_local: bool = False
  interpolated_shape: tuple[int, ...] | None = None
  interpolation_method: str = "bicubic"
  initializer: Initializer = jax.nn.initializers.normal(stddev=0.02)
  norm_type: str = "instance"

  @nn.compact
  def __call__(self, x: Array, is_training: bool) -> Array:

    # Perfoming sanity check.
    if self.upsample_mode not in ["bilinear", "deconv"]:
      raise NotImplementedError(
          "Generator upsample_mode [%s] is not recognized" % self.upsample_mode
      )

    # Saving for a skip connection.
    input_x = x
    ndim = _check_spatial_ndim(x)
    # Kernel dimension for the positional embedding.
    kernel_dim = ndim

    batch_size, *input_spatial, input_channels = x.shape
    applies_global_skip = (
        self.n_upsample_layers == self.n_downsample_layers
        and self.use_global_skip
    )
    if applies_global_skip and input_channels != self.output_nc:
      raise ValueError(
          "use_global_skip requires as many input channels as output_nc, got"
          f" {input_channels} != {self.output_nc}; set use_global_skip=False."
      )

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
        kernel_size=(7,) * ndim,
        strides=(1,) * ndim,
        padding=self.padding,
        kernel_init=self.initializer,
        use_local=self.use_local,
    )(x)
    x = _apply_norm(x, self.norm_type)
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
          kernel_size=_spatial_tuple(self.kernel_size_downsampling, ndim),
          strides=(2,) * ndim,
          padding=self.padding,
          kernel_init=self.initializer,
      )(x)
      x = _apply_norm(x, self.norm_type)
      x = self.act_fun(x)

    # Core (lowest) level.
    # Multiplier of the base number of features.
    mult = 2**self.n_downsample_layers
    for i in range(self.n_res_blocks):
      x = ResnetBlock(
          features=self.ngf * mult,
          kernel_size=_spatial_tuple(self.kernel_size_core, ndim),
          padding=self.padding,
          dropout_layer=functools.partial(
              nn.Dropout, rate=self.dropout_rate, deterministic=not is_training
          ),
          initializer=self.initializer,
          norm_type=self.norm_type,
          name=f"resnet_block_core_number_{i}",
      )(x)

      # Use a transformer core.
      # TODO: add a conformer model.
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
        # Upsampling using (bi)linear interpolation.
        x = functools.partial(
            jax.image.resize,
            shape=(
                x.shape[0],
                *[2 * size for size in x.shape[1:-1]],
                x.shape[-1],
            ),
            method="linear",
        )(x)
        x = layers.ConvLayer(
            features=(self.ngf * mult) // 2,
            kernel_size=_spatial_tuple(self.kernel_size_upsampling, ndim),
            strides=(1,) * ndim,
            padding=self.padding,
            kernel_init=self.initializer,
        )(x)

      elif self.upsample_mode == "deconv":
        # TODO: use channel unrolling for the upsampling.
        x = nn.ConvTranspose(
            features=(self.ngf * mult) // 2,
            kernel_size=_spatial_tuple(self.kernel_size_upsampling, ndim),
            strides=(2,) * ndim,
            padding=self.padding_transpose,
            kernel_init=self.initializer,
        )(x)

      x = _apply_norm(x, self.norm_type)
      x = self.act_fun(x)

      if self.use_skips and i < len(horizontal_skips):
        # If using skips we concatenate the branches before going into the
        # the convolutional layer which will merge the information.

        y = horizontal_skips[i]
        # Further processing each skip branch.
        for j in range(self.n_res_blocks_level):
          y = ResnetBlock(
              features=y.shape[-1],
              kernel_size=_spatial_tuple(self.kernel_size_core, ndim),
              padding=self.padding,
              dropout_layer=functools.partial(
                  nn.Dropout,
                  rate=self.dropout_rate,
                  deterministic=not is_training,
              ),
              initializer=self.initializer,
              norm_type=self.norm_type,
              name=f"resnet_level_{i}_number_{j}",
          )(y)

        x = jnp.concatenate([x, y], axis=-1)
        x = layers.ConvLayer(
            features=(self.ngf * mult) // 2,
            kernel_size=_spatial_tuple(self.kernel_size_upsampling, ndim),
            strides=(1,) * ndim,
            padding=self.padding,
            kernel_init=self.initializer,
        )(x)
        x = _apply_norm(x, self.norm_type)
        x = self.act_fun(x)

    # Last convolution layer to correct the number of output channels.
    x = layers.ConvLayer(
        features=self.output_nc,
        kernel_size=(7,) * ndim,
        strides=(1,) * ndim,
        padding=self.padding,
        kernel_init=self.initializer,
        use_local=self.use_local,
    )(x)

    # Interpolating back to the original size.
    if self.interpolated_shape:
      x = FilteredInterpolation(
          target_shape=tuple(input_spatial),
          output_nc=self.output_nc,
          interpolation_method=self.interpolation_method,
          padding=self.padding,
          initializer=self.initializer,
          use_local=self.use_local,
      )(x)

    # Add skip connection between generator input and output.
    # Reference: https://github.com/leehomyc/cyclegan-1
    if applies_global_skip:
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
  or the output from the generator. Inputs have shape
  `(batch, *spatial, channels)` with one or two spatial dimensions; every
  layer halves the spatial size.

  Attributes:
      base_features: Number of filters in the first conv layer
      n_layers: The number of conv layers in the discriminator.
      kernel_size: Size of the convolutional kernel (reduced to the spatial rank
        of the input).
      padding: Type of padding. An integer input is translated to a uniform
        padding. A string argument means the type of padding.
      use_bias: Flag for using biases in the normalised convolutions (the first
        and the last convolution always have one).
      use_local: Flag for using locally connected layers (This has more dof).
      initializer: Initialization method.
      norm_type: Normalisation after every convolution except the first and the
        last: `"instance"` (default), `"none"` or `"group"`. With instance
          normalisation the discriminator is, at initialisation, invariant to a
          positive rescaling of its input up to the normalisation epsilon: the
          first layer (zero bias, PReLU) is positively homogeneous and every
          later layer is normalised per sample and channel, so `D(c u) = D(u)`
          up to `O(eps / var)` where `var` is the per-channel activation
          variance. Amplitude information can only enter through the bias and
          PReLU of the first layer. Per-channel biases of the normalised
          convolutions (`use_bias=True`) are cancelled algebraically by the mean
          subtraction and, in float32 with the default fast variance estimate
          `E[x^2] - E[x]^2`, biases much larger than the activations degrade the
          normalisation numerically. Amplitude-aware discriminators (e.g. for
          debiasing physical fields) therefore need `norm_type="none"`, not
          `use_bias=True`.
  """

  base_features: int = 64
  n_layers: int = 3
  kernel_size: tuple[int, ...] = (5, 5)
  padding: int | str = 1
  use_bias: bool = False
  use_local: bool = False
  initializer: Initializer = jax.nn.initializers.normal(stddev=0.02)
  norm_type: str = "instance"

  @nn.compact
  def __call__(self, x: Array) -> Array:
    ndim = _check_spatial_ndim(x)
    kernel_size = _spatial_tuple(self.kernel_size, ndim)

    x = layers.ConvLayer(
        features=self.base_features,
        kernel_size=kernel_size,
        strides=(2,) * ndim,
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
          kernel_size=kernel_size,
          strides=(2,) * ndim,
          padding=self.padding,
          use_bias=self.use_bias,
          kernel_init=self.initializer,
      )(x)
      x = _apply_norm(x, self.norm_type)
      x = nn.PReLU(negative_slope_init=0.2)(x)

    feature_multiplier = min(2**self.n_layers, 8)
    x = layers.ConvLayer(
        features=self.base_features * feature_multiplier,
        kernel_size=kernel_size,
        strides=(2,) * ndim,
        padding=self.padding,
        use_bias=self.use_bias,
        kernel_init=self.initializer,
    )(x)
    x = _apply_norm(x, self.norm_type)
    x = nn.PReLU(negative_slope_init=0.2)(x)

    # The output should be just one channel.
    x = layers.ConvLayer(
        features=1,
        kernel_size=kernel_size,
        strides=(1,) * ndim,
        padding=self.padding,
        use_local=self.use_local,
    )(x)

    return x
