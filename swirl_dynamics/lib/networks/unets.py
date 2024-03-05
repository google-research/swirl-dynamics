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

"""U-Net models."""

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import swirl_dynamics.lib.layers.convolutions as conv_lib

Array = jax.Array
Initializer = nn.initializers.Initializer


def default_init(scale: float = 1e-10) -> Initializer:
  return nn.initializers.variance_scaling(
      scale=scale, mode="fan_avg", distribution="uniform"
  )


class CombineResidualSkip(nn.Module):
  project_skip: bool = False

  @nn.compact
  def __call__(self, *, residual: Array, skip: Array) -> Array:
    if self.project_skip:
      skip = nn.Dense(residual.shape[-1], kernel_init=default_init(1.0))(skip)
    return (skip + residual) / np.sqrt(2.0)


class AttentionBlock(nn.Module):
  """Attention block."""

  num_heads: int = 1
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, *, is_training: bool) -> Array:
    h = nn.GroupNorm(min(x.shape[-1] // 4, 32), name="norm")(x)
    h = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=not is_training,
        dtype=self.dtype,
        name="dot_attn",
    )(h, h)
    return CombineResidualSkip()(residual=h, skip=x)


class ResConv1x(nn.Module):
  """Single-layer residual network with size-1 conv kernels."""

  hidden_layer_size: int
  out_channels: int

  @nn.compact
  def __call__(self, x: Array) -> Array:
    skip = x
    kernel_size = (x.ndim - 2) * (1,)
    x = nn.Conv(
        features=self.hidden_layer_size,
        kernel_size=kernel_size,
        kernel_init=default_init(1.0),
    )(x)
    x = nn.swish(x)
    x = nn.Conv(
        features=self.out_channels,
        kernel_size=kernel_size,
        kernel_init=default_init(1.0),
    )(x)
    return CombineResidualSkip()(residual=x, skip=skip)


class ConvBlock(nn.Module):
  """A basic two-layer convolution block.

  Path: 2 x (GroupNorm --> Swish --> Conv) --> Residual

  Attributes:
    channels: The number of output channels.
    kernel_sizes: Kernel size for both conv layers.
    padding: The type of convolution padding to use.
  """

  out_channels: int
  kernel_size: tuple[int, ...]
  padding: str = "CIRCULAR"

  @nn.compact
  def __call__(self, x: Array) -> Array:
    h = x
    h = nn.GroupNorm(min(h.shape[-1] // 4, 32))(h)
    h = nn.swish(h)
    h = conv_lib.ConvLayer(
        features=self.out_channels,
        kernel_size=self.kernel_size,
        padding=self.padding,
        kernel_init=default_init(1.0),
        name="conv_0",
    )(h)
    h = nn.GroupNorm(min(h.shape[-1] // 4, 32))(h)
    h = nn.swish(h)
    h = conv_lib.ConvLayer(
        features=self.out_channels,
        kernel_size=self.kernel_size,
        padding=self.padding,
        kernel_init=default_init(1.0),
        name="conv_1",
    )(h)
    return CombineResidualSkip(project_skip=True)(residual=h, skip=x)


def depth_to_space(x: Array, block_shape: tuple[int, ...]) -> Array:
  """Rearranges data from the channels to spatial as a way to upsample.

  Args:
    x: The array to reshape.
    block_shape: The shape of the block that will be formed from the channel
      dimension. The number of elements (i.e. prod(block_shape) must divide the
      number of channels).

  Returns:
    The reshaped array.
  """
  if not 3 <= x.ndim <= 5:
    raise ValueError(
        f"Ndim of x ({x.ndim}) expected between 3 and 5 following (b, *xyz, c)."
    )

  if len(block_shape) != x.ndim - 2:
    raise ValueError(
        f"Ndim of block shape ({len(block_shape)}) must be that of x ({x.ndim})"
        " minus 2."
    )

  if x.shape[-1] % np.prod(block_shape):
    raise ValueError(
        f"The number of channels ({x.shape[-1]}) must be divisible by the block"
        f" size ({np.prod(block_shape)})."
    )

  old_shape = x.shape
  cout = old_shape[-1] // np.prod(block_shape)
  x = jnp.reshape(x, old_shape[:-1] + tuple(block_shape) + (cout,))
  # interleave old and new spatial axes
  spatial_axes = np.arange(2 * len(block_shape), dtype=np.int32) + 1
  new_axes = spatial_axes.reshape(2, -1).ravel(order="F")
  x = jnp.transpose(x, (0,) + tuple(new_axes) + (len(new_axes) + 1,))
  # collapse interleaved axes
  new_shape = np.asarray(old_shape[1:-1]) * np.asarray(block_shape)
  new_shape = (old_shape[0],) + tuple(new_shape) + (cout,)
  return jnp.reshape(x, new_shape)


class Add1dPosEmbedding(nn.Module):
  """Adds a trainable 1D position embeddings to the inputs."""

  emb_init: Initializer = nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x: Array) -> Array:
    assert x.ndim == 3
    _, l, c = x.shape
    pos_embed = self.param("pos_emb", self.emb_init, (l, c))
    return x + jnp.expand_dims(pos_embed, axis=0)


class Add2dPosEmbedding(nn.Module):
  """Adds a trainable 2D position embeddings to the inputs."""

  emb_init: Initializer = nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, x: Array) -> Array:
    if x.ndim != 4:
      raise ValueError("Only 4D inputs are supported for the two ",
                       "dimensional positional embeddings. Instead the ",
                       "dimension of the input is {x.ndim}.")
    _, h, w, c = x.shape
    if c % 2 == 1:
      raise ValueError(f"Number of channels must be even, instead we had {c}")

    row_embed = self.param("pos_emb_row", self.emb_init, (w, c // 2))
    col_embed = self.param("pos_emb_col", self.emb_init, (h, c // 2))

    row_embed = jnp.tile(jnp.expand_dims(row_embed, axis=0), (h, 1, 1))
    col_embed = jnp.tile(jnp.expand_dims(col_embed, axis=1), (1, w, 1))

    pos_embed = jnp.concatenate([row_embed, col_embed], axis=-1)
    return x + jnp.expand_dims(pos_embed, axis=0)


def position_embedding(ndim: int, **kwargs) -> nn.Module:
  if ndim == 1:
    return Add1dPosEmbedding(**kwargs)
  elif ndim == 2:
    return Add2dPosEmbedding(**kwargs)
  else:
    raise ValueError("Only 1D or 2D position embeddings are supported.")


class DStack(nn.Module):
  """Downsampling stack.

  Repeated convolutional blocks with occaisonal strides for downsampling.
  Features at different resolutions are concatenated into output to use
  for skip connections by the UStack module.
  """

  num_channels: tuple[int, ...]
  num_res_blocks: tuple[int, ...]
  downsample_ratio: tuple[int, ...]
  padding: str = "CIRCULAR"
  use_attention: bool = False
  num_heads: int = 8
  use_position_encoding: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, *, is_training: bool) -> list[Array]:
    assert (
        len(self.num_channels)
        == len(self.num_res_blocks)
        == len(self.downsample_ratio)
    )
    kernel_dim = x.ndim - 2
    res = np.asarray(x.shape[1:-1])
    skips = []
    h = conv_lib.ConvLayer(
        features=128,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=default_init(1.0),
        name="conv_in",
    )(x)
    skips.append(h)

    for level, channel in enumerate(self.num_channels):
      h = conv_lib.DownsampleConv(
          features=channel,
          ratios=kernel_dim * (self.downsample_ratio[level],),
          kernel_init=default_init(1.0),
          name=f"res{'x'.join(res.astype(str))}.downsample_conv",
      )(h)
      res = res // self.downsample_ratio[level]
      for block_id in range(self.num_res_blocks[level]):
        h = ConvBlock(
            out_channels=channel,
            kernel_size=kernel_dim * (3,),
            padding=self.padding,
            name=f"res{'x'.join(res.astype(str))}.down.block{block_id}",
        )(h)
        if self.use_attention and level == len(self.num_channels) - 1:
          b, *hw, c = h.shape
          # Adding positional encoding, only in Dstack.
          if self.use_position_encoding:
            h = position_embedding(
                ndim=kernel_dim,
                name=f"res{'x'.join(res.astype(str))}.down.block.posenc{block_id}",
            )(h)
          h = AttentionBlock(
              num_heads=self.num_heads,
              dtype=self.dtype,
              name=f"res{'x'.join(res.astype(str))}.down.block{block_id}.attn",
          )(h.reshape(b, -1, c), is_training=is_training)
          h = ResConv1x(
              hidden_layer_size=channel * 2,
              out_channels=channel,
              name=f"res{'x'.join(res.astype(str))}.down.block{block_id}.res_conv_1x",
          )(h).reshape(b, *hw, c)
        skips.append(h)

    return skips


class UStack(nn.Module):
  """Upsampling Stack.

  Takes in features at intermediate resolutions from the downsampling stack
  as well as final output, and applies upsampling with convolutional blocks
  and combines together with skip connections in typical UNet style.
  Optionally can use self attention at low spatial resolutions.
  """

  num_channels: tuple[int, ...]
  num_res_blocks: tuple[int, ...]
  upsample_ratio: tuple[int, ...]
  padding: str = "CIRCULAR"
  use_attention: bool = False
  num_heads: int = 8
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, x: Array, skips: list[Array], *, is_training: bool
  ) -> Array:
    assert (
        len(self.num_channels)
        == len(self.num_res_blocks)
        == len(self.upsample_ratio)
    )
    kernel_dim = x.ndim - 2
    res = np.asarray(x.shape[1:-1])
    h = x
    for level, channel in enumerate(self.num_channels):
      for block_id in range(self.num_res_blocks[level]):
        h = CombineResidualSkip(
            project_skip=h.shape[-1] != skips[-1].shape[-1]
        )(residual=h, skip=skips.pop())
        h = ConvBlock(
            out_channels=channel,
            kernel_size=kernel_dim * (3,),
            padding=self.padding,
            name=f"res{'x'.join(res.astype(str))}.up.block{block_id}",
        )(h)
        if self.use_attention and level == 0:  # opposite to DStack
          b, *hw, c = h.shape
          h = AttentionBlock(
              num_heads=self.num_heads,
              dtype=self.dtype,
              name=f"res{'x'.join(res.astype(str))}.up.block{block_id}.attn",
          )(h.reshape(b, -1, c), is_training=is_training)
          h = ResConv1x(
              hidden_layer_size=channel * 2,
              out_channels=channel,
              name=f"res{'x'.join(res.astype(str))}.up.block{block_id}.res_conv_1x",
          )(h).reshape(b, *hw, c)

      # upsampling
      up_ratio = self.upsample_ratio[level]
      h = conv_lib.ConvLayer(
          features=(up_ratio**kernel_dim) * channel,
          kernel_size=kernel_dim * (3,),
          padding=self.padding,
          kernel_init=default_init(1.0),
          name=f"res{'x'.join(res.astype(str))}.conv_upsample",
      )(h)
      h = depth_to_space(h, block_shape=kernel_dim * (up_ratio,))
      res = res * up_ratio

    h = CombineResidualSkip(project_skip=h.shape[-1] != skips[-1].shape[-1])(
        residual=h, skip=skips.pop()
    )
    h = conv_lib.ConvLayer(
        features=self.num_channels[-1],
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=default_init(1.0),
        name="conv_out",
    )(h)
    return h


class UNet(nn.Module):
  """UNet model compatible with 1 or 2 spatial dimensions."""

  out_channels: int
  num_channels: tuple[int, ...] = (128, 256, 256, 256)
  downsample_ratio: tuple[int, ...] = (2, 2, 2, 2)
  num_blocks: int = 4
  padding: str = "CIRCULAR"
  use_attention: bool = True  # lowest resolution only
  use_position_encoding: bool = True
  num_heads: int = 8

  @nn.compact
  # def __call__(self, x: Array, *, is_training: bool) -> Array:
  def __call__(self, x: Array) -> Array:  # bandaid to avoid dropout.
    kernel_dim = x.ndim - 2
    skips = DStack(
        num_channels=self.num_channels,
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        downsample_ratio=self.downsample_ratio,
        padding=self.padding,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
        use_position_encoding=self.use_position_encoding,
    )(x, is_training=False)  # this is a bandaid to avoid dropout.
    # )(x, is_training=is_training)
    h = UStack(
        num_channels=self.num_channels[::-1],
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        upsample_ratio=self.downsample_ratio[::-1],
        padding=self.padding,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
    )(skips[-1], skips, is_training=False)  # bandaid too.
    # )(skips[-1], skips, is_training=is_training)
    h = nn.swish(nn.GroupNorm(min(h.shape[-1] // 4, 32))(h))
    h = conv_lib.ConvLayer(
        features=self.out_channels,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=default_init(),
        name="conv_out",
    )(h)
    return h
