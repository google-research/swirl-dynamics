# Copyright 2023 The swirl_dynamics Authors.
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

"""U-Net model for denoisers."""

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

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


class AdaptiveScale(nn.Module):
  """Adaptively scale the input based on embedding.

  Conditional information is projected to two vectors of length c where c is
  the number of channels of x, then x is scaled channel-wise by first vector
  and offset channel-wise by the second vector.

  This method is now standard practice for conditioning with diffusion models,
  see e.g. https://arxiv.org/abs/2105.05233, and for the
  more general FiLM technique see https://arxiv.org/abs/1709.07871.
  """

  @nn.compact
  def __call__(self, x: Array, emb: Array) -> Array:
    """Adaptive scaling applied to the channel dimension.

    Args:
      x: Tensor to be rescaled.
      emb: Embedding values that drives the rescaling.

    Returns:
      Rescaled tensor plus bias.
    """
    assert emb.ndim == 2, (
        "The dimension of the embedding needs to be two, instead it was : "
        + str(emb.ndim)
    )
    affine = nn.Dense(features=x.shape[-1] * 2, kernel_init=default_init(1.0))
    scale_params = affine(nn.swish(emb))
    # Unsqueeze in the middle to allow broadcasting.
    scale_params = scale_params.reshape(
        scale_params.shape[:1] + (x.ndim - 2) * (1,) + scale_params.shape[1:]
    )
    scale, bias = jnp.split(scale_params, 2, axis=-1)
    return x * (scale + 1.0) + bias


class AttentionBlock(nn.Module):
  """Attention block."""

  num_heads: int = 1
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, is_training: bool) -> Array:
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


class LatLonConv(nn.Module):
  """Convolutional network adapted to lat-lon grids.

  Applies circular padding in longitunidal direction and edge padding in
  latitudinal direction.

  Attributes:
    features: The number of output channels.
    kernel_size: Size of the kernel in the spatial dimension.
    use_bias: Whether to use bias for the conv operation.
  """

  features: int
  kernel_size: tuple[int, ...] | list[int] = (3, 3)
  use_bias: bool = True
  kernel_init: Initializer = default_init(1.0)
  strides: tuple[int, ...] | list[int] = (1, 1)
  use_local: bool = False

  @nn.compact
  def __call__(self, x: Array) -> Array:
    assert x.ndim == 4, f"Input must be 4-D tensor: {x.shape}."
    assert (
        self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1
    ), f"Current kernel size {self.kernel_size} must be odd."

    # Applying periodic padding in the longitunidal direction
    lat_pad, lon_pad = self.kernel_size[0] // 2, self.kernel_size[1] // 2
    x_per = jnp.pad(
        x, ((0, 0), (0, 0), (lat_pad, lat_pad), (0, 0)), mode="wrap"
    )
    # Applying edge padding at the poles
    x_per = jnp.pad(
        x_per, ((0, 0), (lon_pad, lon_pad), (0, 0), (0, 0)), mode="edge"
    )
    # shape: (batch_size, lon, lat, features)
    conv_fn = nn.ConvLocal if self.use_local else nn.Conv

    return conv_fn(
        self.features,
        kernel_size=self.kernel_size,
        use_bias=self.use_bias,
        strides=self.strides,
        kernel_init=self.kernel_init,
        padding="VALID",
    )(x_per)


class DownsampleConv(nn.Module):
  """Downsampling convolution."""

  features: int
  ratio: int
  use_bias: bool = True
  kernel_init: Initializer = default_init(1.0)

  @nn.compact
  def __call__(self, x: Array) -> Array:
    xdim = x.ndim - 2
    assert np.all(np.asarray(x.shape[1:-1]) % self.ratio == 0), (
        f"Input dimensions (spatial) {x.shape[1:-1]} must divide the"
        f" downsampling ratio {self.ratio}."
    )
    x = nn.Conv(
        self.features,
        kernel_size=xdim * (self.ratio,),
        use_bias=self.use_bias,
        strides=xdim * (self.ratio,),
        kernel_init=self.kernel_init,
        padding="VALID",
    )(x)
    return x


def conv_layer(
    padding: str | int, use_local: bool = False, **kwargs
) -> nn.Module:
  """Wrapper for conv layers with non-standard boundary conditions."""
  if isinstance(padding, str) and padding.lower() == "latlon":
    return LatLonConv(use_local=use_local, **kwargs)
  elif use_local:
    return nn.ConvLocal(padding=padding, **kwargs)
  else:
    return nn.Conv(padding=padding, **kwargs)


class ConvBlock(nn.Module):
  """A basic two-layer convolution block with adaptive scaling in between.

  main conv path:
  --> GroupNorm --> Swish --> Conv -->
      GroupNorm --> FiLM --> Swish --> Dropout --> Conv

  shortcut path:
  --> Linear

  Attributes:
    channels: The number of output channels.
    kernel_sizes: Kernel size for both conv layers.
    padding: The type of convolution padding to use.
    dropout: The rate of dropout applied in between the conv layers.
  """

  out_channels: int
  kernel_size: tuple[int, ...]
  padding: str = "CIRCULAR"
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x: Array, emb: Array, is_training: bool) -> Array:
    h = x
    h = nn.GroupNorm(min(h.shape[-1] // 4, 32))(h)
    h = nn.swish(h)
    h = conv_layer(
        features=self.out_channels,
        kernel_size=self.kernel_size,
        padding=self.padding,
        kernel_init=default_init(1.0),
        name="conv_0",
    )(h)
    h = nn.GroupNorm(min(h.shape[-1] // 4, 32))(h)
    h = AdaptiveScale()(h, emb)
    h = nn.swish(h)
    h = nn.Dropout(rate=self.dropout, deterministic=not is_training)(h)
    h = conv_layer(
        features=self.out_channels,
        kernel_size=self.kernel_size,
        padding=self.padding,
        kernel_init=default_init(1.0),
        name="conv_1",
    )(h)
    return CombineResidualSkip(project_skip=True)(residual=h, skip=x)


def depth_to_space(x: Array, block_shape: tuple[int, ...]) -> Array:
  """Rearranges data from the channels to spatial dims as a way to upsample.

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


class FourierEmbedding(nn.Module):
  """Fourier embedding."""

  dims: int = 64
  max_freq: float = 2e4
  projection: bool = True

  @nn.compact
  def __call__(self, x: Array) -> Array:
    assert x.ndim == 1
    logfreqs = jnp.linspace(0, jnp.log(self.max_freq), self.dims // 2)
    x = jnp.pi * jnp.exp(logfreqs)[None, :] * x[:, None]
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)

    if self.projection:
      x = nn.Dense(features=2 * self.dims)(x)
      x = nn.swish(x)
      x = nn.Dense(features=self.dims)(x)

    return x


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
    assert x.ndim == 4
    _, h, w, c = x.shape
    assert c % 2 == 0, "Number of channels must be even."

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


def process_channelwise_cond(
    x: Array,
    cond: dict[str, Array],
    embed_dim: int,
    resize_method: str,
    padding: str,
):
  """Merges conditional inputs along the channel dimension.

  Relevant fields in the conditional input dictionary are first resized and then
  concatenated with the main input along their last axes.

  Args:
    x: The main model input.
    cond: A dictionary of conditional inputs. Those with keys that start with
      "channel:" are processed here while all others are omitted.
    embed_dim: The embedding dimension of the conditional input before the
      channelwise concatenation takes place.
    resize_method: The resizing method to use. See `jax.image.resize`.
    padding: The padding configuration for convolutions.

  Returns:
    Model input merged with channel conditions.
  """
  kernel_dim = x.ndim - 2
  for key, value in cond.items():
    if key.startswith("channel:"):
      if value.ndim != x.ndim:
        raise ValueError(
            f"Channel condition `{key}` does not have the same ndim"
            f" ({value.ndim}) as x ({x.ndim})!"
        )
    value = jax.image.resize(
        value, x.shape[:-1] + (value.shape[-1],), method=resize_method
    )
    value = conv_layer(
        features=embed_dim,
        kernel_size=kernel_dim * (3,),
        padding=padding,
        kernel_init=default_init(),
        name=f"conv_cond_{key}",
    )(value)
    value = nn.swish(nn.GroupNorm(min(value.shape[-1] // 4, 32))(value))
    x = jnp.concatenate([x, value], axis=-1)
  return x


class DStack(nn.Module):
  """Downsampling stack.

  Repeated convolutional blocks with occasional strides for downsampling.
  Features at different resolutions are concatenated into output to use
  for skip connections by the UStack module.
  """

  num_channels: tuple[int, ...]
  num_res_blocks: tuple[int, ...]
  downsample_ratio: tuple[int, ...]
  padding: str = "CIRCULAR"
  dropout_rate: float = 0.0
  use_attention: bool = False
  num_heads: int = 8
  channels_per_head: int = -1
  use_position_encoding: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, emb: Array, *, is_training: bool) -> list[Array]:
    assert (
        len(self.num_channels)
        == len(self.num_res_blocks)
        == len(self.downsample_ratio)
    )
    kernel_dim = x.ndim - 2
    res = np.asarray(x.shape[1:-1])
    skips = []
    h = conv_layer(
        features=128,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=default_init(1.0),
        name="conv_in",
    )(x)
    skips.append(h)

    for level, channel in enumerate(self.num_channels):
      h = DownsampleConv(
          features=channel,
          ratio=self.downsample_ratio[level],
          kernel_init=default_init(1.0),
          name=f"res{'x'.join(res.astype(str))}.downsample_conv",
      )(h)
      res = res // self.downsample_ratio[level]
      for block_id in range(self.num_res_blocks[level]):
        h = ConvBlock(
            out_channels=channel,
            kernel_size=kernel_dim * (3,),
            padding=self.padding,
            dropout=self.dropout_rate,
            name=f"res{'x'.join(res.astype(str))}.down.block{block_id}",
        )(h, emb, is_training=is_training)
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
  dropout_rate: float = 0.0
  use_attention: bool = False
  num_heads: int = 8
  channels_per_head: int = -1
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, x: Array, emb: Array, skips: list[Array], *, is_training: bool
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
            dropout=self.dropout_rate,
            name=f"res{'x'.join(res.astype(str))}.up.block{block_id}",
        )(h, emb, is_training=is_training)
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
      h = conv_layer(
          features=up_ratio**kernel_dim * channel,
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
    h = conv_layer(
        features=128,
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
  noise_embed_dim: int = 128
  padding: str = "CIRCULAR"
  dropout_rate: float = 0.0
  use_attention: bool = True  # lowest resolution only
  use_position_encoding: bool = True
  num_heads: int = 8
  cond_resize_method: str = "bilinear"
  cond_embed_dim: int = 128

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Predicts denoised given noised input and noise level.

    Args:
      x: The model input (i.e. noised sample) with shape `(batch,
        **spatial_dims, channels)`.
      sigma: The noise level, which either shares the same batch dimension as
        `x` or is a scalar (will be broadcasted accordingly).
      cond: The conditional inputs as a dictionary. Currently, only channelwise
        conditioning is supported.
      is_training: A boolean flag that indicates whether the module runs in
        training mode.

    Returns:
      An output array with the same dimension as `x`.
    """
    if sigma.ndim < 1:
      sigma = jnp.broadcast_to(sigma, (x.shape[0],))

    if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
          "sigma must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
      )

    cond = {} if cond is None else cond
    x = process_channelwise_cond(
        x, cond, self.cond_embed_dim, self.cond_resize_method, self.padding
    )

    kernel_dim = x.ndim - 2
    emb = FourierEmbedding(dims=self.noise_embed_dim)(sigma)
    skips = DStack(
        num_channels=self.num_channels,
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        downsample_ratio=self.downsample_ratio,
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
        use_position_encoding=self.use_position_encoding,
    )(x, emb, is_training=is_training)
    h = UStack(
        num_channels=self.num_channels[::-1],
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        upsample_ratio=self.downsample_ratio[::-1],
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
    )(skips[-1], emb, skips, is_training=is_training)
    h = nn.swish(nn.GroupNorm(min(h.shape[-1] // 4, 32))(h))
    h = conv_layer(
        features=self.out_channels,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=default_init(),
        name="conv_out",
    )(h)
    return h


class PreconditionedDenoiser(UNet):
  """Preconditioned denoising model.

  See Appendix B.6 in Karras et al. (https://arxiv.org/abs/2206.00364).
  """

  sigma_data: float = 1.0

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Runs preconditioned denoising."""
    if sigma.ndim < 1:
      sigma = jnp.broadcast_to(sigma, (x.shape[0],))

    if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
          "sigma must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
      )

    total_var = jnp.square(self.sigma_data) + jnp.square(sigma)
    c_skip = jnp.square(self.sigma_data) / total_var
    c_out = sigma * self.sigma_data / jnp.sqrt(total_var)
    c_in = 1 / jnp.sqrt(total_var)
    c_noise = 0.25 * jnp.log(sigma)

    c_in = jnp.expand_dims(c_in, axis=np.arange(x.ndim - 1, dtype=np.int32) + 1)
    c_out = jnp.expand_dims(c_out, axis=np.arange(x.ndim - 1) + 1)
    c_skip = jnp.expand_dims(c_skip, axis=np.arange(x.ndim - 1) + 1)

    f_x = super().__call__(
        jnp.multiply(c_in, x), c_noise, cond, is_training=is_training
    )
    return jnp.multiply(c_skip, x) + jnp.multiply(c_out, f_x)
