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

"""3D U-Net denoiser models.

Intended for inputs with dimensions (batch, time, x, y, channels). The U-Net
stacks successively apply 2D downsampling/upsampling in space only. At each
resolution, an axial attention block (involving space and/or time) is applied.
"""

from collections.abc import Sequence
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets

Array = jax.Array
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)


def _maybe_broadcast_to_list(
    source: bool | Sequence[bool], reference: Sequence[Any]
) -> list[bool]:
  """Broadcasts to a list with the same length if applicable."""
  if isinstance(source, bool):
    return [source] * len(reference)
  else:
    if len(source) != len(reference):
      raise ValueError(f"{source} must have the same length as {reference}!")
    return list(source)


class AxialSelfAttentionBlock(nn.Module):
  """Block consisting of (potentially multiple) axial attention layers."""

  attention_axes: int | Sequence[int] = -2
  add_position_embedding: bool | Sequence[bool] = True
  num_heads: int | Sequence[int] = 1
  precision: PrecisionLike = None
  normalize_qk: bool = False
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, is_training: bool) -> Array:
    # Consolidate configs for all attention layers.
    attn_axes = self.attention_axes
    attn_axes = (attn_axes,) if isinstance(attn_axes, int) else attn_axes
    num_axes = len(attn_axes)

    pos_embed = self.add_position_embedding
    if isinstance(pos_embed, bool):
      pos_embed = (pos_embed,) * num_axes

    num_heads = self.num_heads
    if isinstance(num_heads, int):
      num_heads = (num_heads,) * num_axes

    # Axial attention ops followed by a projection.
    h = x
    for axis, add_emb, num_head in zip(attn_axes, pos_embed, num_heads):
      if add_emb:
        h = layers.AddAxialPositionEmbedding(
            position_axis=axis, name=f"pos_emb_axis{axis}"
        )(h)
      h = nn.GroupNorm(
          min(h.shape[-1] // 4, 32), name=f"attn_prenorm_axis{axis}"
      )(h)
      h = layers.AxialSelfAttention(
          attention_axis=axis,
          num_heads=num_head,
          kernel_init=nn.initializers.xavier_uniform(),
          deterministic=not is_training,
          normalize_qk=self.normalize_qk,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          precision=self.precision,
          name=f"axial_attn_axis{axis}",
      )(h)
      h = nn.GroupNorm(
          min(h.shape[-1] // 4, 32), name=f"dense_prenorm_axis{axis}"
      )(h)
      h = nn.Dense(features=h.shape[-1], kernel_init=unets.default_init(1.0))(h)
    return layers.CombineResidualWithSkip()(residual=h, skip=x)


class DStack(nn.Module):
  """Downsampling stack.

  Repeated convolutional blocks with occasional strides for downsampling
  (spatial only, temporal dimension is left untouched). Features at different
  resolutions are concatenated into output to use for skip connections by the
  UStack module.
  """

  num_channels: Sequence[int]
  num_res_blocks: Sequence[int]
  downsample_ratio: Sequence[int]
  use_spatial_attention: Sequence[bool]
  use_temporal_attention: Sequence[bool]
  num_input_proj_channels: int = 128
  padding: str = "LATLON"
  dropout_rate: float = 0.0
  use_position_encoding: bool = False
  precision: PrecisionLike = None
  num_heads: int = 8
  normalize_qk: bool = False
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array, emb: Array, *, is_training: bool) -> list[Array]:
    # The following should already have been checked at the caller level. Run
    # assserts to verify that it is indeed the case.
    assert x.ndim == 5
    assert x.shape[0] == emb.shape[0]
    assert len(self.use_spatial_attention) == len(self.use_temporal_attention)
    assert len(self.num_channels) == len(self.num_res_blocks)
    assert len(self.downsample_ratio) == len(self.num_res_blocks)

    # Logic.
    spatial_dims = np.asarray(x.shape[2:-1])
    nt = x.shape[1]
    skips = []

    h = layers.ConvLayer(
        features=self.num_input_proj_channels,
        kernel_size=(3, 3),
        padding=self.padding,
        kernel_init=unets.default_init(1.0),
        name="conv2d_in",
    )(x)
    skips.append(h)

    for level, channel in enumerate(self.num_channels):
      h = layers.DownsampleConv(
          features=channel,
          ratios=(self.downsample_ratio[level], self.downsample_ratio[level]),
          kernel_init=unets.default_init(1.0),
          name=f"{nt}xres{'x'.join(spatial_dims.astype(str))}.downsample",
      )(h)
      spatial_dims = spatial_dims // self.downsample_ratio[level]
      dims_str = spatial_dims.astype(str)
      for block_id in range(self.num_res_blocks[level]):
        h = unets.ConvBlock(
            out_channels=channel,
            kernel_size=(3, 3),
            padding=self.padding,
            dropout=self.dropout_rate,
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"{nt}xres{'x'.join(dims_str)}.dblock{block_id}",
        )(h, emb, is_training=is_training)
        if (
            self.use_spatial_attention[level]
            or self.use_temporal_attention[level]
        ):
          attn_axes = []
          if self.use_spatial_attention[level]:
            attn_axes.extend([2, 3])
          if self.use_temporal_attention[level]:
            attn_axes.append(1)

          h = AxialSelfAttentionBlock(
              attention_axes=attn_axes,
              add_position_embedding=self.use_position_encoding,
              num_heads=self.num_heads,
              precision=self.precision,
              dtype=self.dtype,
              param_dtype=self.param_dtype,
              normalize_qk=self.normalize_qk,
              name=f"{nt}xres{'x'.join(dims_str)}.dblock{block_id}.attn",
          )(h, is_training=is_training)
        skips.append(h)

    return skips


class UStack(nn.Module):
  """Upsampling Stack.

  Takes in features at intermediate resolutions from the downsampling stack
  as well as its final output, and applies upsampling with convolutional blocks
  and combines together with skip connections in typical UNet style.
  """

  num_channels: Sequence[int]
  num_res_blocks: Sequence[int]
  upsample_ratio: Sequence[int]
  use_spatial_attention: Sequence[bool]
  use_temporal_attention: Sequence[bool]
  num_output_proj_channels: int = 128
  padding: str = "CIRCULAR"
  dropout_rate: float = 0.0
  use_position_encoding: bool = False
  num_heads: int = 8
  precision: PrecisionLike = None
  normalize_qk: bool = False
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, x: Array, emb: Array, skips: list[Array], *, is_training: bool
  ) -> Array:
    # The following should already have been checked at the caller level. Run
    # assserts to verify that it is indeed the case.
    assert x.ndim == 5
    assert x.shape[0] == emb.shape[0]
    assert len(self.use_spatial_attention) == len(self.use_temporal_attention)
    assert len(self.num_channels) == len(self.num_res_blocks)
    assert len(self.upsample_ratio) == len(self.num_res_blocks)

    # Logic.
    spatial_dims = np.asarray(x.shape[-3:-1])
    dims_str = spatial_dims.astype(str)
    nt = x.shape[1]
    h = x
    for level, channel in enumerate(self.num_channels):
      for block_id in range(self.num_res_blocks[level]):
        h = layers.CombineResidualWithSkip(
            project_skip=h.shape[-1] != skips[-1].shape[-1]
        )(residual=h, skip=skips.pop())
        h = unets.ConvBlock(
            out_channels=channel,
            kernel_size=(3, 3),
            padding=self.padding,
            dropout=self.dropout_rate,
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"{nt}xres{'x'.join(dims_str)}.ublock{block_id}",
        )(h, emb, is_training=is_training)
        if (
            self.use_spatial_attention[level]
            or self.use_temporal_attention[level]
        ):
          attn_axes = []
          if self.use_spatial_attention[level]:
            attn_axes.extend([2, 3])
          if self.use_temporal_attention[level]:
            attn_axes.append(1)

          h = AxialSelfAttentionBlock(
              attention_axes=attn_axes,
              add_position_embedding=self.use_position_encoding,
              num_heads=self.num_heads,
              precision=self.precision,
              dtype=self.dtype,
              param_dtype=self.param_dtype,
              normalize_qk=self.normalize_qk,
              name=f"{nt}xres{'x'.join(dims_str)}.ublock{block_id}.attn",
          )(h, is_training=is_training)

      # upsampling
      up_ratio = self.upsample_ratio[level]
      h = layers.ConvLayer(
          features=up_ratio**2 * channel,
          kernel_size=(3, 3),
          padding=self.padding,
          kernel_init=unets.default_init(1.0),
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"{nt}xres{'x'.join(dims_str)}.conv2d_preupsample",
      )(h)
      h = layers.channel_to_space(inputs=h, block_shape=(up_ratio, up_ratio))
      spatial_dims = spatial_dims * up_ratio
      dims_str = spatial_dims.astype(str)

    h = layers.CombineResidualWithSkip(
        project_skip=h.shape[-1] != skips[-1].shape[-1]
    )(residual=h, skip=skips.pop())
    h = layers.ConvLayer(
        features=self.num_output_proj_channels,
        kernel_size=(3, 3),
        padding=self.padding,
        kernel_init=unets.default_init(1.0),
        name="conv2d_out",
    )(h)
    return h


class UNet3d(nn.Module):
  """UNet model for 3D time-space input.

  This model processes 3D spatiotemporal data using a UNet architecture. It
  progressively downsamples the input for efficient feature extraction at
  multiple scales. Features are extracted using 2D spatial convolutions along
  with spatial and/or temporal axial attention blocks. Upsampling and
  combination of features across scales produce an output with the same shape as
  the input.

  Attributes:
    out_channels: Number of output channels (should match the input).
    resize_to_shape: Optional input resizing shape. Facilitates greater
      downsampling flexibility. Output is resized to the original input shape.
    num_channels: Number of feature channels in intermediate convolutions.
    downsample_ratio: Spatial downsampling ratio per resolution (must evenly
      divide spatial dimensions).
    num_blocks: Number of residual convolution blocks per resolution.
    noise_embed_dim: Embedding dimensions for noise levels.
    input_proj_channels: Number of input projection channels.
    output_proj_channels: Number of output projection channels.
    padding: 2D padding type for spatial convolutions.
    dropout_rate: Dropout rate between convolution layers.
    use_spatial_attention: Whether to enable axial attention in spatial
      directions at each resolution.
    use_temporal_attention: Whether to enable axial attention in the temporal
      direction at each resolution.
    use_position_encoding: Whether to add position encoding before axial
      attention.
    num_heads: Number of attention heads.
    normalize_qk: Whether to apply normalization to the Q and K matrices of
      attention layers.
    cond_resize_method: Resize method for channel-wise conditioning.
    cond_embed_dim: Embedding dimension for channel-wise conditioning.
  """

  out_channels: int
  resize_to_shape: Sequence[int] | None = None
  num_channels: Sequence[int] = (128, 256, 256, 256)
  downsample_ratio: Sequence[int] = (2, 2, 2, 2)
  num_blocks: int = 4
  noise_embed_dim: int = 128
  input_proj_channels: int = 128
  output_proj_channels: int = 128
  padding: str = "LATLON"
  dropout_rate: float = 0.0
  use_spatial_attention: bool | Sequence[bool] = (False, False, False, True)
  use_temporal_attention: bool | Sequence[bool] = (False, False, False, True)
  use_position_encoding: bool = True
  num_heads: int = 8
  normalize_qk: bool = False
  cond_resize_method: str = "cubic"
  cond_embed_dim: int = 128
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

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
          "`sigma` must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
      )

    if x.ndim != 5:
      raise ValueError(
          "Only accept 5D input (batch, time, x, y, features)! x.shape:"
          f" {x.shape}"
      )

    use_spatial_attn = _maybe_broadcast_to_list(
        source=self.use_spatial_attention, reference=self.num_channels
    )
    use_temporal_attn = _maybe_broadcast_to_list(
        source=self.use_temporal_attention, reference=self.num_channels
    )

    if len(self.num_channels) != len(self.downsample_ratio):
      raise ValueError(
          f"`num_channels` {self.num_channels} and `downsample_ratio`"
          f" {self.downsample_ratio} must have the same lengths!"
      )

    input_size = x.shape[1:-1]
    if self.resize_to_shape is not None:
      x = layers.FilteredResize(
          output_size=self.resize_to_shape,
          kernel_size=(7, 7),
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x)

    cond = {} if cond is None else cond
    x = unets.InterpConvMerge(
        embed_dim=self.cond_embed_dim,
        resize_method=self.cond_resize_method,
        kernel_size=(3, 3),
        padding=self.padding,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x, cond)

    emb = unets.FourierEmbedding(dims=self.noise_embed_dim)(sigma)
    skips = DStack(
        num_channels=self.num_channels,
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        downsample_ratio=self.downsample_ratio,
        num_input_proj_channels=self.input_proj_channels,
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_spatial_attention=use_spatial_attn,
        use_temporal_attention=use_temporal_attn,
        use_position_encoding=self.use_position_encoding,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        num_heads=self.num_heads,
        normalize_qk=self.normalize_qk,
    )(x, emb, is_training=is_training)
    h = UStack(
        num_channels=self.num_channels[::-1],
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        upsample_ratio=self.downsample_ratio[::-1],
        num_output_proj_channels=self.output_proj_channels,
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_spatial_attention=use_spatial_attn,
        use_temporal_attention=use_temporal_attn,
        num_heads=self.num_heads,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        normalize_qk=self.normalize_qk,
    )(skips[-1], emb, skips, is_training=is_training)
    h = nn.swish(nn.GroupNorm(min(h.shape[-1] // 4, 32))(h))
    h = layers.ConvLayer(
        features=self.out_channels,
        kernel_size=(3, 3),
        padding=self.padding,
        kernel_init=unets.default_init(),
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="conv2d_out",
    )(h)

    if self.resize_to_shape is not None:
      h = layers.FilteredResize(
          output_size=input_size,
          kernel_size=(7, 7),
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(h)
    return h


class PreconditionedDenoiser3d(UNet3d):
  """Preconditioned 3-dimensional UNet denoising model.

  Attributes:
    sigma_data: The standard deviation of the data population. Used to derive
      the appropriate preconditioning coefficients to help ensure that the
      network deal with inputs and outputs that have zero mean and unit
      variance.
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
