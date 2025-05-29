# Copyright 2025 The swirl_dynamics Authors.
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

"""Backbone architectures for conditional downscaling modeling."""

from collections.abc import Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets

Array = jax.Array
Initializer = nn.initializers.Initializer


class InterpConvConcat(nn.Module):
  """Interpolates and concatenates inputs with a dealiasing convolution.

  Attributes:
    out_spatial_shape: The desired output spatial dimensions. Input tensors are
      assumed to have shape `(*batch_dims, *spatial_dims, channels)`, the
      outputs will have shape `(*batch_dims, *out_spatial_shape, channels)`.
    kernel_size: The kernel size of the convolution layer used for dealiasing.
    resize_method: The resizing method used by `jax.image.resize`. The default
      is `cubic`.
    padding: The padding method used by `jax.image.resize`. The default is
      `SAME`.
  """

  out_spatial_shape: tuple[int, ...]
  kernel_size: Sequence[int]
  resize_method: str = "cubic"
  padding: str = "SAME"

  @nn.compact
  def __call__(self, cond: dict[str, Array]):
    """Concatenates filtered conditional inputs along the channel dimension."""

    concat = []
    for key, value in sorted(cond.items()):

      if value.shape[-3:-1] != self.out_spatial_shape:

        value = layers.FilteredResize(
            output_size=self.out_spatial_shape,
            kernel_size=self.kernel_size,
            method=self.resize_method,
            padding=self.padding,
            name=f"resize_{key}",
        )(value)

      concat.append(value)

    return jnp.concatenate(concat, axis=-1)


class ResizeUStack(nn.Module):
  """Upsampling Stack through resize and convolution operations.

  Takes in features at intermediate resolutions from the downsampling stack
  as well as final output, and applies upsampling with convolutional blocks
  and combines together with skip connections in typical UNet style. Upsampling
  is performed through a `FilteredResize` operation. Optionally can use self
  attention at low spatial resolutions.

  Attributes:
    num_channels: Number of channels at each resolution level.
    num_res_blocks: Number of resnest blocks at each resolution level.
    upsample_ratio: The upsampling ratio between levels.
    padding: Type of padding for the convolutional layers.
    dropout_rate: Rate for the dropout inside the resnet blocks.
    use_attention: Whether to use attention at the coarsest (deepest) level.
    num_heads: Number of attentions heads inside the attention block.
    channels_per_head: Number of channels per head.
    dtype: Data type.
    resize_method: Method passed to `jax.image.resize`.
  """

  num_channels: tuple[int, ...]
  num_res_blocks: tuple[int, ...]
  upsample_ratio: tuple[int, ...]
  padding: str = "SAME"
  dropout_rate: float = 0.0
  use_attention: bool = False
  num_heads: int = 8
  channels_per_head: int = -1
  dtype: jnp.dtype = jnp.float32
  resize_method: str = "bilinear"

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
        h = layers.CombineResidualWithSkip(
            project_skip=h.shape[-1] != skips[-1].shape[-1]
        )(residual=h, skip=skips.pop())
        h = unets.ConvBlock(
            out_channels=channel,
            kernel_size=kernel_dim * (3,),
            padding=self.padding,
            dropout=self.dropout_rate,
            name=f"res{'x'.join(res.astype(str))}.up.block{block_id}",
        )(h, emb, is_training=is_training)
        if self.use_attention and level == 0:  # opposite to DStack
          b, *hw, c = h.shape
          h = unets.AttentionBlock(
              num_heads=self.num_heads,
              dtype=self.dtype,
              name=f"res{'x'.join(res.astype(str))}.up.block{block_id}.attn",
          )(h.reshape(b, -1, c), is_training=is_training)
          h = unets.ResConv1x(
              hidden_layer_size=channel * 2,
              out_channels=channel,
              name=f"res{'x'.join(res.astype(str))}.up.block{block_id}.res_conv_1x",
          )(h).reshape(b, *hw, c)

      # upsampling
      up_ratio = self.upsample_ratio[level]
      res = res * up_ratio
      h = layers.FilteredResize(
          output_size=tuple(res),
          kernel_size=kernel_dim * (3,),
          padding=self.padding,
          method=self.resize_method,
      )(h)

    h = layers.CombineResidualWithSkip(
        project_skip=h.shape[-1] != skips[-1].shape[-1]
    )(residual=h, skip=skips.pop())
    h = layers.ConvLayer(
        features=128,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=unets.default_init(1.0),
        name="conv_out",
    )(h)
    return h


class GlobalSkipUNet(nn.Module):
  """UNet with global skip connection for 1 or 2 spatial dimensions.

  The model combines a deterministic global skip connection based on the
  conditioning inputs, with a residual UNet that leverages both the conditioning
  and the input noise. Upsampling is performed through resizing and convolution.

  Attributes:
    out_channels: Number of channels of the output.
    resize_to_shape: Tuple specifying the spatial latent shape used internally
      by the model. Ideally, the sizes should be powers of 2.
    num_channels: Number of channels at each resolution level.
    num_blocks: Number of resnest blocks at each resolution level.
    downsample_ratio: The downsampling ratio between levels.
    noise_embed_dim: Dimension of the Fourier embedding of the input noise.
    padding: Type of padding for the convolutional layers.
    dropout_rate: Rate for the dropout inside the resnet blocks.
    use_attention: Whether to use attention at the coarser (deepest) level.
    use_position_encoding: Whether to use position encoding in the attention
      block of the DStack, if `use_attention`.
    num_heads: Number of attentions heads inside the attention block, if
      `use_attention`.
    cond_resize_method: The resizing method used by FilteredResize.
    cond_embed_dim: The dimension used to embed the conditioning inputs.
    cond_merging_fn: The function used to merge the conditioning inputs with the
      input noise.
  """

  out_channels: int
  resize_to_shape: tuple[int, ...] | None = None  # spatial dims only
  num_channels: tuple[int, ...] = (128, 256, 256, 256)
  downsample_ratio: tuple[int, ...] = (2, 2, 2, 2)
  num_blocks: int = 4
  noise_embed_dim: int = 128
  padding: str = "SAME"
  dropout_rate: float = 0.0
  use_attention: bool = True  # lowest resolution only
  use_position_encoding: bool = True
  num_heads: int = 8
  cond_resize_method: str = "bilinear"
  cond_embed_dim: int = 128
  cond_merging_fn: type[unets.MergeChannelCond] = unets.InterpConvMerge

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array],
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

    input_size = x.shape[1:-1]

    if self.resize_to_shape is not None:
      if not all([
          res_s >= inp_s
          for res_s, inp_s in zip(self.resize_to_shape, input_size)
      ]):
        raise ValueError(
            f"resize_to_shape {self.resize_to_shape} must be at",
            f" least as large as input_size {input_size} to avoid",
            " grid artifacts.",
        )
      x = layers.FilteredResize(
          output_size=self.resize_to_shape,
          kernel_size=(7, 7),
          padding=self.padding,
      )(x)

    kernel_dim = x.ndim - 2

    # Interpolate conditioning inputs to latent spatial shape
    global_skip = InterpConvConcat(
        out_spatial_shape=x.shape[1:-1],
        kernel_size=(7,) * kernel_dim,
        resize_method=self.cond_resize_method,
        padding=self.padding,
    )(cond)

    x = self.cond_merging_fn(
        embed_dim=self.cond_embed_dim,
        resize_method=self.cond_resize_method,
        kernel_size=(7,) * kernel_dim,
        padding=self.padding,
    )(x, cond)

    emb = unets.FourierEmbedding(dims=self.noise_embed_dim)(sigma)
    skips = unets.DStack(
        num_channels=self.num_channels,
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        downsample_ratio=self.downsample_ratio,
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
        use_position_encoding=self.use_position_encoding,
    )(x, emb, is_training=is_training)
    h = ResizeUStack(
        num_channels=self.num_channels[::-1],
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        upsample_ratio=self.downsample_ratio[::-1],
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
        resize_method=self.cond_resize_method,
    )(skips[-1], emb, skips, is_training=is_training)

    h = nn.swish(nn.GroupNorm(min(h.shape[-1] // 4, 32))(h))
    h = layers.ConvLayer(
        features=self.out_channels,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=unets.default_init(),
        name="conv_out",
    )(h)

    # Combine deterministic skip conditioning with UNet output
    h = layers.CombineResidualWithSkip(
        project_skip=h.shape[-1] != global_skip.shape[-1]
    )(residual=h, skip=global_skip)

    if self.resize_to_shape:
      h = layers.FilteredResize(
          output_size=input_size, kernel_size=(7, 7), padding=self.padding
      )(h)
    return h


class GlobalSkipUNetDenoiser(GlobalSkipUNet):
  """Preconditioned denoising model with a GlobalSkipUnet backbone.

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


class ResConvNet(nn.Module):
  """Residual ConvNet for two-dimensional fields.

  Attributes:
    out_channels: Number of channels of the output.
    resize_to_shape: Tuple specifying the spatial latent shape used internally
      by the model. Ideally, the sizes should be powers of 2.
    num_channels: Number of channels at each resolution level.
    num_blocks: Number of resnest blocks at each resolution level.
    noise_embed_dim: Dimension of the Fourier embedding of the input noise.
    padding: Type of padding for the convolutional layers.
    dropout_rate: Rate for the dropout inside the resnet blocks.
    cond_resize_method: The resizing method used by FilteredResize.
    cond_embed_dim: The dimension used to embed the conditioning inputs.
    cond_merging_fn: The function used to merge the conditioning inputs with the
      input noise.
  """

  out_channels: int
  resize_to_shape: tuple[int, int] | None = None  # spatial dims only
  num_channels: int = 256
  num_blocks: int = 4
  noise_embed_dim: int = 128
  padding: str = "SAME"
  dropout_rate: float = 0.0
  cond_resize_method: str = "bilinear"
  cond_embed_dim: int = 128
  cond_merging_fn: type[unets.MergeChannelCond] = unets.InterpConvMerge
  latent_kernel_size: int = 3

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array],
      *,
      is_training: bool,
  ) -> Array:
    """Denoises a noised input given a noise level and conditioning inputs.

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

    input_size = x.shape[1:-1]

    # Resize input noise to latent spatial dimension
    if self.resize_to_shape is not None:
      if not all([
          res_s >= inp_s
          for res_s, inp_s in zip(self.resize_to_shape, input_size)
      ]):
        raise ValueError(
            f"resize_to_shape {self.resize_to_shape} must be at",
            f" least as large as input_size {input_size} to avoid",
            " grid artifacts.",
        )
      x = layers.FilteredResize(
          output_size=self.resize_to_shape,
          kernel_size=(7, 7),
          padding=self.padding,
      )(x)
    kernel_dim = x.ndim - 2
    x = layers.ConvLayer(
        features=self.noise_embed_dim,
        kernel_size=(3,) * kernel_dim,
        padding=self.padding,
        name="conv2d_embed_noisy_input",
    )(x)

    # Interpolate conditioning inputs to latent spatial shape, retain for skip
    global_skip = InterpConvConcat(
        out_spatial_shape=x.shape[1:-1],
        kernel_size=(7,) * kernel_dim,
        resize_method=self.cond_resize_method,
        padding=self.padding,
    )(cond)

    # Merge conditioning and noisy input embedding
    h = self.cond_merging_fn(
        embed_dim=self.cond_embed_dim,
        resize_method=self.cond_resize_method,
        kernel_size=(7,) * kernel_dim,
        padding=self.padding,
    )(x, cond)
    emb = unets.FourierEmbedding(dims=self.noise_embed_dim)(sigma)

    for block_id in range(self.num_blocks):

      h = unets.ConvBlock(
          out_channels=self.num_channels,
          kernel_size=kernel_dim * (self.latent_kernel_size,),
          padding=self.padding,
          dropout=self.dropout_rate,
          name=f"ConvBlock_{block_id}",
      )(h, emb, is_training=is_training)

    h = nn.swish(nn.GroupNorm(min(h.shape[-1] // 4, 32))(h))
    # Combine deterministic skip conditioning with ConvBlocks output
    h = layers.CombineResidualWithSkip(
        project_skip=h.shape[-1] != global_skip.shape[-1]
    )(residual=h, skip=global_skip)

    if self.resize_to_shape:
      h = layers.FilteredResize(
          output_size=input_size, kernel_size=(7, 7), padding=self.padding
      )(h)

    h = layers.ConvLayer(
        features=self.out_channels,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=unets.default_init(),
        name="conv_out",
    )(h)

    return h


class ResConvNetDenoiser(ResConvNet):
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
