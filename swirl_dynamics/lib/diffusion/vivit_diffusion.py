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

"""Lightweight implementation of ViViT model for diffusion models.

We consider a ViViT model as the backbone of a diffusion model for video
generation of trajectories of dyamics systems.

For the conditioning we follow the latent diffusion transformer approch in [1].

References:
[1] William Peebles, Saining Xie, 'Scalable Diffusion Models with Transformers'
    ICCV 2023.
[2] Tero Karras, Miika Aittala, and Timo Aila and Samuli Laine, 'Elucidating
    the Design Space of Diffusion-Based Generative Models', NeurIPS 2022.
"""

import functools
from typing import Any, Optional, Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from swirl_dynamics.lib.diffusion import reshape_utils
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.lib.diffusion import vivit


Array = jax.Array
Shape = Sequence[int]
Initializer = Callable[[Array, Sequence[int], jnp.dtype], Array]

_KERNEL_INITIALIZERS = dict({
    'zero': nn.initializers.zeros,
    'xavier': nn.initializers.xavier_uniform(),
})

_AXIS_TO_NAME = dict({
    1: 'time',
    2: 'space',
})

_AXIS_TO_NAME_3D = dict({
    1: 'time',
    2: 'height',
    3: 'width',
})


class EncoderEmbeddingBlock(nn.Module):
  """Transformer encoder block.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    attention_axis: Axis over which we run attention.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    droplayer_p: Probability of dropping a layer.
    attention_kernel_initializer: Initializer to use for attention
      layers.
    deterministic: Deterministic or not (to apply dropout).
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    dtype: The dtype of the computation (default: float32).

  Returns:
    Output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: jnp.dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_kernel_initializer: Initializer = nn.initializers.xavier_uniform()
  attention_fn: Any = nn.dot_product_attention
  droplayer_p: float = 0.0

  def get_drop_pattern(self, x: Array, deterministic: bool) -> Array:
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return jnp.array([0.0])

  @nn.compact
  def __call__(self, inputs: Array, emb: Array, deterministic: bool) -> Array:
    """Applies Encoder1DBlock module."""

    if inputs.ndim != 3:
      raise ValueError('The input of the encoder block needs to be a',
                       '3-dimensional tensor following (batch_size, num_tokes, '
                       f'emb_dim). Instead got {inputs.shape}.')

    # Attention block.
    x = nn.LayerNorm(dtype=self.dtype)(inputs)

    # Add FiLM layer before the attention layer (see Fig. 3 in [1]).
    x = unets.AdaptiveScale()(x, emb)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        attention_fn=self.attention_fn,
        dtype=self.dtype)(
            x, x, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    # Add FiLM layer after the attention layer (see Fig. 3 in [1]).
    x = unets.AdaptiveScale()(x, emb)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    x = x * (1.0 - drop_pattern) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    # Add FiLM layer before the MLP block (see Fig. 3 in [1]).
    y = unets.AdaptiveScale()(y, emb)
    y = vivit.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)
    # Add FiLM layer after the MLP block (see Fig. 3 in [1]).
    y = unets.AdaptiveScale()(y, emb)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    return y * (1.0 - drop_pattern) + x


class FactorizedSelfAttentionEmbeddingBlock(nn.Module):
  """Encoder with factorized self attention block.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    temporal_dims: Number of temporal dimensions in the flattened input
    attention_kernel_initializer: Initializer to use for attention layers.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    droplayer_p: Probability of dropping a layer.
    attention_order: The order to do the attention. Choice of {time_space,
      space_time}.
    dtype: the dtype of the computation (default: float32).
  """
  mlp_dim: int
  num_heads: int
  temporal_dims: int
  attention_kernel_initializer: Initializer
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droplayer_p: Optional[float] = None
  attention_order: str = 'time_space'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, inputs: Array, emb: Array, *, deterministic: bool
  ) -> Array:
    """Applies Encoder1DBlock module."""

    batch_size, num_tokens, emb_dims = inputs.shape

    # shape: (batch, num_frames, hw, emb_dim)
    inputs = reshape_utils.reshape_to_time_space(inputs, self.temporal_dims)

    self_attention = functools.partial(
        nn.SelfAttention,
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype)

    # Order of the Axial Transformer.
    if self.attention_order == 'time_space':
      attention_axes = (1, 2)
    elif self.attention_order == 'space_time':
      attention_axes = (2, 1)
    else:
      raise ValueError(f'Invalid attention order {self.attention_order}.')

    def _run_attention_on_axis(
        inputs: Array, emb: Array, axis: int, two_d_shape: tuple[int, ...]
    ):
      """Reshapes the input and run attention on the given axis."""
      # shape: (batch, num_tokens, emb_dim)
      inputs = reshape_utils.reshape_2d_to_1d_factorized(inputs, axis=axis)
      x = nn.LayerNorm(
          dtype=self.dtype, name='LayerNorm_{}'.format(_AXIS_TO_NAME[axis])
      )(inputs)
      # Add first FiLM layer, some reshaping is necessary. (see Fig. 3 in [1]).
      in_shape = x.shape
      x = unets.AdaptiveScale()(x.reshape(two_d_shape[0], -1, two_d_shape[-1]),
                                emb).reshape(in_shape)

      x = self_attention(
          name='MultiHeadDotProductAttention_{}'.format(_AXIS_TO_NAME[axis])
      )(x, deterministic=deterministic)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)

      # Second FiLM layer (see Fig. 3 in [1]).
      in_shape = x.shape
      x = unets.AdaptiveScale()(x.reshape(two_d_shape[0], -1, two_d_shape[-1]),
                                emb).reshape(in_shape)

      x = x + inputs

      # shape: (batch, num_frames, hw, emb_dim)
      return reshape_utils.reshape_to_2d_factorized(
          x, axis=axis, two_d_shape=two_d_shape
      )

    x = inputs
    two_d_shape = inputs.shape

    # shape: (batch, num_frames, hw, emb_dim)
    for axis in attention_axes:
      x = _run_attention_on_axis(x, emb, axis, two_d_shape)

    # MLP block.
    x = jnp.reshape(x, [batch_size, num_tokens, emb_dims])
    y = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_mlp')(x)
    # Add FiLM layer before the attention layer (see Fig. 3 in [1]).
    y = unets.AdaptiveScale()(y, emb)
    y = vivit.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name='MlpBlock')(
            y, deterministic=deterministic)
    # Add FiLM layer after the attention layer (see Fig. 3 in [1]).
    y = unets.AdaptiveScale()(y, emb)

    return x + y


class Factorized3DSelfAttentionEmbeddingBlock(nn.Module):
  """Encoder with factorized self attention block.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    temporal_dims: Number of temporal dimensions in the flattened input
    attention_kernel_initializer: Initializer to use for attention layers.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    droplayer_p: Probability of dropping a layer.
    attention_order: The order to do the attention. In this case the two choices
      are 'time_height_width' and 'height_width_time'.
    dtype: the dtype of the computation (default: float32).
  """
  mlp_dim: int
  num_heads: int
  three_dim_shape: tuple[int, int, int, int, int]
  attention_kernel_initializer: Initializer
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droplayer_p: Optional[float] = None
  attention_order: str = 'time_height_width'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array, emb: Array, *, deterministic: bool):
    """Applies Encoder1DBlock module."""

    batch_size, num_tokens, emb_dim = inputs.shape
    _, enc_t, enc_h, enc_w, _ = self.three_dim_shape

    if num_tokens != (enc_t * enc_h * enc_w):
      raise ValueError('The product of the encoded dimensions for time, height',
                       f' and width ( {enc_t}, {enc_h}, {enc_w}) respectively,',
                       ' should match with the number of of tokens ',
                       f'({num_tokens}) in the input.')

    inputs = jnp.reshape(inputs, self.three_dim_shape)

    self_attention = functools.partial(
        nn.SelfAttention,
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype)

    # Order of the Axial Transformer.
    if self.attention_order == 'time_height_width':
      attention_axes = (1, 2, 3)
    elif self.attention_order == 'height_width_time':
      attention_axes = (2, 3, 1)
    else:
      raise ValueError(f'Invalid attention order {self.attention_order}.')

    def _run_attention_on_axis(
        inputs: Array,
        emb: Array,
        axis: int,
        three_dim_shape: tuple[int, int, int, int, int],
    ):
      """Reshapes the input and run attention on the given axis."""
      # shape: (batch, num_tokens, emb_dim)
      inputs = reshape_utils.reshape_3d_to_1d_factorized(inputs, axis=axis)
      x = nn.LayerNorm(
          dtype=self.dtype, name='LayerNorm_{}'.format(_AXIS_TO_NAME_3D[axis])
      )(inputs)
      # Add first FiLM layer, some reshaping is necessary. (see Fig. 3 in [1]).
      in_shape = x.shape
      x = unets.AdaptiveScale()(
          x.reshape(three_dim_shape[0], -1, three_dim_shape[-1]), emb
      ).reshape(in_shape)

      x = self_attention(
          name='MultiHeadDotProductAttention_{}'.format(_AXIS_TO_NAME_3D[axis])
      )(x, deterministic=deterministic)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)

      # Second FiLM layer (see Fig. 3 in [1]).
      in_shape = x.shape
      x = unets.AdaptiveScale()(
          x.reshape(three_dim_shape[0], -1, three_dim_shape[-1]), emb
      ).reshape(in_shape)

      x = x + inputs

      # shape: (batch, num_frames, hw, emb_dim)
      return reshape_utils.reshape_to_3d_factorized(
          x, axis=axis, three_d_shape=three_dim_shape
      )

    x = inputs
    three_dim_shape = inputs.shape

    # shape: (batch, num_frames, hw, emb_dim)
    for axis in attention_axes:
      x = _run_attention_on_axis(x, emb, axis, three_dim_shape)

    # MLP block.
    x = jnp.reshape(x, (batch_size, num_tokens, emb_dim))
    y = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_mlp')(x)
    # Add FiLM layer before the attention layer (see Fig. 3 in [1]).
    y = unets.AdaptiveScale()(y, emb)
    y = vivit.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name='MlpBlock')(
            y, deterministic=deterministic)
    # Add FiLM layer after the attention layer (see Fig. 3 in [1]).
    y = unets.AdaptiveScale()(y, emb)

    return x + y


class TransformerEmbeddingBlock(nn.Module):
  """Transformer Block with embeddings.

  Attributes:
    inputs: nd-array, Input data
    temporal_dims: Number of temporal dimensions in the input.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of attention heads.
    attention_config: Has parameters for the type of attention.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer linearly
      grows from 0 to the provided value. Our implementation of stochastic
      depth follows timm library, which does per-example layer dropping and
      uses independent dropping patterns for each skip-connection.
    positional_embedding: The type of positional embedding to use. Supported
      values are {learned_1d, sinusoidal_1d, sinusoidal_3d, none}.
    encoded_shape: Three-dimensional shapes of the equivalent tensor. This is
      required for the three-dimensional axial transformer.
    normalise_output: If True, perform layernorm on the output.
  """

  temporal_dims: Optional[int]
  mlp_dim: int
  num_layers: int
  num_heads: int
  attention_config: ml_collections.ConfigDict | None = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32
  positional_embedding: str = 'sinusoidal_3d'
  encoded_shape: Optional[tuple[int, ...]] | None = None
  normalise_output: bool = True

  @nn.compact
  def __call__(self, inputs: Array, emb: Array, *, train: bool) -> Array:
    """Applies Transformer model on the inputs."""
    assert inputs.ndim == 3  # (batch, len, emb)
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    # Choosing the type of embedding.
    # TODO: add more embeddings in here.
    if self.positional_embedding == 'sinusoidal_3d':
      batch, num_tokens, hidden_dim = inputs.shape
      height = width = int(np.sqrt(num_tokens // self.temporal_dims))
      if height * width * self.temporal_dims != num_tokens:
        raise ValueError('Input is assumed to be square for sinusoidal init.',
                         f'Instead the input shape is {inputs.shape}, which ',
                         f'leads to {height}x{width}x{self.temporal_dims}')

      inputs_reshape = inputs.reshape([batch, self.temporal_dims, height, width,
                                       hidden_dim])

      x = vivit.AddFixedSinCosPositionEmbedding()(inputs_reshape)
      x = x.reshape([batch, num_tokens, hidden_dim])
    elif self.positional_embedding == 'none':
      x = inputs
    else:
      raise ValueError(
          f'Unknown positional embedding {self.positional_embedding}')

    # Choosing the type of attention mechanism to use.
    if self.attention_config is None or self.attention_config.type in [  # pytype: disable=attribute-error
        'spacetime', 'factorized_encoder'
    ]:
      encoder_block = EncoderEmbeddingBlock
    elif self.attention_config.type == 'factorized_self_attention_block':  # pytype: disable=attribute-error
      encoder_block = functools.partial(
          FactorizedSelfAttentionEmbeddingBlock,
          attention_order=self.attention_config.attention_order,  # pytype: disable=attribute-error
          attention_kernel_initializer=_KERNEL_INITIALIZERS[
              self.attention_config.get('attention_kernel_init_method',
                                        'xavier')],  # pytype: disable=attribute-error
          temporal_dims=self.temporal_dims)
    elif self.attention_config.type == 'factorized_3d_self_attention_block':  # pytype: disable=attribute-error
      encoder_block = functools.partial(
          Factorized3DSelfAttentionEmbeddingBlock,
          attention_order=self.attention_config.attention_order,  # pytype: disable=attribute-error
          attention_kernel_initializer=_KERNEL_INITIALIZERS[
              self.attention_config.get('attention_kernel_init_method',  # pytype: disable=attribute-error
                                        'xavier')],
          three_dim_shape=self.encoded_shape)
    else:
      raise ValueError(f'Unknown attention type {self.attention_config.type}')  # pytype: disable=attribute-error

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # We apply the attention module num_layers time.
    for layer in range(self.num_layers):
      droplayer_p = (
          layer / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate

      # Applyign the encoder block with all the common options.
      x = encoder_block(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droplayer_p=droplayer_p,
          name=f'encoderblock_{layer}',
          dtype=dtype)(
              x, emb, deterministic=not train)

    if self.normalise_output:
      encoded = nn.LayerNorm(name='encoder_norm')(x)
    else:
      encoded = x

    return encoded


class ViViTDiffusion(nn.Module):
  """Vision Transformer model for Video.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    output_features: Number of output features.
    num_heads: Number of self-attention heads.
    num_layers: Number of layers.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    temporal_encoding_config: ConfigDict which defines the type of input
      encoding when tokenising the video.
    attention_config: ConfigDict which defines the type of spatio-temporal
      attention applied in the model.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer. Linearly
      increases from 0 to the provided value..
    positional_embedding: Type of positional encoding.
    cond_resize_method: Resize method for channel-wise conditioning.
    cond_embed_dim: Embedding dimension for channel-wise conditioning.
    dtype: JAX data type.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  output_features: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  temporal_encoding_config: ml_collections.ConfigDict
  attention_config: ml_collections.ConfigDict
  dropout_rate: float = 0.1
  noise_embed_dim: int = 256
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  positional_embedding: str = 'sinusoidal_3d'
  cond_resize_method: str = 'cubic'
  cond_embed_dim: int = 128
  cond_padding: str = 'SAME'
  cond_kernel_size: Sequence[int] = (3, 3)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool = False
  ) -> Array:
    batch_size_input, num_frames, height, width, _ = x.shape

    cond = {} if cond is None else cond
    x = unets.InterpConvMerge(
        embed_dim=self.cond_embed_dim,
        resize_method=self.cond_resize_method,
        kernel_size=self.cond_kernel_size,
        padding=self.cond_padding,
    )(x, cond)

    # Computing the embedding for modulation.
    emb = unets.FourierEmbedding(dims=self.noise_embed_dim)(sigma)

    # Shape: (batch_size, num_frames//patch_time, height//patch_height,
    #         width//patch_width, emd_dim).
    x, temporal_dims = vivit.TemporalEncoder(
        temporal_encoding_config=self.temporal_encoding_config,
        patches=self.patches.size,
        hidden_size=self.hidden_size,
    )(x, train=is_training)

    batch_size, enc_t, enc_h, enc_w, emb_dim = x.shape
    assert batch_size == batch_size_input, (
        'Batch size was modified during temporal encoder. They should be ',
        f'equal, instead we obtained: {batch_size_input} and {batch_size}',
    )
    num_tokens = enc_t * enc_h * enc_w

    x = jnp.reshape(x, (batch_size, num_tokens, emb_dim))

    x = TransformerEmbeddingBlock(
        temporal_dims=temporal_dims,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        attention_config=self.attention_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        positional_embedding=self.positional_embedding,
        dtype=self.dtype,
        encoded_shape=(batch_size, enc_t, enc_h, enc_w, emb_dim),
        name='Transformer')(
            x, emb, train=is_training)

    x = vivit.TemporalDecoder(
        patches=self.patches.size,
        features_out=self.output_features,
        encoded_shapes=(enc_t, enc_h, enc_w),
        name='spatio-temporal_decoder',
    )(x, train=is_training)

    assert x.shape == (
        batch_size,
        num_frames,
        height,
        width,
        self.output_features,
    ), (f'Shape of the output is {x.shape},but it should have been ',
        f'({batch_size, num_frames, height, width, self.output_features})')
    return x


class PreconditionedDenoiser(ViViTDiffusion):
  """Preconditioned denoising model using a ViViT backbone.

  Attributes:
    sigma_data: The variance of the data.

  For more details, see Appendix B.6 in [2].
  """

  sigma_data: float = 1.0

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool = False,
  ) -> Array:
    """Runs preconditioned denoising."""
    if sigma.ndim < 1:
      sigma = jnp.broadcast_to(sigma, (x.shape[0],))

    if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
          'sigma must be 1D and have the same leading (batch) dimension as x'
          f' ({x.shape[0]})!'
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
