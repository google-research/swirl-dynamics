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

"""Lightweight Implementation of Video Vision Transformer (ViViT).

[1] Joao Carreira, and Andrew Zisserman. 'Quo vadis, action recognition? a
    new model and the kinetics dataset".
"""

import functools
from typing import Any, Optional, Callable, Sequence

import flax.linen as nn
from flax.linen import linear
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from swirl_dynamics.lib.diffusion import reshape_utils


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

# Permutation to perform a depth to space in three dimension.
# Basically we seek to go from:
# (batch_size, time, height, width,
#  time_patch, height_patch, width_patch, emb_dim )
# to
# (batch_size, time, time_patch, height, height_patch,
#  width, width_patch, emb_dim)
_PERMUTATION = (0, 1, 4, 2, 5, 3, 6, 7)


def get_fixed_sincos_position_embedding(
    x_shape: Shape,
    temperature: float = 10_000,
    dtype: jnp.dtype = jnp.float32
) -> Array:
  """Provides a fixed positional encoding for 2D and 3D coordinates.

  The embedding follows the initialisation method used in multiple papers such
  as "Attention is All You Need", https://arxiv.org/abs/1706.03762 and
  "Better plain ViT baselines for ImageNet-1k", https://arxiv.org/abs/2205.01580

  Args:
    x_shape: Shape of the inputs for which a positional embedding is needed.
    temperature: Temperature parameter.
    dtype: Data type of the positional encoding.

  Returns:
    Matrix of position embeddings with shape [1, ...], where ... = x_shape[1:].
  """
  if len(x_shape) not in (4, 5):
    raise ValueError(f'Unsupported input shape: {x_shape}. It should describe',
                     ' either a four- or five-tensor.')

  num_parts = 4 if len(x_shape) == 4 else 6
  channels = x_shape[-1]
  assert channels % num_parts == 0, f'Channels must be multiple of {num_parts}'
  omega = jnp.arange(
      channels // num_parts, dtype=jnp.float32) / (channels / num_parts)
  omega = 1. / (temperature**omega)

  # Two-dimensional input.
  if len(x_shape) == 4:
    _, h, w, _ = x_shape
    y, x = jnp.mgrid[:h, :w]
    y = jnp.einsum('m,d->md', y.flatten(), omega)
    x = jnp.einsum('m,d->md', x.flatten(), omega)
    p = [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)]
    shape = (1, h, w, channels)
  # Three-dimensional input.
  elif len(x_shape) == 5:
    _, t, h, w, _ = x_shape
    z, y, x = jnp.mgrid[:t, :h, :w]
    z = jnp.einsum('m,d->md', z.flatten(), omega)
    y = jnp.einsum('m,d->md', y.flatten(), omega)
    x = jnp.einsum('m,d->md', x.flatten(), omega)
    p = [jnp.sin(z), jnp.cos(z),
         jnp.sin(x), jnp.cos(x),
         jnp.sin(y), jnp.cos(y)]
    shape = (1, t, h, w, channels)
  else:
    raise ValueError(f'Unsupported input shape: {x_shape}')

  assert (shape[0] == 1) and (shape[1:] == x_shape[1:])

  pe = jnp.concatenate(p, axis=1)
  return jnp.asarray(pe, dtype).reshape(*shape)


class AddFixedSinCosPositionEmbedding(nn.Module):
  """Wrapper for a fixed positional encoding for 2D and 3D coordinates.

  The embedding follows the initialisation method used in multiple papers such
  as "Attention is All You Need", https://arxiv.org/abs/1706.03762 and
  "Better plain ViT baselines for ImageNet-1k", https://arxiv.org/abs/2205.01580

  Attributes:
    temperature: Temperature parameter.
    dtype: Data type of the positional encoding.
  """
  temperature: float = 10_000
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Adds the fixed embedding to the inputs.

    Args:
      inputs: Either an [N, W, H, C] or [N, T, W, H, C] input array.

    Returns:
      inputs with position encodings added to them.
    """
    return inputs + get_fixed_sincos_position_embedding(
        inputs.shape, self.temperature, self.dtype)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    mlp_dim: Internal dimension, this is usual 4 times the input
        dimension.
    out_dim: The embedding dimension of the output, if None, the output
        dimension is the same ast he input one.
    dropout_rate: Rate for droping some of the values.
    kernel_init: Initializer for the kernel.
    bias_init: Initializer for the bias.
    activation_fn: Activation function.
    precision: Option for specifiying the precicion of the matmuls.
    dtype: Data type of the weights/inputs.
  """

  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  use_bias: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[Array], Array] = nn.gelu
  precision: Optional[jax.lax.Precision] = None
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array, *, deterministic: bool) -> Array:
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            inputs)
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            x)

    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output


def central_frame_initializer():
  """Initialization function for 3D convolutional kernels.

  The filter is initialised such that it only depends on the input at the
  central (w.r.t the time dimension) frame. This has been taken from
  https://github.com/google-research/scenic/blob/main/scenic/projects/vivit/model.py

  Returns:
    init: Initialization function for Flax.
  """

  def init(key: Array, shape: Sequence[int], dtype=jnp.float32):

    if len(shape) != 5:
      raise ValueError('This function should initializing 5-dimensional',
                       'kernels whose shapes follow (num_time_frames, height,'
                       'with, emd_dim_in, emb_dim_out), instead the shapes of'
                       f' the input are {shape}')

    init_kernel = linear.default_kernel_init(key, shape, dtype)
    central_time_index = shape[0] // 2
    init_kernel = init_kernel.at[:, :, :central_time_index, :, :].set(0.0)
    init_kernel = init_kernel.at[:, :, central_time_index + 1:, :, :].set(0.0)

    return init_kernel

  return init


def average_frame_initializer():
  """Initialization function for 3D convolutional kernels.

  The filter is initialised such that it applies the same weights on each
  frame of the input. This is similar to "filter inflation" in [1]
  However, "filter inflation" uses the filter weights from a pretrained 2D CNN,
  and replicates them over all time dimensions.

  Returns:
    init: Initialization function for Flax.
  """

  def init(key: Array, shape: Sequence[int], dtype=jnp.float32) -> Array:

    if len(shape) != 5:
      raise ValueError('This function should initializing 5-dimensional',
                       'kernels whose shapes follow (num_time_frames, height,'
                       'with, emd_dim_in, emb_dim_out), instead the shapes of'
                       f' the input are {shape}')

    if shape[0] <= 1:
      raise ValueError('Temporal dimension should be > 1')

    # Tiling the temporal dimension of a larger kernel ensures that the
    # normalisation is handled by default_kernel_init().
    init_kernel = linear.default_kernel_init(key, shape, dtype)
    init_kernel = jnp.tile(init_kernel[0:1, :, :, :, :],
                           [init_kernel.shape[0], 1, 1, 1, 1])

    return init_kernel

  return init


class Embedding3D(nn.Module):
  """Spatio-Temporal Embedding.

  Attributes:
    patches: The size of each patch used for the temporal embedding in time,
        height and width, respectively.
    embedding_dim: Number of features in the output.
    kernel_init_method: Shape in time, height, and width of the encoded inputs.
  """

  patches: tuple[int, int, int]
  embedding_dim: int
  kernel_init_method: str

  @nn.compact
  def __call__(self, x: Array, *, train: bool) -> Array:

    if len(self.patches) != 3:
      raise ValueError('patches.size must have 3 elements, instead the path ',
                       f'size provided was {self.patches}')

    ft, fh, fw = self.patches

    if self.kernel_init_method == 'central_frame_initializer':
      kernel_initializer = central_frame_initializer()
    elif self.kernel_init_method == 'average_frame_initializer':
      kernel_initializer = average_frame_initializer()
    else:
      kernel_initializer = linear.default_kernel_init

    x = nn.Conv(
        features=self.embedding_dim,
        kernel_size=(ft, fh, fw),
        strides=(ft, fh, fw),
        padding='VALID',
        name='_conv_3d_embedding',
        kernel_init=kernel_initializer,
    )(x)

    return x


class TemporalEncoder(nn.Module):
  """Encoder for the statio-temporal embedding.

  Attributes:
    temporal_encoding_config: Dictionary containing some of the options.
    patches: The size of each patch for the temporal embedding.
    features_out: Number of features in the output.
    encoded_shapes: Shape in time, height, and width of the encoded inputs.
  """

  temporal_encoding_config: ml_collections.ConfigDict
  patches: tuple[int, ...]
  hidden_size: int
  return_1d: bool = False

  @nn.compact
  def __call__(self, inputs: Array, *, train: bool) -> tuple[Array, int]:

    kernel_init_method = self.temporal_encoding_config.get('kernel_init_method',
                                                           None)

    x = Embedding3D(patches=self.patches,
                    embedding_dim=self.hidden_size,
                    kernel_init_method=kernel_init_method)(inputs, train=train)

    temporal_dims = x.shape[1]

    if self.return_1d:
      batch_size, t, h, w, c = x.shape
      x = jnp.reshape(x, [batch_size, t * h * w, c])

    assert x.size > 0, ('Found zero tokens after temporal encoding. '
                        'Perhaps one of the patch sizes is such that '
                        'floor(dim_size / patch_size) = 0?')

    return x, temporal_dims


class TemporalDecoder(nn.Module):
  """Temporal Decoder from latent space to original 3d space.

  Attributes:
    patches: The size of each patch used for the temporal embedding.
    features_out: Number of features in the output.
    encoded_shapes: Shape of the encoded input following [time, height, width].
  """
  patches: tuple[int, ...]
  features_out: int
  encoded_shapes: tuple[int, ...]

  @nn.compact
  def __call__(self, inputs: Array, *, train: bool) -> Array:
    """Applies Transformer model on the inputs."""
    # We suppose that the input is batch_size, num_tokens, emb_dim
    batch_size, _, emb_dim = inputs.shape

    t, h, w = self.patches
    enc_t, enc_h, enc_w = self.encoded_shapes
    x = jnp.reshape(inputs, (batch_size, *self.encoded_shapes, emb_dim))

    x = nn.Conv(
        features=self.features_out * t * h * w,
        kernel_size=(1, 1, 1),
        strides=(1, 1, 1),
        name='conv_transpose_temporal_decoder',
    )(x)

    # TODO: Use unets.depth_to_space here instead.
    x = jnp.reshape(
        x, (batch_size, *self.encoded_shapes, t, h, w, self.features_out)
    )
    x = jnp.transpose(x, _PERMUTATION)
    x = jnp.reshape(
        x, (batch_size, enc_t * t, enc_h * h, enc_w * w, self.features_out)
    )

    return x


class EncoderFactorizedSelfAttentionBlock(nn.Module):
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
  def __call__(self, inputs: Array, *, deterministic: bool) -> Array:
    """Applies Encoder1DBlock module."""

    batch_size, num_tokens, emb_dim = inputs.shape
    inputs = reshape_utils.reshape_to_time_space(inputs, self.temporal_dims)

    self_attention = functools.partial(
        nn.SelfAttention,
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype)

    if self.attention_order == 'time_space':
      attention_axes = (1, 2)
    elif self.attention_order == 'space_time':
      attention_axes = (2, 1)
    else:
      raise ValueError(f'Invalid attention order {self.attention_order}.')

    def _run_attention_on_axis(inputs, axis, two_d_shape):
      """Reshapes the input and run attention on the given axis."""
      inputs = reshape_utils.reshape_2d_to_1d_factorized(inputs, axis=axis)
      x = nn.LayerNorm(
          dtype=self.dtype, name='LayerNorm_{}'.format(_AXIS_TO_NAME[axis]))(
              inputs)
      x = self_attention(
          name='MultiHeadDotProductAttention_{}'.format(_AXIS_TO_NAME[axis]))(
              x, deterministic=deterministic)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
      x = x + inputs
      return reshape_utils.reshape_to_2d_factorized(
          x, axis=axis, two_d_shape=two_d_shape)

    x = inputs
    two_d_shape = inputs.shape
    for axis in attention_axes:
      x = _run_attention_on_axis(x, axis, two_d_shape)

    # MLP block.
    x = jnp.reshape(x, [batch_size, num_tokens, emb_dim])
    # Perhaps add the embedding there.
    y = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_mlp')(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name='MlpBlock')(
            y, deterministic=deterministic)
    return x + y


class Encoder3DFactorizedSelfAttentionBlock(nn.Module):
  """Encoder with 3D factorized self attention block.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    three_dim_shape: Equivalent three-dimensional shape of the input.
    attention_kernel_initializer: Initializer to use for attention layers.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    droplayer_p: Probability of dropping a layer.
    attention_order: The order in which the axial attention is performed. You
      can choose either `time_space` (time_height_width) or `space_time`
      (height_width_time).
    dtype: The data type of the computation.
  """
  mlp_dim: int
  num_heads: int
  three_dim_shape: tuple[int, int, int, int, int]
  attention_kernel_initializer: Initializer = nn.initializers.xavier_uniform()
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droplayer_p: Optional[float] = None
  attention_order: str = 'time_height_width'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array, *, deterministic: bool) -> Array:
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

    if self.attention_order == 'time_height_width':
      attention_axes = (1, 2, 3)
    elif self.attention_order == 'height_width_time':
      attention_axes = (2, 3, 1)
    else:
      raise ValueError(f'Invalid attention order {self.attention_order}.')

    def _run_attention_on_axis(
        inputs: Array, axis: int, three_d_shape: tuple[int, ...]
    ) -> Array:
      """Reshapes the input and run attention on the given axis.

      Args:
        inputs: Input tensor in 3d space (i.e. a 5-tensor).
        axis: Index of the axis in which perform the axial attention.
        three_d_shape: Original three dimensional shape.

      Returns:
        An tensor with the same spatial dimensions as the input.
      """
      inputs = reshape_utils.reshape_3d_to_1d_factorized(inputs, axis=axis)
      x = nn.LayerNorm(
          dtype=self.dtype, name='LayerNorm_{}'.format(_AXIS_TO_NAME_3D[axis])
      )(inputs)
      x = self_attention(
          name='MultiHeadDotProductAttention_{}'.format(_AXIS_TO_NAME_3D[axis])
      )(x, deterministic=deterministic)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
      x = x + inputs
      return reshape_utils.reshape_to_3d_factorized(
          x, axis=axis, three_d_shape=three_d_shape)

    x = inputs
    three_d_shape = inputs.shape
    for axis in attention_axes:
      x = _run_attention_on_axis(x, axis, three_d_shape)

    # MLP block.
    x = jnp.reshape(x, [batch_size, num_tokens, emb_dim])
    # Perhaps add the embedding there.
    y = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_mlp')(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name='MlpBlock')(
            y, deterministic=deterministic)
    return x + y


class EncoderBlock(nn.Module):
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
  def __call__(self, inputs: Array, deterministic: bool) -> Array:
    """Applies Encoder Block module for a 1D sequence.

    Args:
      inputs: Input tensor of dimension (batch_size, num_tokens, emb_dim).
      deterministic: Option to add stochasticity, in the form of dropout, to the
          evaluation of the model.

    Returns:
      Dense networks applied to the embedding dimension (last one).
    """

    # Attention block.
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        attention_fn=self.attention_fn,
        dtype=self.dtype)(
            x, x, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    x = x * (1.0 - drop_pattern) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    return y * (1.0 - drop_pattern) + x


class TransformerBlock(nn.Module):
  """Transformer Block.

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
    normalise_output: If True, perform layernorm on the output.
  """

  temporal_dims: Optional[int]
  mlp_dim: int
  num_layers: int
  num_heads: int
  # TODO: encapsulate the configurations in its own container.
  attention_config: ml_collections.ConfigDict | None = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32
  positional_embedding: str = 'sinusoidal_3d'
  normalise_output: bool = True
  encoded_shape: Optional[tuple[int, ...]] | None = None

  @nn.compact
  def __call__(self, inputs: Array, *, train: bool) -> Array:
    """Applies Transformer model on the inputs."""
    assert inputs.ndim == 3  # (batch, len, emb)
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    # Computing positional embeddings.
    # TODO: Introduce more types of positional encoding.
    if self.positional_embedding == 'sinusoidal_3d':
      batch, num_tokens, hidden_dim = inputs.shape
      # TODO: change this one to handle non-square domains.
      height = width = int(np.sqrt(num_tokens // self.temporal_dims))
      if height * width * self.temporal_dims != num_tokens:
        raise ValueError('Input is assumed to be square in the '
                         'spatial dimensions for sinusoidal init. Instead the '
                         f'dimensions are {height} and {width}.')

      inputs_reshape = inputs.reshape([batch, self.temporal_dims, height, width,
                                       hidden_dim])

      x = AddFixedSinCosPositionEmbedding()(inputs_reshape)
      x = x.reshape([batch, num_tokens, hidden_dim])
    elif self.positional_embedding == 'none':
      x = inputs
    else:
      raise ValueError(
          f'Unknown positional embedding {self.positional_embedding}')

    # Choose the type of attention.
    if self.attention_config is None or self.attention_config.type in [  # pytype: disable=attribute-error
        'spacetime', 'factorized_encoder'
    ]:
      encoder_block = EncoderBlock
    elif self.attention_config.type == 'factorized_self_attention_block':  # pytype: disable=attribute-error
      encoder_block = functools.partial(
          EncoderFactorizedSelfAttentionBlock,
          attention_order=self.attention_config.attention_order,  # pytype: disable=attribute-error
          attention_kernel_initializer=_KERNEL_INITIALIZERS[
              self.attention_config.get('attention_kernel_init_method',  # pytype: disable=attribute-error
                                        'xavier')],
          temporal_dims=self.temporal_dims)
    elif self.attention_config.type == 'factorized_3d_self_attention_block':  # pytype: disable=attribute-error
      encoder_block = functools.partial(
          Encoder3DFactorizedSelfAttentionBlock,
          attention_order=self.attention_config.attention_order,  # pytype: disable=attribute-error
          attention_kernel_initializer=_KERNEL_INITIALIZERS[
              self.attention_config.get('attention_kernel_init_method',  # pytype: disable=attribute-error
                                        'xavier')],
          three_dim_shape=self.encoded_shape)
    else:
      raise ValueError(f'Unknown attention type {self.attention_config.type}')  # pytype: disable=attribute-error

    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Apply the attention module num_layers times with random layer drop.
    for layer in range(self.num_layers):
      droplayer_p = (
          layer / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
      x = encoder_block(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droplayer_p=droplayer_p,
          name=f'encoderblock_{layer}',
          dtype=dtype)(
              x, deterministic=not train)

    if self.normalise_output:
      encoded = nn.LayerNorm(name='encoder_norm')(x)
    else:
      encoded = x

    return encoded


class ViViT(nn.Module):
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
      increases from 0 to the provided value.
    dtype: JAX data type for the weights and activation functions.
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
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, x: Array, *, is_training: bool, debug: bool = False
  ) -> Array:

    # Performing the encoding. The output is a 5-tensor.
    x, temporal_dims = TemporalEncoder(
        temporal_encoding_config=self.temporal_encoding_config,
        patches=self.patches.size,
        hidden_size=self.hidden_size,
    )(x, train=is_training)

    batch_size, enc_t, enc_h, enc_w, emb_dim = x.shape
    num_tokens = enc_t * enc_h * enc_w

    x = jnp.reshape(x, (batch_size, num_tokens, emb_dim))

    x = TransformerBlock(
        temporal_dims=temporal_dims,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        attention_config=self.attention_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        dtype=self.dtype,
        name='Transformer',
        # TODO: clean this input/remove the temporal dims.
        encoded_shape=(batch_size, enc_t, enc_h, enc_w, emb_dim)
    )(x, train=is_training)

    x = TemporalDecoder(
        patches=self.patches.size,
        features_out=self.output_features,
        encoded_shapes=(enc_t, enc_h, enc_w),
        name='spatio-temporal_decoder',
    )(x, train=is_training)

    return x
