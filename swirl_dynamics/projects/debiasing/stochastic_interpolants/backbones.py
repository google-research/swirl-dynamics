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

"""Backbones for the stochastic interpolants."""

from collections.abc import Callable

from flax import linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets


Array = jax.Array
Initializer = nn.initializers.Initializer
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)


class UNet(nn.Module):
  """UNet model with conditional categorical embeddings.

  This is a small modification of the UNet model from the diffusion library
  (swirl_dynamics.lib.diffusion.unets), to which we add categorical conditional
  embeddings.
  """

  out_channels: int
  resize_to_shape: tuple[int, ...] | None = None  # spatial dims only
  num_channels: tuple[int, ...] = (128, 256, 256, 256)
  downsample_ratio: tuple[int, ...] = (2, 2, 2, 2)
  num_blocks: int = 4
  noise_embed_dim: int = 128
  padding: str = "CIRCULAR"
  dropout_rate: float = 0.0
  use_attention: bool = True  # lowest resolution only
  use_position_encoding: bool = True
  num_heads: int = 8
  normalize_qk: bool = False
  cond_resize_method: str = "bilinear"
  cond_embed_dim: int = 128
  cond_merging_fn: type[unets.MergeChannelCond] | None = None
  cond_embed_fn: type[unets.MergeEmdCond] | None = None
  cond_embed_kwargs: dict[str, jax.typing.ArrayLike] | None = None
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
          "sigma must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
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

    kernel_dim = x.ndim - 2
    cond = {} if cond is None else cond

    # In corporates the inputs by merging them with the main input along the
    # channel dimension.
    if self.cond_merging_fn is not None:
      x = self.cond_merging_fn(
          embed_dim=self.cond_embed_dim,
          resize_method=self.cond_resize_method,
          kernel_size=(3,) * kernel_dim,
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x, cond)

    # Incorporating the noise embedding.
    emb = unets.FourierEmbedding(dims=self.noise_embed_dim)(sigma)
    # Incorporating the embedding from the conditional inputs.
    if self.cond_embed_fn:
      emb = self.cond_embed_fn(**self.cond_embed_kwargs)(
          emb, cond, is_training=is_training
      )

    skips = unets.DStack(
        num_channels=self.num_channels,
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        downsample_ratio=self.downsample_ratio,
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
        use_position_encoding=self.use_position_encoding,
        normalize_qk=self.normalize_qk,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(x, emb, is_training=is_training)
    h = unets.UStack(
        num_channels=self.num_channels[::-1],
        num_res_blocks=len(self.num_channels) * (self.num_blocks,),
        upsample_ratio=self.downsample_ratio[::-1],
        padding=self.padding,
        dropout_rate=self.dropout_rate,
        use_attention=self.use_attention,
        num_heads=self.num_heads,
        normalize_qk=self.normalize_qk,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(skips[-1], emb, skips, is_training=is_training)

    h = nn.swish(nn.GroupNorm(min(h.shape[-1] // 4, 32))(h))
    h = layers.ConvLayer(
        features=self.out_channels,
        kernel_size=kernel_dim * (3,),
        padding=self.padding,
        kernel_init=unets.default_init(),
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="conv_out",
    )(h)

    if self.resize_to_shape:
      h = layers.FilteredResize(
          output_size=input_size,
          kernel_size=(7, 7),
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(h)
    return h


class MergeCategoricalEmbCond(unets.MergeEmdCond):
  """Computes an embedding for the categorical conditional inputs.

  This is designed for the case where the conditional inputs are categorical
  and we want to compute an embedding for them.

  Attributes:
    cond_key: The key of the conditional input to be used as label.
    num_classes: The number of classes in the categorical conditional input.
    features_embedding: The number of features in the embedding.
    act_fun: The activation function to use in the embedding.
    precision: The precision to use in the embedding.
    dtype: The dtype to use in the embedding.
    param_dtype: The param dtype to use in the embedding.
  """
  cond_key: str = "emb:label"
  num_classes: int = 10  # This is for the MNIST dataset.
  features_embedding: int = 128
  act_fun: Callable[[Array], Array] = nn.silu
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, emb: Array, cond: dict[str, Array], is_training: bool):
    # This is mostly to avoid the initialization of the embedding.
    cond_emb = jnp.astype(cond[self.cond_key], jnp.int32)
    cond_emb = nn.Embed(
        num_embeddings=self.num_classes, features=self.features_embedding
    )(cond_emb)

    # Process the conditional embedding. And projecto to the same dimension as
    # the embedding.
    cond_emb = nn.Dense(
        features=2 * self.features_embedding,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(cond_emb)
    cond_emb = self.act_fun(cond_emb)
    cond_emb = nn.Dense(
        features=emb.shape[-1],
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(cond_emb)

    return emb + cond_emb
