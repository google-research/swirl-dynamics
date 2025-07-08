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

"""Neural network backbones for the flow map matching.

This is a modified version of the UNet model in
swirl_dynamics.lib.diffusion.unets. As the input consists of two time-inputs,
we add an extra Fourier embeddings to the model. In addition, we add the
conditional inputs to the model following a labeled embedding.

We also implement a new Fourier embedding that is more flexible than the one
in swirl_dynamics.lib.diffusion.unets. We also change the activation function to
silu from swish, and we add a different scaling of the frequencies.
"""

from typing import Any, Callable, Literal, Mapping, Sequence, TypeAlias

from clu import metrics as clu_metrics
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets


MergeChannelCond = unets.MergeChannelCond
PrecisionLike: TypeAlias = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)
Array: TypeAlias = jax.Array
ArrayShape: TypeAlias = Sequence[int]
CondDict: TypeAlias = Mapping[str, Array]
Metrics: TypeAlias = clu_metrics.Collection
ShapeDict: TypeAlias = Mapping[str, Any]  # may be nested
PyTree: TypeAlias = Any


class FourierEmbedding(nn.Module):
  """Fourier embedding.

  This is one is a more flexible version of the FourierEmbedding in
  swirl_dynamics.lib.diffusion.unets. We also change the activation function to
  silu from swish, and we add a different scaling of the frequencies.

  Attributes:
    dims: The output channel dimension.
    max_freq: The maximum frequency (only used for exponential scaling)
    projection: Whether to add a projection layer.
    act_fun: The activation function.
    frequency_scaling: Type of scaling of the frequencies, either "linear" or
      "exponential".
    frequency_shift: The shift of the frequencies.
    precision: The precision of the computation.
    dtype: The dtype of the computation.
    param_dtype: The dtype of the parameters.
  """

  dims: int = 64
  max_freq: float = 2e4
  projection: bool = True
  act_fun: Callable[[Array], Array] = nn.silu
  frequency_scaling: Literal["linear", "exponential"] = "exponential"
  frequency_shift: float = 0.0
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    assert x.ndim == 1
    if self.frequency_scaling == "linear":
      # Using a linear scaling of the frequencies.
      freqs = jnp.linspace(0, self.dims // 2 - 1, self.dims // 2)
      x = jnp.pi * freqs[None, :] * x[:, None] - self.frequency_shift
    elif self.frequency_scaling == "exponential":
      # Using an exponential scaling of the frequencies.
      logfreqs = (
          jnp.linspace(0, jnp.log(self.max_freq), self.dims // 2)
          - self.frequency_shift
      )
      x = jnp.pi * jnp.exp(logfreqs)[None, :] * x[:, None]
    else:
      raise ValueError(
          f"Unsupported frequency scaling: {self.frequency_scaling}!"
      )

    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)

    if self.projection:
      x = nn.Dense(
          features=2 * self.dims,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x)
      x = self.act_fun(x)
      x = nn.Dense(
          features=self.dims,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x)

    return x


class FlowMapUNet(nn.Module):
  """UNet model with two time-inputs."""

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
  use_global_skip: bool = False
  cond_resize_method: str = "bilinear"
  cond_embed_dim: int = 128
  frequency_scaling: Literal["linear", "exponential"] = "exponential"
  frequency_shift: float = 0.0
  time_embedding_merge: Literal["concat", "add"] = "concat"
  cond_merging_fn: type[unets.MergeChannelCond] | None = None
  cond_embed_fn: type[unets.MergeEmdCond] | None = None
  cond_embed_kwargs: dict[str, jax.typing.ArrayLike] | None = None
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x_s: Array,
      t: Array,
      s: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Predicts x_t given x_s and t.

    Args:
      x_s: The model input (i.e. trajectory at time s) with shape `(batch,
        **spatial_dims, channels)`.
      t: The target time of the trajectory.
      s: The time index of the trajectory.
      cond: The conditional inputs as a dictionary. Currently, only channelwise
        conditioning is supported.
      is_training: A boolean flag that indicates whether the module runs in
        training mode.

    Returns:
      An output array with the same dimension as `x`.
    """
    if t.ndim < 1:
      t = jnp.broadcast_to(t, (x_s.shape[0],))

    if t.ndim != 1 or x_s.shape[0] != t.shape[0]:
      raise ValueError(
          "t must be 1D and have the same leading (batch) dimension as x"
          f" ({x_s.shape[0]})!"
      )

    if self.use_global_skip:
      skip = x_s
    else:
      skip = jnp.zeros_like(x_s)

    input_size = x_s.shape[1:-1]
    if self.resize_to_shape is not None:
      x_s = layers.FilteredResize(
          output_size=self.resize_to_shape,
          kernel_size=(7, 7),
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x_s)

    kernel_dim = x_s.ndim - 2
    cond = {} if cond is None else cond
    if self.cond_merging_fn is not None:
      x_s = self.cond_merging_fn(
          embed_dim=self.cond_embed_dim,
          resize_method=self.cond_resize_method,
          kernel_size=(3,) * kernel_dim,
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x_s, cond)

    # TODO: Refactor this outside of this function call.
    emb_t = FourierEmbedding(
        dims=self.noise_embed_dim,
        frequency_scaling=self.frequency_scaling,
        frequency_shift=self.frequency_shift,
        name="time_embedding_for_t",
    )(t)
    emb_s = FourierEmbedding(
        dims=self.noise_embed_dim,
        frequency_scaling=self.frequency_scaling,
        frequency_shift=self.frequency_shift,
        name="time_embedding_for_s",
    )(s)

    # Concatenate the noise and conditional embedding.
    # TODO: find a better way to do this.
    if self.time_embedding_merge == "concat":
      emb = jnp.concatenate([emb_t, emb_s], axis=-1)
    elif self.time_embedding_merge == "add":
      emb = emb_t + emb_s
    else:
      raise ValueError(
          f"Unsupported time embedding merge: {self.time_embedding_merge}!"
      )

    # Incorporating the embedding from the conditional inputs.
    if self.cond_embed_fn is not None:
      if self.cond_embed_kwargs is None:
        raise ValueError(
            "cond_embed_kwargs must be defined if cond_embed_fn is not None."
        )
      else:
        emb = self.cond_embed_fn(
            **self.cond_embed_kwargs,
        )(emb, cond, is_training=is_training)

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
    )(x_s, emb, is_training=is_training)
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
    x_t = layers.ConvLayer(
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
      x_t = layers.FilteredResize(
          output_size=input_size,
          kernel_size=(7, 7),
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x_t)

    # Add the skip connection if defined, otherwise add zero. The sign is to
    # substract it. (we have then x_s * (1 - (t-s)) + (t-s) * (s_theta))
    x_t -= skip
    return x_t
