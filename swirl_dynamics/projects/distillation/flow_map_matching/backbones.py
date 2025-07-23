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

References:
[1] Analyzing and Improving the Training Dynamics of Diffusion Models. Tero
Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, Samuli Laine.
CVPR 2024.
"""
import functools
from typing import Any, Callable, Literal, Mapping, Sequence, TypeAlias

from clu import metrics as clu_metrics
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets


MergeChannelCond: TypeAlias = unets.MergeChannelCond
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


def logit_normal_dist(
    rng: Array,
    shape: ArrayShape,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
):
  rnd_normal = jax.random.normal(rng, shape, dtype=dtype)
  return nn.sigmoid(rnd_normal * std + mean)


def time_sampler_mean_flow(
    rng: Array,
    batch_size: int,
    min_train_time: float = 1e-4,
    max_train_time: float = 1.0 - 1e-4,
    time_sampling: Callable[
        [Array, tuple[int, ...]], Array
    ] = functools.partial(jax.random.uniform, dtype=jnp.float32),
) -> tuple[Array, Array]:
  """Samples the time for the mean flow model.

  Args:
    rng: The random key.
    batch_size: The batch size.
    min_train_time: The minimum time at which the flow map and flow are sampled
      at training.
    max_train_time: The maximum time at which the flow map and flow are sampled
      at training.
    time_sampling: The function to use for the time sampling of t and s.

  Returns:
    The sampled times t and s.
  """
  time_sample_rng_t, time_sample_rng_s = jax.random.split(
      rng, num=2
  )

  time_range = max_train_time - min_train_time
  time_t = (
      time_range * time_sampling(time_sample_rng_t, (batch_size,))
      + min_train_time
  )
  time_s = (
      time_range * time_sampling(time_sample_rng_s, (batch_size,))
      + min_train_time
  )

  # Ensures that t>s.
  time_t, time_s = (
      jnp.where(time_t >= time_s, time_t, time_s),
      jnp.where(time_t < time_s, time_t, time_s),
  )

  return time_t, time_s


class FourierEmbedding(nn.Module):
  """Fourier embedding.

  This is one is a more flexible version of the FourierEmbedding in
  swirl_dynamics.lib.diffusion.unets. We also change the activation function to
  silu from swish, and we add a different scaling of the frequencies.

  Attributes:
    dims: The output channel dimension.
    max_freq: The maximum frequency (only used for exponential scaling). The
      default is quite high. For flow-map and mean-flow models, this value
      should be much lower, as the time-derivate in the losses potentially
      renders the training unstable.
    min_freq: The minimum frequency (only used for exponential scaling). The
      default is 1.0 which is the same as the one used in the original
      swirl_dynamics.lib.diffusion.unets. For flow-map and mean-flow models,
      this value should be much lower, the training dynamics seem to benefit
      from having a smaller minimum frequency.
    projection: Whether to add a projection layer.
    act_fun: The activation function.
    frequency_scaling: Type of scaling of the frequencies, either "linear" or
      "exponential".
    frequency_shift: The shift of the frequencies.
    precision: The precision of the computation.
    normalization: Whether to normalize the Fourier embedding following [1].
    dtype: The dtype of the computation.
    param_dtype: The dtype of the parameters.
  """

  dims: int = 64
  max_freq: float = 2e4
  min_freq: float = 1.0
  projection: bool = True
  act_fun: Callable[[Array], Array] = nn.silu
  frequency_scaling: Literal["linear", "exponential"] = "exponential"
  frequency_shift: float = 0.0
  precision: PrecisionLike = None
  normalization: bool = False
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
      logfreqs = jnp.linspace(
          0, jnp.log(self.max_freq / self.min_freq), self.dims // 2
      )
      logfreqs *= jnp.log(self.min_freq)
      logfreqs -= self.frequency_shift
      x = jnp.pi * jnp.exp(logfreqs)[None, :] * x[:, None]
    else:
      raise ValueError(
          f"Unsupported frequency scaling: {self.frequency_scaling}!"
      )

    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    if self.normalization:
      x = x * jnp.sqrt(2)

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
  """UNet model with two time-inputs.

  Attributes:
    out_channels: Number of output channels.
    resize_to_shape: The shape of the internal resolution of the UNet. If None,
      the output resolution will be the same as the input resolution.
    num_channels: Number of channels for each UNet level. The number of levels
      is determined by the length of the tuple.
    downsample_ratio: The downsample ratio for each UNet level. The number of
      levels is determined by the length of the tuple, and it should be the same
      as the number of channels.
    num_blocks: Number of residual blocks in each UNet level.
    noise_embed_dim: Dimension of the Fourier embedding for the noise.
    padding: Type of padding to use in the UNet.
    dropout_rate: Dropout rate for the UNet.
    use_attention: Whether to use attention in the UNet at the coarsest level.
    use_position_encoding: Whether to use position encoding in the UNet.
    num_heads: Number of heads in the self-attention layer.
    normalize_qk: Whether to normalize the query and key in the self-attention
      layer.
    use_global_skip: Whether to use a global skip connection in the UNet.
    cond_resize_method: If the resize_to_shape is defined, this parameter
      defines the method used for resizing the conditional inputs.
    cond_embed_dim: The dimension of the conditional embedding.
    time_embed_act_fun: The function used for the activation of the time
      embedding.
    max_freq: The maximum frequency for the time embedding.
    min_freq: The minimum frequency for the time embedding.
    frequency_scaling: The scaling of the frequencies for the time embedding,
      either "linear" or "exponential".
    frequency_shift: Frequency shift for the time embedding.
    time_embedding_merge: Mode for merging the time embedding stemming from the
      time t and s, either "concat" or "add".
    cond_merging_fn: Function for merging the conditional inputs across the
      channel dimension, which will be concatenated to the input of the UNet.
    cond_embed_fn: Function for embedding the conditional inputs. The embedding
      will be concatenated to the time embedding.
    cond_embed_kwargs: Parameters for the conditional embedding function.
    precision: Precicion for the computation.
    dtype: The data type of the computation.
    param_dtype: The data type of the parameters.
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
  use_global_skip: bool = False
  cond_resize_method: str = "bilinear"
  cond_embed_dim: int = 128
  time_embed_act_fun: Callable[[Array], Array] = nn.silu
  max_freq: float = 2e4
  min_freq: float = 1.0
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
        min_freq=self.min_freq,
        max_freq=self.max_freq,
        act_fun=self.time_embed_act_fun,
        name="time_embedding_for_t",
    )(t)
    emb_s = FourierEmbedding(
        dims=self.noise_embed_dim,
        frequency_scaling=self.frequency_scaling,
        frequency_shift=self.frequency_shift,
        min_freq=self.min_freq,
        max_freq=self.max_freq,
        act_fun=self.time_embed_act_fun,
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
