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

"""Two-time UNet backbones and embeddings for flow-map and mean-flow models.

References:
  [1] Analyzing and Improving the Training Dynamics of Diffusion Models.
      Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila,
      and Samuli Laine. CVPR 2024.
  [2] Flow Map Matching with Stochastic Interpolants: A Mathematical Framework
      for Consistency Models. Nicholas M. Boffi, Michael S. Albergo, and
      Eric Vanden-Eijnden. 2024.
  [3] Mean Flows for One-step Generative Modeling. Zhengyang Geng, Mingyang
      Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. 2025.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets

Array = jax.Array
ArrayShape = Sequence[int]
CondDict = Mapping[str, Array]
DTypeLike = jax.typing.DTypeLike
MergeChannelCond = unets.MergeChannelCond
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)

_DEFAULT_SCALAR_VALUES: Mapping[str, float] = {
    "cfg_scale": 1.0,
    "cfg_t_min": 0.0,
    "cfg_t_max": 1.0,
}


def _safe_num_groups(channels: int, max_groups: int = 32) -> int:
  """Returns a valid GroupNorm count dividing `channels` (<= `max_groups`)."""
  target = max(1, min(channels // 4, max_groups))
  for groups in range(target, 0, -1):
    if channels % groups == 0:
      return groups
  return 1


def validate_flow_map_inputs(
    x_s: Array, t: Array, s: Array, out_channels: int
) -> tuple[Array, Array]:
  """Validates channel/time shapes and broadcasts scalar `t` and `s`.

  Args:
    x_s: Input state array of shape `(batch, *spatial_dims, channels)`.
    t: Upper time coordinate (scalar or 1D array of shape `(batch,)`).
    s: Lower time coordinate (scalar or 1D array of shape `(batch,)`).
    out_channels: Expected channel dimension of `x_s`.

  Returns:
    Tuple `(t_vec, s_vec)` of 1D arrays of shape `(batch,)`.

  Raises:
    ValueError: If channel counts differ, `t.ndim != s.ndim`, or the batch
      dimensions of `x_s`, `t`, and `s` do not match.
  """
  if x_s.shape[-1] != out_channels:
    raise ValueError(
        f"Number of channels in the input ({x_s.shape[-1]}) must match the"
        f" number of channels in the output ({out_channels})."
    )
  if t.ndim != s.ndim:
    raise ValueError(f"Time dimensions must be equal ({t.ndim} != {s.ndim}).")

  if s.ndim < 1:
    s = jnp.broadcast_to(s, (x_s.shape[0],))
    t = jnp.broadcast_to(t, (x_s.shape[0],))

  if s.ndim != 1 or x_s.shape[0] != s.shape[0] or x_s.shape[0] != t.shape[0]:
    raise ValueError(
        "s must be 1D and have the same leading (batch) dimension as x and t."
        f" Instead x_s has leading dimension ({x_s.shape[0]}), s has leading"
        f" dimension ({s.shape[0]}), and t has leading dimension"
        f" ({t.shape[0]})."
    )
  return t, s


class FourierEmbedding(nn.Module):
  """Fourier time/scalar embedding with linear or exponential frequency scaling.

  Phases and trigonometric evaluations are performed in `float32` prior to
  casting to `dtype`, preserving forward-mode JVP accuracy under `bfloat16`.

  Attributes:
    dims: Even output embedding dimension (`>= 2`).
    max_freq: Maximum frequency for `"exponential"` scaling (`>= min_freq`).
    min_freq: Minimum strictly positive frequency for `"exponential"` scaling.
    projection: Whether to apply a two-layer MLP projection after `sin`/`cos`.
    act_fun: Activation function for the MLP projection.
    frequency_scaling: Frequency spacing mode (`"linear"` or `"exponential"`).
    frequency_shift: Phase offset (for `"linear"`) or log-frequency shift (for
      `"exponential"`).
    precision: Numerical precision passed to dense layers.
    normalization: Whether to multiply sinusoidal features by `sqrt(2)` per [1].
    dtype: Compute dtype of outputs and activations.
    param_dtype: Parameter dtype for projection layers.
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
  dtype: DTypeLike = jnp.float32
  param_dtype: DTypeLike = jnp.float32

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """Embeds a 1D batch of scalars into `(batch, dims)`.

    Args:
      x: 1D scalar batch of shape `(batch,)`.

    Returns:
      Embedded features of shape `(batch, dims)` with dtype `self.dtype`.

    Raises:
      ValueError: If `dims` is not an even integer >= 2, `x` is not 1D,
        `min_freq`/`max_freq` are invalid, or `frequency_scaling` is unknown.
    """
    if self.dims < 2 or self.dims % 2 != 0:
      raise ValueError(f"`dims` must be an even integer >= 2, got {self.dims}.")
    if x.ndim != 1:
      raise ValueError(f"`x` must be 1D, got shape {x.shape}.")

    x_f32 = jnp.asarray(x, dtype=jnp.float32)
    half_dims = self.dims // 2
    if self.frequency_scaling == "linear":
      freqs = jnp.arange(half_dims, dtype=jnp.float32)
      phase = jnp.pi * freqs[None, :] * x_f32[:, None] - self.frequency_shift
    elif self.frequency_scaling == "exponential":
      if self.min_freq <= 0.0 or self.max_freq < self.min_freq:
        raise ValueError(
            "Expected `0 < min_freq <= max_freq` for exponential scaling, got "
            f"min_freq={self.min_freq}, max_freq={self.max_freq}."
        )
      logfreqs = (
          jnp.linspace(
              jnp.log(self.min_freq),
              jnp.log(self.max_freq),
              half_dims,
              dtype=jnp.float32,
          )
          - self.frequency_shift
      )
      phase = jnp.pi * jnp.exp(logfreqs)[None, :] * x_f32[:, None]
    else:
      raise ValueError(
          f"Unsupported frequency scaling: {self.frequency_scaling!r}."
      )

    out = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)
    if self.normalization:
      out = out * jnp.sqrt(jnp.float32(2.0))
    out = out.astype(self.dtype)

    if self.projection:
      out = nn.Dense(
          features=2 * self.dims,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(out)
      out = self.act_fun(out)
      out = nn.Dense(
          features=self.dims,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(out)
    return out


class MergeCategoricalEmbCond(unets.MergeEmdCond):
  """Merges a categorical class embedding additively into the time embedding.

  Attributes:
    cond_key: Dictionary key containing integer class labels of shape `(batch,)`
      or `(batch, 1)`.
    num_classes: Number of categorical classes in the embedding table.
    features_embedding: Hidden width of the class lookup table.
    act_fun: Activation function used in the projection MLP.
    precision: Precision for dense projections.
    dtype: Compute dtype.
    param_dtype: Parameter dtype.
  """

  cond_key: str = "emb:label"
  num_classes: int = 10
  features_embedding: int = 128
  act_fun: Callable[[Array], Array] = nn.silu
  precision: PrecisionLike = None
  dtype: DTypeLike = jnp.float32
  param_dtype: DTypeLike = jnp.float32

  @nn.compact
  def __call__(
      self, emb: Array, cond: Mapping[str, Array], is_training: bool
  ) -> Array:
    """Projects categorical `cond[cond_key]` and adds it to `emb`."""
    del is_training
    labels = jnp.asarray(cond[self.cond_key], dtype=jnp.int32).reshape((
        emb.shape[0],
    ))
    cond_emb = nn.Embed(
        num_embeddings=self.num_classes,
        features=self.features_embedding,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )(labels)
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


class MergeGuidanceEmbCond(unets.MergeEmdCond):
  """Merges optional class label and scalar CFG guidance into time embeddings.

  Reserves index `num_classes` as the null class for classifier-free guidance
  when `label_key` is set, and encodes each scalar key in `scalar_keys` with
  `FourierEmbedding` followed by a linear projection into `emb.shape[-1]`.

  Attributes:
    label_key: Optional key for integer class labels (`0..num_classes`, where
      `num_classes` denotes the null class). When absent from `cond`, defaults
      to the null class index. Set to `None` to disable class conditioning.
    num_classes: Number of data classes (table allocates `num_classes + 1`).
    scalar_keys: Tuple of scalar guidance keys in `cond` to embed.
    features_embedding: Feature width for class and Fourier embeddings.
    act_fun: Activation function for class and Fourier MLPs.
    min_freq: Minimum Fourier frequency for scalar guidance keys.
    max_freq: Maximum Fourier frequency for scalar guidance keys.
    precision: Numerical precision for dense layers.
    dtype: Compute dtype.
    param_dtype: Parameter dtype.
  """

  label_key: str | None = "label"
  num_classes: int = 10
  scalar_keys: tuple[str, ...] = ("cfg_scale", "cfg_t_min", "cfg_t_max")
  features_embedding: int = 128
  act_fun: Callable[[Array], Array] = nn.silu
  min_freq: float = 0.25
  max_freq: float = 256.0
  precision: PrecisionLike = None
  dtype: DTypeLike = jnp.float32
  param_dtype: DTypeLike = jnp.float32

  @nn.compact
  def __call__(
      self, emb: Array, cond: Mapping[str, Array], is_training: bool
  ) -> Array:
    """Adds class and scalar CFG embeddings to `emb` of shape `(batch, D)`."""
    del is_training
    batch_size, out_dim = emb.shape[0], emb.shape[-1]
    out = emb

    if self.label_key is not None:
      if self.label_key in cond:
        labels = jnp.asarray(cond[self.label_key], dtype=jnp.int32)
        if labels.ndim == 0:
          labels = jnp.broadcast_to(labels, (batch_size,))
        else:
          labels = labels.reshape((batch_size,))
      else:
        labels = jnp.full((batch_size,), self.num_classes, dtype=jnp.int32)
      cls_emb = nn.Embed(
          num_embeddings=self.num_classes + 1,
          features=self.features_embedding,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name="class_embed",
      )(labels)
      cls_emb = nn.Dense(
          features=2 * self.features_embedding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name="class_mlp_in",
      )(cls_emb)
      cls_emb = self.act_fun(cls_emb)
      cls_emb = nn.Dense(
          features=out_dim,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name="class_mlp_out",
      )(cls_emb)
      out = out + cls_emb

    for key in self.scalar_keys:
      if key in cond:
        val = jnp.asarray(cond[key], dtype=jnp.float32)
        if val.ndim == 0:
          val = jnp.broadcast_to(val, (batch_size,))
        else:
          val = val.reshape((batch_size,))
      else:
        default_val = _DEFAULT_SCALAR_VALUES.get(key, 0.0)
        val = jnp.full((batch_size,), default_val, dtype=jnp.float32)
      scalar_feat = FourierEmbedding(
          dims=self.features_embedding,
          min_freq=self.min_freq,
          max_freq=self.max_freq,
          act_fun=self.act_fun,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"scalar_embed_{key}",
      )(val)
      scalar_proj = nn.Dense(
          features=out_dim,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"scalar_proj_{key}",
      )(scalar_feat)
      out = out + scalar_proj

    return out


class FlowMapUNet(nn.Module):
  """N-D UNet backbone conditioned on two time inputs `(t, s)` or `(t, t - s)`.

  Attributes:
    out_channels: Number of output channels.
    resize_to_shape: Optional internal spatial shape for `FilteredResize`.
    num_channels: Channel counts per UNet resolution level.
    downsample_ratio: Spatial downsampling factors per UNet level.
    num_blocks: Number of residual blocks per resolution level.
    noise_embed_dim: Feature dimension for each time Fourier embedding.
    padding: Convolution padding mode (e.g., `"CIRCULAR"`, `"SAME"`).
    dropout_rate: Dropout probability in residual blocks.
    use_attention: Whether to enable attention at the coarsest level.
    use_position_encoding: Whether to add sinusoidal position encoding in
      DStack.
    num_heads: Number of self-attention heads.
    normalize_qk: Whether to normalize queries and keys in attention blocks.
    use_global_skip: Whether to subtract the input `x_s` from the UNet output.
    cond_resize_method: Interpolation method for channel conditioning.
    cond_embed_dim: Channel projection width for `cond_merging_fn`.
    time_embed_act_fun: Activation function for Fourier time embeddings.
    max_freq: Maximum frequency for time embeddings.
    min_freq: Minimum frequency for time embeddings.
    frequency_scaling: Frequency scaling mode (`"linear"` or `"exponential"`).
    frequency_shift: Frequency shift for time embeddings.
    time_embedding_merge: How to combine the two time embeddings (`"concat"` or
      `"add"`).
    time_conditioning: Pair of time quantities embedded (`"t_s"` embeds `(t, s)`
      while `"t_dt"` embeds `(t, t - s)`).
    cond_merging_fn: Optional channel-wise conditioning module class.
    cond_embed_fn: Optional embedding-wise conditioning module class/factory.
    cond_embed_kwargs: Optional keyword arguments for `cond_embed_fn`.
    precision: Numerical precision for convolutions and dense layers.
    dtype: Compute dtype.
    param_dtype: Parameter dtype.
  """

  out_channels: int
  resize_to_shape: tuple[int, ...] | None = None
  num_channels: tuple[int, ...] = (128, 256, 256, 256)
  downsample_ratio: tuple[int, ...] = (2, 2, 2, 2)
  num_blocks: int = 4
  noise_embed_dim: int = 128
  padding: str = "CIRCULAR"
  dropout_rate: float = 0.0
  use_attention: bool = True
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
  time_conditioning: Literal["t_s", "t_dt"] = "t_s"
  cond_merging_fn: type[unets.MergeChannelCond] | None = None
  cond_embed_fn: Callable[..., nn.Module] | None = None
  cond_embed_kwargs: Mapping[str, Any] | None = None
  precision: PrecisionLike = None
  dtype: DTypeLike = jnp.float32
  param_dtype: DTypeLike = jnp.float32

  @nn.compact
  def __call__(
      self,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Evaluates the two-time UNet on `(x_s, t, s, cond)`.

    Args:
      x_s: Input field of shape `(batch, *spatial_dims, channels)`.
      t: Target time coordinate (scalar or 1D `(batch,)`).
      s: Source time coordinate (scalar or 1D `(batch,)`).
      cond: Optional conditioning dictionary.
      is_training: Whether the network runs in training mode.

    Returns:
      Output field of shape `(batch, *spatial_dims, out_channels)`.

    Raises:
      ValueError: If `t` or `s` has invalid shape, or if `time_conditioning` or
        `time_embedding_merge` is unsupported.
    """
    batch_size = x_s.shape[0]
    if t.ndim < 1:
      t = jnp.broadcast_to(t, (batch_size,))
    if s.ndim < 1:
      s = jnp.broadcast_to(s, (batch_size,))

    if t.ndim != 1 or batch_size != t.shape[0]:
      raise ValueError(
          "t must be 1D and have the same leading (batch) dimension as x"
          f" ({batch_size}), got shape {t.shape}."
      )
    if s.ndim != 1 or batch_size != s.shape[0]:
      raise ValueError(
          "s must be 1D and have the same leading (batch) dimension as x"
          f" ({batch_size}), got shape {s.shape}."
      )

    skip = x_s
    input_size = x_s.shape[1:-1]
    resize_kernel = (7,) * len(input_size)

    if self.resize_to_shape is not None:
      x_s = layers.FilteredResize(
          output_size=self.resize_to_shape,
          kernel_size=resize_kernel,
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x_s)

    kernel_dim = x_s.ndim - 2
    cond_dict = dict(cond) if cond is not None else {}
    if self.cond_merging_fn is not None:
      x_s = self.cond_merging_fn(
          embed_dim=self.cond_embed_dim,
          resize_method=self.cond_resize_method,
          kernel_size=(3,) * kernel_dim,
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x_s, cond_dict)

    if self.time_conditioning == "t_s":
      second_time = s
    elif self.time_conditioning == "t_dt":
      second_time = t - s
    else:
      raise ValueError(
          f"Unsupported time_conditioning: {self.time_conditioning!r}."
      )

    emb_t = FourierEmbedding(
        dims=self.noise_embed_dim,
        frequency_scaling=self.frequency_scaling,
        frequency_shift=self.frequency_shift,
        min_freq=self.min_freq,
        max_freq=self.max_freq,
        act_fun=self.time_embed_act_fun,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="time_embedding_for_t",
    )(t)
    emb_s = FourierEmbedding(
        dims=self.noise_embed_dim,
        frequency_scaling=self.frequency_scaling,
        frequency_shift=self.frequency_shift,
        min_freq=self.min_freq,
        max_freq=self.max_freq,
        act_fun=self.time_embed_act_fun,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="time_embedding_for_s",
    )(second_time)

    if self.time_embedding_merge == "concat":
      emb = jnp.concatenate([emb_t, emb_s], axis=-1)
    elif self.time_embedding_merge == "add":
      emb = emb_t + emb_s
    else:
      raise ValueError(
          f"Unsupported time embedding merge: {self.time_embedding_merge!r}."
      )

    if self.cond_embed_fn is not None:
      emb = self.cond_embed_fn(**dict(self.cond_embed_kwargs or {}))(
          emb, cond_dict, is_training=is_training
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

    h = nn.swish(nn.GroupNorm(_safe_num_groups(h.shape[-1]))(h))
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

    if self.resize_to_shape is not None:
      x_t = layers.FilteredResize(
          output_size=input_size,
          kernel_size=resize_kernel,
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(x_t)

    return x_t - skip if self.use_global_skip else x_t


class RescaledFlowMapUNet(FlowMapUNet):
  """Flow-map UNet imposing exact boundary condition `X_{s, s}(x_s) = x_s`.

  Attributes:
    time_rescale: Multiplicative factor rescaling `[0, 1]` time inputs prior to
      Fourier embedding.
  """

  time_rescale: float = 1.0

  @nn.compact
  def __call__(
      self,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Evaluates `x_s * (1 - (t - s)) + (t - s) * s_theta(x_s, t, s, cond)`."""
    t, s = validate_flow_map_inputs(x_s, t, s, self.out_channels)
    s_theta = super().__call__(
        x_s,
        t * self.time_rescale,
        s * self.time_rescale,
        cond,
        is_training=is_training,
    )
    delta_time = (t - s).reshape((s.shape[0],) + (x_s.ndim - 1) * (1,))
    delta_time = delta_time.astype(x_s.dtype)
    return x_s * (1.0 - delta_time) + delta_time * s_theta


class RescaledMeanFlowUNet(FlowMapUNet):
  """Mean-flow UNet predicting average velocity `u_theta(x_s, t, s, cond)`.

  Attributes:
    time_rescale: Multiplicative factor rescaling `[0, 1]` time inputs prior to
      Fourier embedding.
  """

  time_rescale: float = 1.0

  @nn.compact
  def __call__(
      self,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Evaluates `u_theta(x_s, t * time_rescale, s * time_rescale, cond)`."""
    t, s = validate_flow_map_inputs(x_s, t, s, self.out_channels)
    return super().__call__(
        x_s,
        t * self.time_rescale,
        s * self.time_rescale,
        cond,
        is_training=is_training,
    )
