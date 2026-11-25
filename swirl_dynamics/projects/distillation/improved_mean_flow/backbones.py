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

"""In-Context Transformer and dual-head UNet backbones for Improved Mean Flows.

Implements the adaLN-free multi-token In-Context Transformer (Section 3.3,
Tables 2 and 4, and Appendix A of Geng et al., CVPR 2026, arXiv:2512.02012v1)
with axial 1D/2D RoPE, QK-RMSNorm, SwiGLU, `float32` JVP-safe scaled
dot-product attention, zero-initialized residual block scales, and optional
auxiliary `v`-head, alongside a convolutional dual-head UNet counterpart.
"""

from collections.abc import Callable, Mapping
from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import flow_map
from swirl_dynamics.lib import layers
from swirl_dynamics.lib.diffusion import unets

Array = jax.Array
CondDict = flow_map.CondDict
DTypeLike = jax.typing.DTypeLike
HeadMode = Literal["u", "v", "both"]
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)

_IMF_PRESETS: Mapping[str, Mapping[str, int]] = {
    "B": {
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
        "aux_head_depth": 8,
    },
    "M": {
        "hidden_size": 768,
        "depth": 24,
        "num_heads": 12,
        "aux_head_depth": 8,
    },
    "L": {
        "hidden_size": 1024,
        "depth": 32,
        "num_heads": 16,
        "aux_head_depth": 8,
    },
    "XL": {
        "hidden_size": 1024,
        "depth": 48,
        "num_heads": 16,
        "aux_head_depth": 8,
    },
}

_DEFAULT_SCALAR_COND: Mapping[str, float] = {
    "cfg_scale": 1.0,
    "cfg_t_min": 0.0,
    "cfg_t_max": 1.0,
}


def imf_transformer_config(
    variant: Literal["B", "M", "L", "XL"], **overrides: Any
) -> dict[str, Any]:
  """Returns Table 2/4 `InContextMeanFlowTransformer` preset kwargs.

  Args:
    variant: Model scale preset (`"B"`, `"M"`, `"L"`, or `"XL"`).
    **overrides: Keyword overrides merged on top of the preset.

  Returns:
    Dictionary of keyword arguments for `InContextMeanFlowTransformer`.

  Raises:
    ValueError: If `variant` is not one of `"B"`, `"M"`, `"L"`, `"XL"`.
  """
  if variant not in _IMF_PRESETS:
    raise ValueError(
        f"Unknown iMF variant {variant!r}; expected one of "
        f"{sorted(_IMF_PRESETS)}."
    )
  return {**_IMF_PRESETS[variant], **overrides}


def _safe_num_groups(channels: int, max_groups: int = 32) -> int:
  """Returns a valid GroupNorm count dividing `channels` (<= `max_groups`)."""
  target = max(1, min(channels // 4, max_groups))
  for groups in range(target, 0, -1):
    if channels % groups == 0:
      return groups
  return 1


def _build_1d_rope(length: int, dim: int, base: float) -> tuple[Array, Array]:
  """Builds `float32` 1D RoPE `(cos, sin)` tables `(length, dim // 2)`."""
  inv_freq = base ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
  positions = jnp.arange(length, dtype=jnp.float32)
  angles = positions[:, None] * inv_freq[None, :]
  return jnp.cos(angles), jnp.sin(angles)


def _rotate_half_pairs(x: Array, cos: Array, sin: Array) -> Array:
  """Applies pairwise rotary rotation in `float32` to the last axis of `x`."""
  x_f32 = jnp.asarray(x, dtype=jnp.float32)
  x_even = x_f32[..., 0::2]
  x_odd = x_f32[..., 1::2]
  cos_b = cos[None, :, None, :]
  sin_b = sin[None, :, None, :]
  rot_even = x_even * cos_b - x_odd * sin_b
  rot_odd = x_even * sin_b + x_odd * cos_b
  stacked = jnp.stack([rot_even, rot_odd], axis=-1)
  return stacked.reshape(x.shape).astype(x.dtype)


def _build_axial_rope(
    grid_shape: tuple[int, ...], head_dim: int, base: float
) -> tuple[Array, Array]:
  """Builds 1D or axial 2D `(cos, sin)` tables for spatial tokens."""
  if len(grid_shape) == 1:
    return _build_1d_rope(grid_shape[0], head_dim, base)
  h_grid, w_grid = grid_shape
  half_dim = head_dim // 2
  cos_h, sin_h = _build_1d_rope(h_grid, half_dim, base)
  cos_w, sin_w = _build_1d_rope(w_grid, half_dim, base)
  cos_h_2d = jnp.broadcast_to(
      cos_h[:, None, :], (h_grid, w_grid, half_dim // 2)
  )
  sin_h_2d = jnp.broadcast_to(
      sin_h[:, None, :], (h_grid, w_grid, half_dim // 2)
  )
  cos_w_2d = jnp.broadcast_to(
      cos_w[None, :, :], (h_grid, w_grid, half_dim // 2)
  )
  sin_w_2d = jnp.broadcast_to(
      sin_w[None, :, :], (h_grid, w_grid, half_dim // 2)
  )
  cos = jnp.concatenate([cos_h_2d, cos_w_2d], axis=-1).reshape(
      (h_grid * w_grid, head_dim // 2)
  )
  sin = jnp.concatenate([sin_h_2d, sin_w_2d], axis=-1).reshape(
      (h_grid * w_grid, head_dim // 2)
  )
  return cos, sin


class _InContextBlock(nn.Module):
  """AdaLN-free pre-RMSNorm Transformer block with axial RoPE and SwiGLU.

  Attributes:
    hidden_size: Model channel width `D`.
    num_heads: Number of attention heads.
    mlp_dim: Hidden width of the SwiGLU gated feed-forward branch.
    normalize_qk: Whether to apply per-head RMSNorm to queries and keys.
    dropout_rate: Dropout probability on attention and FFN residual projections.
    init_variance_scale: Variance scaling factor for linear projections.
    residual_scale_init: Initial value of Appendix A per-channel residual scales
      `gamma_attn` and `gamma_mlp`. Set to `None` for un-gated residuals.
    precision: Numerical precision for dense projections.
    dtype: Compute dtype.
    param_dtype: Parameter dtype.
  """

  hidden_size: int
  num_heads: int
  mlp_dim: int
  normalize_qk: bool = True
  dropout_rate: float = 0.0
  init_variance_scale: float = 0.1
  residual_scale_init: float | None = 0.0
  precision: PrecisionLike = None
  dtype: DTypeLike = jnp.float32
  param_dtype: DTypeLike = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: Array,
      rope_cos: Array,
      rope_sin: Array,
      num_prefix: int,
      *,
      is_training: bool,
  ) -> Array:
    """Applies pre-RMSNorm attention + SwiGLU block to `(B, T, D)` tokens."""
    batch_size, seq_len, hidden = x.shape
    head_dim = hidden // self.num_heads
    proj_init = nn.initializers.variance_scaling(
        self.init_variance_scale, "fan_in", "normal"
    )

    h = nn.RMSNorm(
        dtype=self.dtype, param_dtype=self.param_dtype, name="attn_norm"
    )(x)
    qkv = nn.Dense(
        features=3 * hidden,
        use_bias=False,
        kernel_init=proj_init,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="qkv_proj",
    )(h)
    qkv = qkv.reshape((batch_size, seq_len, 3, self.num_heads, head_dim))
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    if self.normalize_qk:
      q = nn.RMSNorm(
          dtype=self.dtype, param_dtype=self.param_dtype, name="q_norm"
      )(q)
      k = nn.RMSNorm(
          dtype=self.dtype, param_dtype=self.param_dtype, name="k_norm"
      )(k)

    q_sp = _rotate_half_pairs(q[:, num_prefix:], rope_cos, rope_sin)
    k_sp = _rotate_half_pairs(k[:, num_prefix:], rope_cos, rope_sin)
    q = jnp.concatenate([q[:, :num_prefix], q_sp], axis=1)
    k = jnp.concatenate([k[:, :num_prefix], k_sp], axis=1)

    scale = jnp.float32(head_dim**-0.5)
    logits = (
        jnp.einsum(
            "bthd,bshd->bhts",
            jnp.asarray(q, dtype=jnp.float32),
            jnp.asarray(k, dtype=jnp.float32),
        )
        * scale
    )
    attn = jax.nn.softmax(logits, axis=-1)
    h_attn = jnp.einsum(
        "bhts,bshd->bthd", attn, jnp.asarray(v, dtype=jnp.float32)
    ).astype(self.dtype)
    h_attn = h_attn.reshape((batch_size, seq_len, hidden))
    h_attn = nn.Dense(
        features=hidden,
        use_bias=False,
        kernel_init=proj_init,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="out_proj",
    )(h_attn)
    if self.dropout_rate > 0.0:
      h_attn = nn.Dropout(rate=self.dropout_rate)(
          h_attn, deterministic=not is_training
      )

    if self.residual_scale_init is not None:
      gamma_attn = self.param(
          "gamma_attn",
          nn.initializers.constant(self.residual_scale_init),
          (hidden,),
          self.param_dtype,
      )
      x = x + gamma_attn.astype(self.dtype) * h_attn
    else:
      x = x + h_attn

    h_ffn = nn.RMSNorm(
        dtype=self.dtype, param_dtype=self.param_dtype, name="ffn_norm"
    )(x)
    gate_up = nn.Dense(
        features=2 * self.mlp_dim,
        use_bias=False,
        kernel_init=proj_init,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="ffn_gate_up",
    )(h_ffn)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    h_ffn = nn.Dense(
        features=hidden,
        use_bias=False,
        kernel_init=proj_init,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="ffn_down",
    )(nn.silu(gate) * up)
    if self.dropout_rate > 0.0:
      h_ffn = nn.Dropout(rate=self.dropout_rate)(
          h_ffn, deterministic=not is_training
      )

    if self.residual_scale_init is not None:
      gamma_mlp = self.param(
          "gamma_mlp",
          nn.initializers.constant(self.residual_scale_init),
          (hidden,),
          self.param_dtype,
      )
      x = x + gamma_mlp.astype(self.dtype) * h_ffn
    else:
      x = x + h_ffn

    return x


class InContextMeanFlowTransformer(nn.Module):
  """AdaLN-free multi-token In-Context Transformer for Improved Mean Flows.

  Conditions on `(t, t - s)` plus optional class label `c`, CFG scale `omega`,
  and CFG interval `(t_min, t_max)` as prepended in-context prefix tokens
  (8 unconditional tokens; 28 full CFG-interval tokens with defaults).

  Attributes:
    out_channels: Number of field channels.
    hidden_size: Transformer embedding dimension `D`.
    depth: Total number of blocks along a single forward pass.
    num_heads: Number of self-attention heads.
    patch_size: Patch side length along each spatial axis (1D or 2D).
    mlp_ratio: Ratio of SwiGLU hidden width to `hidden_size`.
    aux_head_depth: Number of trailing blocks split into dedicated `u_head` and
      auxiliary `v_head` branches (`0 <= aux_head_depth < depth`).
    num_classes: Optional number of classes (allocates `num_classes + 1` with
      index `num_classes` reserved as the CFG null class).
    use_cfg_conditioning: Whether to prepend CFG scale `omega` tokens.
    use_interval_conditioning: Whether to prepend `(t_min, t_max)` tokens.
    num_time_tokens: Prefix tokens allocated to each of `t` and `t - s`.
    num_class_tokens: Prefix tokens allocated to the class condition.
    num_cfg_tokens: Prefix tokens allocated to `cfg_scale`.
    num_interval_tokens: Prefix tokens allocated to each of `t_min` and `t_max`.
    label_key: Conditioning dictionary key for class labels.
    cfg_scale_key: Conditioning dictionary key for CFG scale `omega`.
    cfg_t_min_key: Conditioning dictionary key for CFG interval lower bound.
    cfg_t_max_key: Conditioning dictionary key for CFG interval upper bound.
    normalize_qk: Whether to apply QK-RMSNorm inside attention blocks.
    dropout_rate: Dropout probability in attention and SwiGLU residual branches.
    min_freq: Minimum Fourier frequency for scalar conditioning tokens.
    max_freq: Maximum Fourier frequency for scalar conditioning tokens.
    rope_base: Rotary embedding base wavelength.
    init_variance_scale: Variance scale for QKV/FFN/output projections.
    residual_scale_init: Initial value of per-channel residual scales `gamma`
      (`0.0` per Appendix A for exact identity residual blocks, or `None`).
    zero_init_output: Whether to zero-initialize `u_head_proj` and
      `v_head_proj`.
    precision: Numerical precision for dense layers.
    dtype: Compute dtype.
    param_dtype: Parameter dtype.
  """

  out_channels: int
  hidden_size: int = 768
  depth: int = 12
  num_heads: int = 12
  patch_size: int = 2
  mlp_ratio: float = 8.0 / 3.0
  aux_head_depth: int = 0
  num_classes: int | None = None
  use_cfg_conditioning: bool = False
  use_interval_conditioning: bool = False
  num_time_tokens: int = 4
  num_class_tokens: int = 8
  num_cfg_tokens: int = 4
  num_interval_tokens: int = 4
  label_key: str = "label"
  cfg_scale_key: str = "cfg_scale"
  cfg_t_min_key: str = "cfg_t_min"
  cfg_t_max_key: str = "cfg_t_max"
  normalize_qk: bool = True
  dropout_rate: float = 0.0
  min_freq: float = 0.25
  max_freq: float = 256.0
  rope_base: float = 10000.0
  init_variance_scale: float = 0.1
  residual_scale_init: float | None = 0.0
  zero_init_output: bool = True
  precision: PrecisionLike = None
  dtype: DTypeLike = jnp.float32
  param_dtype: DTypeLike = jnp.float32

  @property
  def num_prefix_tokens(self) -> int:
    """Total number of prepended in-context conditioning tokens."""
    total = 2 * self.num_time_tokens
    if self.num_classes is not None:
      total += self.num_class_tokens
    if self.use_cfg_conditioning:
      total += self.num_cfg_tokens
    if self.use_interval_conditioning:
      total += 2 * self.num_interval_tokens
    return total

  def _validate_config(
      self, spatial_shape: tuple[int, ...], head: HeadMode
  ) -> tuple[tuple[int, ...], int]:
    """Validates architecture hyperparameters and spatial dimensions."""
    if head not in ("u", "v", "both"):
      raise ValueError(f"Unsupported head {head!r}; expected 'u', 'v', 'both'.")
    if not 0 <= self.aux_head_depth < self.depth:
      raise ValueError(
          "Expected 0 <= aux_head_depth < depth, got "
          f"aux_head_depth={self.aux_head_depth}, depth={self.depth}."
      )
    if head in ("v", "both") and self.aux_head_depth == 0:
      raise ValueError(f"`head={head!r}` requires `aux_head_depth > 0`, got 0.")
    if self.hidden_size % self.num_heads != 0:
      raise ValueError(
          f"`hidden_size` ({self.hidden_size}) must be divisible by "
          f"`num_heads` ({self.num_heads})."
      )
    head_dim = self.hidden_size // self.num_heads
    spatial_ndim = len(spatial_shape)
    if spatial_ndim not in (1, 2):
      raise ValueError(
          f"Only 1D and 2D spatial fields are supported, got {spatial_shape}."
      )
    if self.patch_size < 1:
      raise ValueError(f"`patch_size` must be >= 1, got {self.patch_size}.")
    for dim in spatial_shape:
      if dim % self.patch_size != 0:
        raise ValueError(
            f"Spatial shape {spatial_shape} must be divisible by "
            f"patch_size={self.patch_size}."
        )
    rope_div = 2 if spatial_ndim == 1 else 4
    if head_dim % rope_div != 0:
      raise ValueError(
          f"head_dim ({head_dim}) must be divisible by {rope_div} for "
          f"{spatial_ndim}D RoPE."
      )
    grid_shape = tuple(dim // self.patch_size for dim in spatial_shape)
    return grid_shape, head_dim

  def _embed_scalar_tokens(
      self, val: Array, num_tokens: int, name: str
  ) -> Array:
    """Maps a `(B,)` scalar to `(B, num_tokens, hidden_size)` prefix tokens."""
    feat = flow_map.FourierEmbedding(
        dims=self.hidden_size,
        min_freq=self.min_freq,
        max_freq=self.max_freq,
        projection=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name=f"{name}_fourier",
    )(val)
    feat = nn.Dense(
        features=self.hidden_size,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name=f"{name}_mlp_in",
    )(feat)
    feat = nn.silu(feat)
    feat = nn.Dense(
        features=self.hidden_size,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name=f"{name}_mlp_out",
    )(feat)
    tok_emb = self.param(
        f"{name}_token_embed",
        nn.initializers.normal(stddev=0.02),
        (1, num_tokens, self.hidden_size),
        self.param_dtype,
    )
    return feat[:, None, :] + tok_emb.astype(self.dtype)

  def _extract_scalar(self, cond: CondDict, key: str, batch_size: int) -> Array:
    """Extracts a 1D `float32` conditioning vector with sensible defaults."""
    if key in cond:
      val = jnp.asarray(cond[key], dtype=jnp.float32)
      if val.ndim == 0:
        return jnp.broadcast_to(val, (batch_size,))
      return val.reshape((batch_size,))
    return jnp.full(
        (batch_size,), _DEFAULT_SCALAR_COND.get(key, 0.0), dtype=jnp.float32
    )

  def _patchify(self, x: Array, grid_shape: tuple[int, ...]) -> Array:
    """Converts `(B, *spatial, C)` to `(B, N_patches, patch_dim)`."""
    batch_size = x.shape[0]
    p = self.patch_size
    if len(grid_shape) == 1:
      return x.reshape((batch_size, grid_shape[0], p * self.out_channels))
    h_p, w_p = grid_shape
    patches = x.reshape((batch_size, h_p, p, w_p, p, self.out_channels))
    patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
    return patches.reshape((batch_size, h_p * w_p, p * p * self.out_channels))

  def _unpatchify(self, tokens: Array, grid_shape: tuple[int, ...]) -> Array:
    """Converts `(B, N_patches, patch_dim)` back to `(B, *spatial, C)`."""
    batch_size = tokens.shape[0]
    p = self.patch_size
    if len(grid_shape) == 1:
      return tokens.reshape((batch_size, grid_shape[0] * p, self.out_channels))
    h_p, w_p = grid_shape
    patches = tokens.reshape((batch_size, h_p, w_p, p, p, self.out_channels))
    patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
    return patches.reshape((batch_size, h_p * p, w_p * p, self.out_channels))

  @nn.compact
  def __call__(
      self,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None = None,
      *,
      is_training: bool,
      head: HeadMode = "u",
  ) -> Array | tuple[Array, Array]:
    """Evaluates average velocity `u`, auxiliary `v`, or `(u, v)`.

    Args:
      x_s: Input field of shape `(batch, L, C)` or `(batch, H, W, C)`.
      t: Upper time coordinate (scalar or `(batch,)`).
      s: Lower time coordinate (scalar or `(batch,)`).
      cond: Optional conditioning mapping (`label`, `cfg_scale`, `cfg_t_min`,
        `cfg_t_max`).
      is_training: Whether the network runs in training mode.
      head: Branch selector (`"u"`, `"v"`, or `"both"`).

    Returns:
      Predicted field matching `x_s.shape` when `head in ("u", "v")`, or
      `(u_out, v_out)` when `head == "both"`.
    """
    t, s = flow_map.validate_flow_map_inputs(x_s, t, s, self.out_channels)
    spatial_shape = tuple(x_s.shape[1:-1])
    grid_shape, head_dim = self._validate_config(spatial_shape, head)
    batch_size = x_s.shape[0]
    cond_dict = cond or {}

    patches = self._patchify(x_s.astype(self.dtype), grid_shape)
    patch_dim = patches.shape[-1]
    sp_tokens = nn.Dense(
        features=self.hidden_size,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="patch_embed",
    )(patches)

    prefix_list = [
        self._embed_scalar_tokens(t, self.num_time_tokens, name="time_t"),
        self._embed_scalar_tokens(t - s, self.num_time_tokens, name="time_dt"),
    ]
    if self.num_classes is not None:
      if self.label_key in cond_dict:
        labels = jnp.asarray(cond_dict[self.label_key], dtype=jnp.int32)
        if labels.ndim == 0:
          labels = jnp.broadcast_to(labels, (batch_size,))
        else:
          labels = labels.reshape((batch_size,))
      else:
        labels = jnp.full((batch_size,), self.num_classes, dtype=jnp.int32)
      cls_feat = nn.Embed(
          num_embeddings=self.num_classes + 1,
          features=self.hidden_size,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name="class_embed",
      )(labels)
      cls_feat = nn.Dense(
          features=self.hidden_size,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name="class_mlp_in",
      )(cls_feat)
      cls_feat = nn.silu(cls_feat)
      cls_feat = nn.Dense(
          features=self.hidden_size,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name="class_mlp_out",
      )(cls_feat)
      cls_tok = self.param(
          "class_token_embed",
          nn.initializers.normal(stddev=0.02),
          (1, self.num_class_tokens, self.hidden_size),
          self.param_dtype,
      )
      prefix_list.append(cls_feat[:, None, :] + cls_tok.astype(self.dtype))

    if self.use_cfg_conditioning:
      w = self._extract_scalar(cond_dict, self.cfg_scale_key, batch_size)
      prefix_list.append(
          self._embed_scalar_tokens(w, self.num_cfg_tokens, name="cfg_scale")
      )

    if self.use_interval_conditioning:
      t_min = self._extract_scalar(cond_dict, self.cfg_t_min_key, batch_size)
      t_max = self._extract_scalar(cond_dict, self.cfg_t_max_key, batch_size)
      prefix_list.append(
          self._embed_scalar_tokens(
              t_min, self.num_interval_tokens, name="cfg_t_min"
          )
      )
      prefix_list.append(
          self._embed_scalar_tokens(
              t_max, self.num_interval_tokens, name="cfg_t_max"
          )
      )

    prefix = jnp.concatenate(prefix_list, axis=1)
    num_prefix = prefix.shape[1]
    tokens = jnp.concatenate([prefix, sp_tokens], axis=1)

    rope_cos, rope_sin = _build_axial_rope(grid_shape, head_dim, self.rope_base)
    mlp_dim = int(self.mlp_ratio * self.hidden_size)
    shared_depth = self.depth - self.aux_head_depth

    block_kwargs = dict(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        mlp_dim=mlp_dim,
        normalize_qk=self.normalize_qk,
        dropout_rate=self.dropout_rate,
        init_variance_scale=self.init_variance_scale,
        residual_scale_init=self.residual_scale_init,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
    )

    h = tokens
    for i in range(shared_depth):
      h = _InContextBlock(**block_kwargs, name=f"shared_block_{i}")(
          h, rope_cos, rope_sin, num_prefix, is_training=is_training
      )

    if self.zero_init_output:
      out_init = nn.initializers.zeros
    else:
      out_init = nn.initializers.variance_scaling(
          self.init_variance_scale, "fan_in", "normal"
      )

    def _eval_head(h_in: Array, prefix_name: str) -> Array:
      h_head = h_in
      for i in range(self.aux_head_depth):
        h_head = _InContextBlock(
            **block_kwargs, name=f"{prefix_name}_block_{i}"
        )(h_head, rope_cos, rope_sin, num_prefix, is_training=is_training)
      h_sp = nn.RMSNorm(
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"{prefix_name}_norm",
      )(h_head[:, num_prefix:])
      out_patches = nn.Dense(
          features=patch_dim,
          kernel_init=out_init,
          bias_init=nn.initializers.zeros,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"{prefix_name}_proj",
      )(h_sp)
      return self._unpatchify(out_patches, grid_shape)

    if head == "u":
      return _eval_head(h, "u_head")
    if head == "v":
      return _eval_head(h, "v_head")
    return _eval_head(h, "u_head"), _eval_head(h, "v_head")


class DualHeadMeanFlowUNet(nn.Module):
  """Convolutional 1D/2D dual-head UNet backbone for Improved Mean Flows.

  Shares `DStack` and `UStack` across the average-velocity head (`head="u"`)
  and an optional auxiliary instantaneous-velocity head (`head="v"`).

  Attributes:
    out_channels: Number of output field channels.
    num_channels: Channel counts per UNet resolution level.
    downsample_ratio: Downsampling ratios per UNet level.
    num_blocks: Number of residual blocks per UNet level.
    aux_num_blocks: Number of head-specific residual blocks per branch.
    noise_embed_dim: Feature dimension for each time Fourier embedding.
    padding: Convolution padding mode.
    dropout_rate: Dropout rate inside residual blocks.
    use_attention: Whether to enable self-attention at the coarsest level.
    use_position_encoding: Whether to use position encoding in `DStack`.
    num_heads: Number of self-attention heads.
    normalize_qk: Whether to apply QK normalization in attention blocks.
    cond_resize_method: Interpolation method for channel conditioning.
    cond_embed_dim: Channel width for `cond_merging_fn`.
    time_embed_act_fun: Activation function for Fourier embeddings.
    min_freq: Minimum Fourier frequency.
    max_freq: Maximum Fourier frequency.
    frequency_scaling: Frequency spacing (`"linear"` or `"exponential"`).
    time_conditioning: Time pair embedded (`"t_dt"` or `"t_s"`).
    time_embedding_merge: How to combine the two time embeddings.
    time_rescale: Rescaling factor on `[0, 1]` times.
    cond_merging_fn: Optional channel-wise conditioning module class.
    cond_embed_fn: Optional embedding-wise conditioning module class/factory.
    cond_embed_kwargs: Optional keyword arguments for `cond_embed_fn`.
    zero_init_output: Whether to zero-initialize head output convolutions.
    precision: Numerical precision for convolutions and dense layers.
    dtype: Compute dtype.
    param_dtype: Parameter dtype.
  """

  out_channels: int
  num_channels: tuple[int, ...] = (64, 128)
  downsample_ratio: tuple[int, ...] = (2, 2)
  num_blocks: int = 2
  aux_num_blocks: int = 1
  noise_embed_dim: int = 64
  padding: str = "CIRCULAR"
  dropout_rate: float = 0.0
  use_attention: bool = False
  use_position_encoding: bool = False
  num_heads: int = 4
  normalize_qk: bool = True
  cond_resize_method: str = "bilinear"
  cond_embed_dim: int = 64
  time_embed_act_fun: Callable[[Array], Array] = nn.silu
  min_freq: float = 0.25
  max_freq: float = 256.0
  frequency_scaling: Literal["linear", "exponential"] = "exponential"
  time_conditioning: Literal["t_dt", "t_s"] = "t_dt"
  time_embedding_merge: Literal["concat", "add"] = "concat"
  time_rescale: float = 1.0
  cond_merging_fn: type[unets.MergeChannelCond] | None = None
  cond_embed_fn: Callable[..., nn.Module] | None = None
  cond_embed_kwargs: Mapping[str, Any] | None = None
  zero_init_output: bool = True
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
      head: HeadMode = "u",
  ) -> Array | tuple[Array, Array]:
    """Evaluates average velocity `u`, auxiliary `v`, or `(u, v)`."""
    if head not in ("u", "v", "both"):
      raise ValueError(f"Unsupported head {head!r}; expected 'u', 'v', 'both'.")
    if self.aux_num_blocks < 0:
      raise ValueError(
          f"`aux_num_blocks` must be >= 0, got {self.aux_num_blocks}."
      )
    t, s = flow_map.validate_flow_map_inputs(x_s, t, s, self.out_channels)
    t_scaled = t * self.time_rescale
    s_scaled = s * self.time_rescale
    if self.time_conditioning == "t_dt":
      second_time = t_scaled - s_scaled
    elif self.time_conditioning == "t_s":
      second_time = s_scaled
    else:
      raise ValueError(
          f"Unsupported time_conditioning: {self.time_conditioning!r}."
      )

    kernel_dim = x_s.ndim - 2
    cond_dict = dict(cond) if cond is not None else {}
    h_in = x_s
    if self.cond_merging_fn is not None:
      h_in = self.cond_merging_fn(
          embed_dim=self.cond_embed_dim,
          resize_method=self.cond_resize_method,
          kernel_size=(3,) * kernel_dim,
          padding=self.padding,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(h_in, cond_dict)

    emb_t = flow_map.FourierEmbedding(
        dims=self.noise_embed_dim,
        frequency_scaling=self.frequency_scaling,
        min_freq=self.min_freq,
        max_freq=self.max_freq,
        act_fun=self.time_embed_act_fun,
        precision=self.precision,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        name="time_embedding_for_t",
    )(t_scaled)
    emb_s = flow_map.FourierEmbedding(
        dims=self.noise_embed_dim,
        frequency_scaling=self.frequency_scaling,
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
          f"Unsupported time_embedding_merge: {self.time_embedding_merge!r}."
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
    )(h_in, emb, is_training=is_training)
    shared = unets.UStack(
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

    if self.zero_init_output:
      out_init = nn.initializers.zeros
    else:
      out_init = unets.default_init()

    def _eval_head(feat: Array, prefix_name: str) -> Array:
      h = feat
      channels = h.shape[-1]
      for i in range(self.aux_num_blocks):
        residual = h
        h = nn.silu(
            nn.GroupNorm(
                _safe_num_groups(channels), name=f"{prefix_name}_gn1_{i}"
            )(h)
        )
        h = layers.ConvLayer(
            features=channels,
            kernel_size=(3,) * kernel_dim,
            padding=self.padding,
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"{prefix_name}_conv1_{i}",
        )(h)
        h = unets.AdaptiveScale(
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"{prefix_name}_scale_{i}",
        )(h, emb)
        h = nn.silu(
            nn.GroupNorm(
                _safe_num_groups(channels), name=f"{prefix_name}_gn2_{i}"
            )(h)
        )
        if self.dropout_rate > 0.0:
          h = nn.Dropout(rate=self.dropout_rate)(
              h, deterministic=not is_training
          )
        h = layers.ConvLayer(
            features=channels,
            kernel_size=(3,) * kernel_dim,
            padding=self.padding,
            kernel_init=unets.default_init(),
            precision=self.precision,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=f"{prefix_name}_conv2_{i}",
        )(h)
        h = residual + h
      h = nn.swish(
          nn.GroupNorm(
              _safe_num_groups(channels), name=f"{prefix_name}_out_gn"
          )(h)
      )
      return layers.ConvLayer(
          features=self.out_channels,
          kernel_size=(3,) * kernel_dim,
          padding=self.padding,
          kernel_init=out_init,
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          name=f"{prefix_name}_conv_out",
      )(h)

    if head == "u":
      return _eval_head(shared, "u_head")
    if head == "v":
      return _eval_head(shared, "v_head")
    return _eval_head(shared, "u_head"), _eval_head(shared, "v_head")
