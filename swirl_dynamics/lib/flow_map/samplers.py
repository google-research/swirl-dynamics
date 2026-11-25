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

"""Time/guidance samplers, loss helpers, and integrators for flow maps."""

from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
ArrayShape = Sequence[int]
CondDict = Mapping[str, Array]
DTypeLike = jax.typing.DTypeLike


def logit_normal_dist(
    rng: Array,
    shape: ArrayShape,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: DTypeLike = jnp.float32,
) -> Array:
  """Samples logit-normal times in `(0, 1)` with `std >= 0`.

  Args:
    rng: JAX PRNG key.
    shape: Output sample shape.
    mean: Mean of the underlying Gaussian distribution prior to `sigmoid`.
    std: Non-negative standard deviation of the underlying Gaussian.
    dtype: Floating-point dtype of the returned array.

  Returns:
    Array of samples in `(0, 1)` with shape `tuple(shape)` and dtype `dtype`.

  Raises:
    ValueError: If `std < 0`.
  """
  if std < 0.0:
    raise ValueError(f"`std` must be non-negative, got {std}.")
  rnd_normal = jax.random.normal(rng, tuple(shape), dtype=dtype)
  return nn.sigmoid(rnd_normal * std + mean)


def time_sampler_mean_flow(
    rng: Array,
    batch_size: int,
    min_train_time: float = 1e-4,
    max_train_time: float = 1.0 - 1e-4,
    diagonal_sampling_rate: float = 0.25,
    time_sampling: Callable[
        [Array, tuple[int, ...]], Array
    ] = functools.partial(jax.random.uniform, dtype=jnp.float32),
) -> tuple[Array, Array]:
  """Samples ordered times `(time_t, time_s)` with `time_t >= time_s`.

  Args:
    rng: JAX PRNG key.
    batch_size: Number of `(t, s)` time pairs to draw.
    min_train_time: Lower bound on sampled training times.
    max_train_time: Upper bound on sampled training times.
    diagonal_sampling_rate: Probability in `[0, 1]` of setting `s = t`
      (instantaneous velocity boundary condition).
    time_sampling: Base `[0, 1]` sampler accepting `(rng, shape)`.

  Returns:
    Tuple `(time_t, time_s)` of 1D arrays of shape `(batch_size,)` satisfying
    `min_train_time <= time_s <= time_t <= max_train_time`.

  Raises:
    ValueError: If `diagonal_sampling_rate` is outside `[0, 1]` or
      `max_train_time < min_train_time`.
  """
  if not 0.0 <= diagonal_sampling_rate <= 1.0:
    raise ValueError(
        "`diagonal_sampling_rate` must be in [0, 1], got "
        f"{diagonal_sampling_rate}."
    )
  if max_train_time < min_train_time:
    raise ValueError(
        f"`max_train_time` ({max_train_time}) must be >= "
        f"`min_train_time` ({min_train_time})."
    )
  diag_rng, rng_t, rng_s = jax.random.split(rng, num=3)
  span = max_train_time - min_train_time
  raw_t = span * time_sampling(rng_t, (batch_size,)) + min_train_time
  raw_s = span * time_sampling(rng_s, (batch_size,)) + min_train_time
  is_diag = jax.random.bernoulli(
      diag_rng, diagonal_sampling_rate, (batch_size,)
  )
  raw_s = jnp.where(is_diag, raw_t, raw_s)
  return jnp.maximum(raw_t, raw_s), jnp.minimum(raw_t, raw_s)


def time_sampler_flow_map(
    rng: Array,
    batch_size: int,
    min_train_time: float = 1e-4,
    max_train_time: float = 1.0 - 1e-4,
    time_sampling: Callable[
        [Array, tuple[int, ...]], Array
    ] = functools.partial(jax.random.uniform, dtype=jnp.float32),
) -> tuple[Array, Array]:
  """Samples off-diagonal ordered times `(time_t, time_s)` for flow maps.

  Args:
    rng: JAX PRNG key.
    batch_size: Number of `(t, s)` time pairs to draw.
    min_train_time: Lower bound on sampled training times.
    max_train_time: Upper bound on sampled training times.
    time_sampling: Base `[0, 1]` sampler accepting `(rng, shape)`.

  Returns:
    Tuple `(time_t, time_s)` of 1D arrays of shape `(batch_size,)` with
    `time_t >= time_s`.
  """
  return time_sampler_mean_flow(
      rng,
      batch_size,
      min_train_time=min_train_time,
      max_train_time=max_train_time,
      diagonal_sampling_rate=0.0,
      time_sampling=time_sampling,
  )


def sample_cfg_scale(
    rng: Array,
    shape: ArrayShape,
    w_min: float = 1.0,
    w_max: float = 8.0,
    beta: float | Array = 1.0,
    dtype: DTypeLike = jnp.float32,
) -> Array:
  """Samples CFG scale `w ~ p(w) propto w**(-beta)` on `[w_min, w_max]`.

  Evaluates the exact inverse CDF in cancellation- and overflow-free log space,
  using a smooth Taylor expansion around `(1 - beta) * log(w_max / w_min) = 0`
  so both values and `jax.grad` with respect to `beta` remain accurate at and
  near `beta = 1`.

  Args:
    rng: JAX PRNG key.
    shape: Output sample shape.
    w_min: Strictly positive lower bound of the guidance scale interval.
    w_max: Upper bound of the guidance scale interval (`w_max >= w_min`).
    beta: Power-law exponent in `p(w) propto w**(-beta)`.
    dtype: Floating-point dtype of the returned array.

  Returns:
    Array of sampled guidance scales in `[w_min, w_max]` with shape
    `tuple(shape)` and dtype `dtype`.

  Raises:
    ValueError: If `w_min <= 0` or `w_max < w_min`.
  """
  if w_min <= 0.0 or w_max < w_min:
    raise ValueError(
        f"Expected `0 < w_min <= w_max`, got w_min={w_min}, w_max={w_max}."
    )
  u = jax.random.uniform(rng, tuple(shape), dtype=jnp.float32)
  w_min_f = jnp.asarray(w_min, dtype=jnp.float32)
  w_max_f = jnp.asarray(w_max, dtype=jnp.float32)
  log_ratio = jnp.log(w_max_f / w_min_f)
  omb = 1.0 - jnp.asarray(beta, dtype=jnp.float32)
  a = omb * log_ratio
  abs_a = jnp.abs(a)
  is_small = abs_a < 1e-3
  safe_omb = jnp.where(is_small, 1.0, omb)
  taylor = (
      log_ratio
      * u
      * (
          1.0
          + 0.5 * (1.0 - u) * a
          + (1.0 / 6.0) * (1.0 - u) * (1.0 - 2.0 * u) * a * a
      )
  )
  mid_num = jnp.log1p(u * jnp.expm1(jnp.clip(a, -0.5, 0.5)))
  exp_neg = jnp.exp(-jnp.minimum(abs_a, 80.0))
  tiny = jnp.finfo(jnp.float32).tiny
  far_pos = abs_a + jnp.log(jnp.maximum((1.0 - u) * exp_neg + u, tiny))
  far_neg = jnp.log(jnp.maximum((1.0 - u) + u * exp_neg, tiny))
  num = jnp.where(abs_a <= 0.5, mid_num, jnp.where(a > 0.0, far_pos, far_neg))
  log_offset = jnp.where(is_small, taylor, num / safe_omb)
  return jnp.clip(w_min_f * jnp.exp(log_offset), w_min_f, w_max_f).astype(dtype)


def sample_cfg_interval(
    rng: Array,
    batch_size: int,
    t_min_range: tuple[float, float] = (0.0, 0.5),
    t_max_range: tuple[float, float] = (0.5, 1.0),
    dtype: DTypeLike = jnp.float32,
) -> tuple[Array, Array]:
  """Samples ordered CFG intervals `(t_min, t_max)` with `t_min <= t_max`.

  Args:
    rng: JAX PRNG key.
    batch_size: Number of guidance intervals to sample.
    t_min_range: `(low, high)` uniform range for the lower interval endpoint.
    t_max_range: `(low, high)` uniform range for the upper interval endpoint.
    dtype: Floating-point dtype of the returned arrays.

  Returns:
    Tuple `(t_min, t_max)` of 1D arrays with shape `(batch_size,)` and
    `t_min <= t_max`.

  Raises:
    ValueError: If either range has `low > high`.
  """
  if t_min_range[0] > t_min_range[1] or t_max_range[0] > t_max_range[1]:
    raise ValueError(
        f"Invalid interval ranges: {t_min_range=}, {t_max_range=}."
    )
  rng_min, rng_max = jax.random.split(rng)
  a = jax.random.uniform(
      rng_min,
      (batch_size,),
      dtype=dtype,
      minval=t_min_range[0],
      maxval=t_min_range[1],
  )
  b = jax.random.uniform(
      rng_max,
      (batch_size,),
      dtype=dtype,
      minval=t_max_range[0],
      maxval=t_max_range[1],
  )
  return jnp.minimum(a, b), jnp.maximum(a, b)


def adaptive_weighted_loss(
    error: Array,
    norm_power: float = 1.0,
    norm_regularization: float = 1e-2,
) -> tuple[Array, Array]:
  """Computes adaptive per-sample weighted MSE and raw MSE in `float32`.

  Evaluates `mean_b[e_b^2 / sg((e_b^2 + c)^p)]` where `e_b^2` is the mean
  squared error over all non-batch axes for sample `b`, `p = norm_power`, and
  `c = norm_regularization`.

  Args:
    error: Residual array of shape `(batch, ...)`.
    norm_power: Exponent `p` in the stop-gradient denominator `(e^2 + c)^p`.
    norm_regularization: Positive stabilizer `c` preventing division by zero.

  Returns:
    Tuple `(adaptive_loss, raw_mse)` of scalar `float32` arrays.

  Raises:
    ValueError: If `norm_regularization <= 0` or `error` is 0D.
  """
  if norm_regularization <= 0.0:
    raise ValueError(
        f"`norm_regularization` must be positive, got {norm_regularization}."
    )
  err_f32 = jnp.asarray(error, dtype=jnp.float32)
  if err_f32.ndim < 1:
    raise ValueError(
        f"`error` must have a leading batch axis, got shape {err_f32.shape}."
    )
  reduce_axes = tuple(range(1, err_f32.ndim))
  per_sample_mse = jnp.mean(jnp.square(err_f32), axis=reduce_axes)
  weight = jnp.power(per_sample_mse + norm_regularization, norm_power)
  loss = jnp.mean(per_sample_mse / jax.lax.stop_gradient(weight))
  return loss, jnp.mean(per_sample_mse)


def compute_compound_velocity(
    u_fn: Callable[[Array, Array, Array], tuple[Array, Any]],
    x_t: Array,
    time_t: Array,
    time_s: Array,
    tangent_v: Array,
) -> tuple[Array, Array, Array, Any]:
  """Computes `V = u(x_t, t, s) + (t - s) * sg(du/dt)` via `jax.jvp`.

  Applies forward-mode differentiation along tangent `(sg(tangent_v), 1, 0)`
  so that `du/dt = partial_x u * sg(tangent_v) + partial_t u`.

  Args:
    u_fn: Callable `(x_t, time_t, time_s) -> (u, aux)` returning the average
      velocity field `u` (shape `(batch, ...)`) and auxiliary data.
    x_t: Interpolant state of shape `(batch, ...)`.
    time_t: Upper time coordinate of shape `(batch,)`.
    time_s: Lower time coordinate of shape `(batch,)`, satisfying `t >= s`.
    tangent_v: Velocity tangent of shape `(batch, ...)` used in the JVP.

  Returns:
    Tuple `(compound_v, u_ts, du_dt, aux)` where `compound_v`, `u_ts`, and
    `du_dt` match the shape and dtype of `u_fn`'s primary output.
  """
  v_sg = jax.lax.stop_gradient(tangent_v).astype(x_t.dtype)
  u_ts, du_dt, aux = jax.jvp(
      u_fn,
      (x_t, time_t, time_s),
      (v_sg, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
      has_aux=True,
  )
  dt = jnp.maximum(time_t - time_s, 0.0).reshape(
      (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
  )
  dt = dt.astype(u_ts.dtype)
  compound_v = u_ts + dt * jax.lax.stop_gradient(du_dt)
  return compound_v, u_ts, du_dt, aux


def compute_guided_velocity_target(
    v_base: Array,
    v_cond: Array,
    v_uncond: Array,
    cfg_scale: Array,
    time_t: Array,
    t_min: Array,
    t_max: Array,
    is_dropped: Array,
) -> tuple[Array, Array]:
  """Computes stop-gradient Eq. 17 guided target and effective scale `w_eff`.

  Evaluates `v_tgt = sg(v_base + (1 - 1 / w_eff) * (v_cond - v_uncond))` where
  `w_eff = cfg_scale` when `(~is_dropped) & (t_min <= time_t <= t_max)` and
  `1.0` otherwise.

  Args:
    v_base: Base velocity target of shape `(batch, ...)`.
    v_cond: Conditioned marginal velocity prediction of shape `(batch, ...)`.
    v_uncond: Unconditional (null-class) velocity prediction of shape `(batch,
      ...)`.
    cfg_scale: Sampled guidance scale `omega` of shape `(batch,)`.
    time_t: Current time coordinate of shape `(batch,)`.
    t_min: Lower CFG interval bound of shape `(batch,)`.
    t_max: Upper CFG interval bound of shape `(batch,)`.
    is_dropped: Boolean mask of shape `(batch,)` indicating class dropout.

  Returns:
    Tuple `(v_target, omega_eff)` where `v_target` is stop-gradiented with the
    shape and dtype of `v_base`, and `omega_eff` has shape `(batch,)`.
  """
  in_interval = (time_t >= t_min) & (time_t <= t_max)
  active = (~is_dropped) & in_interval
  omega_eff = jnp.where(active, cfg_scale, jnp.ones_like(cfg_scale))
  safe_omega = jnp.maximum(omega_eff.astype(jnp.float32), 1e-6)
  coeff = (1.0 - 1.0 / safe_omega).reshape(
      (v_base.shape[0],) + (1,) * (v_base.ndim - 1)
  )
  v_tgt = (
      jnp.asarray(v_base, dtype=jnp.float32)
      + coeff
      * (
          jnp.asarray(v_cond, dtype=jnp.float32)
          - jnp.asarray(v_uncond, dtype=jnp.float32)
      )
  ).astype(v_base.dtype)
  return jax.lax.stop_gradient(v_tgt), omega_eff


def sample_forward_flow_map(
    flow_map_fn: Callable[..., Array],
    x_0: Array,
    num_steps: int = 1,
    cond: CondDict | None = None,
    time_grid: Array | None = None,
) -> Array:
  """Composes a forward flow map `X_{s, t}` from `t = 0` to `t = 1`.

  Args:
    flow_map_fn: Callable `(x, t, s)` (or `(x, t, s, cond)` when `cond` is
      provided) returning `X_{s, t}(x)`.
    x_0: Initial state of shape `(batch, ...)`.
    num_steps: Number of uniform steps when `time_grid` is `None`.
    cond: Optional conditioning dictionary forwarded to `flow_map_fn`.
    time_grid: Optional 1D ascending time grid of length `>= 2`.

  Returns:
    Transported state `x_1` with the same shape and dtype as `x_0`.

  Raises:
    ValueError: If `num_steps < 1` or `time_grid` is not 1D of length `>= 2`.
  """
  grid = _resolve_time_grid(num_steps, time_grid, descending=False)

  def _step(i: int, x: Array) -> Array:
    s_i = jnp.full((x.shape[0],), grid[i], dtype=jnp.float32)
    t_i = jnp.full((x.shape[0],), grid[i + 1], dtype=jnp.float32)
    if cond is None:
      return flow_map_fn(x, t_i, s_i)
    return flow_map_fn(x, t_i, s_i, cond)

  return jax.lax.fori_loop(0, grid.shape[0] - 1, _step, x_0)


def sample_backward_mean_flow(
    mean_flow_fn: Callable[..., Array],
    x_1: Array,
    num_steps: int = 1,
    cond: CondDict | None = None,
    time_grid: Array | None = None,
) -> Array:
  """Integrates `x_s = x_t - (t - s) * u(x_t, t, s)` from `t = 1` to `t = 0`.

  Args:
    mean_flow_fn: Callable `(x, t, s)` (or `(x, t, s, cond)` when `cond` is
      provided) returning average velocity `u(x, t, s)`.
    x_1: Prior state of shape `(batch, ...)`.
    num_steps: Number of uniform steps when `time_grid` is `None`.
    cond: Optional conditioning dictionary forwarded to `mean_flow_fn`.
    time_grid: Optional 1D descending time grid of length `>= 2`.

  Returns:
    Generated state `x_0` with the same shape and dtype as `x_1`.

  Raises:
    ValueError: If `num_steps < 1` or `time_grid` is not 1D of length `>= 2`.
  """
  grid = _resolve_time_grid(num_steps, time_grid, descending=True)

  def _step(i: int, x: Array) -> Array:
    t_i = jnp.full((x.shape[0],), grid[i], dtype=jnp.float32)
    s_i = jnp.full((x.shape[0],), grid[i + 1], dtype=jnp.float32)
    u = (
        mean_flow_fn(x, t_i, s_i)
        if cond is None
        else mean_flow_fn(x, t_i, s_i, cond)
    )
    dt = (grid[i] - grid[i + 1]).astype(x.dtype)
    return x - dt * u

  return jax.lax.fori_loop(0, grid.shape[0] - 1, _step, x_1)


def _resolve_time_grid(
    num_steps: int, time_grid: Array | None, *, descending: bool
) -> Array:
  """Validates shape and builds a 1D time grid of length `>= 2`."""
  if time_grid is not None:
    grid = jnp.asarray(time_grid, dtype=jnp.float32)
    if grid.ndim != 1 or grid.shape[0] < 2:
      raise ValueError(
          f"`time_grid` must be 1D with length >= 2, got shape {grid.shape}."
      )
    return grid
  if num_steps < 1:
    raise ValueError(f"`num_steps` must be >= 1, got {num_steps}.")
  start, stop = (1.0, 0.0) if descending else (0.0, 1.0)
  return jnp.linspace(start, stop, num_steps + 1, dtype=jnp.float32)
