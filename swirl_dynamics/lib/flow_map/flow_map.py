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

r"""Lagrangian and self-distilled flow-map models.

Starting from a stochastic interpolant [1]:
  X_t = \alpha(t) * X_0 + \beta(t) * X_1 + \gamma(t) * Z,
flow-map matching [2] learns the two-time pushforward X_{s, t}(X_s) = X_t
(s <= t) via teacher velocity distillation (`LagrangianFlowMapModel`) or
teacher-free self-distillation [3] (`LagrangianSelfDistilledFlowMapModel`).

References:
  [1] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.
      Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. 2023.
  [2] Flow Map Matching with Stochastic Interpolants: A Mathematical Framework
      for Consistency Models. Nicholas M. Boffi, Michael S. Albergo, and
      Eric Vanden-Eijnden. 2024.
  [3] How to Build a Consistency Model: Learning Flow Maps via
      Self-Distillation. Nicholas M. Boffi, Michael S. Albergo, and
      Eric Vanden-Eijnden. 2025.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.diffusion import denoising_models
from swirl_dynamics.lib.flow_map import samplers
from swirl_dynamics.lib.flow_matching import interpolants
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers

Array = jax.Array
ArrayShape = Sequence[int]
CondDict = samplers.CondDict
ShapeDict = denoising_models.ShapeDict
VariableDict = trainers.VariableDict

cond_sample_from_shape = denoising_models.cond_sample_from_shape


def _resolve_teacher_vars(
    params_flow: VariableDict | models.PyTree,
) -> Mapping[str, Any]:
  """Normalizes either raw teacher params or a full Flax variable dict."""
  if isinstance(params_flow, Mapping) and "params" in params_flow:
    return params_flow
  return {"params": params_flow}


@dataclasses.dataclass(frozen=True, kw_only=True)
class LagrangianFlowMapModel(models.BaseModel):
  """Lagrangian flow-map distillation model trained against a teacher flow.

  Minimizes `E[|d/dt X_{s, t}(x_s) - sg(v_phi(X_{s, t}(x_s), t))|]` for
  ordered times `t >= s`.

  Attributes:
    input_shape: Tensor shape of a single sample (excluding batch dimension).
    flow_map_model: Flax module representing `X_{s, t}(x_s)`.
    flow_model: Frozen teacher velocity module `v_phi(x, sigma)`.
    params_flow: Parameters or full Flax variable dictionary of `flow_model`.
    interpolant: Stochastic interpolant linking `x_0` and `x_1`.
    noising_process: Callable sampling latent noise matching `x_0.shape`.
    time_sampling: Callable `(rng, batch_size, min_t, max_t) -> (t, s)`.
    min_train_time: Lower bound on sampled training times.
    max_train_time: Upper bound on sampled training times.
    number_of_eval_steps: Step counts evaluated via forward map composition.
  """

  input_shape: tuple[int, ...]
  flow_map_model: nn.Module
  flow_model: nn.Module
  params_flow: VariableDict | models.PyTree
  interpolant: interpolants.StochasticInterpolant
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  time_sampling: Callable[[Array, int, float, float], tuple[Array, Array]] = (
      samplers.time_sampler_flow_map
  )
  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4
  number_of_eval_steps: tuple[int, ...] = (1, 4, 8)

  def initialize(self, rng: Array) -> Mapping[str, Any]:
    """Initializes variables of `flow_map_model`."""
    x = jnp.ones((1,) + self.input_shape)
    return self.flow_map_model.init(
        rng,
        x_s=x,
        t=jnp.ones((1,)),
        s=jnp.ones((1,)),
        is_training=False,
    )

  def _extract_cond(self, batch: models.BatchType) -> CondDict | None:
    """Extracts optional conditioning inputs from `batch`."""
    del batch
    return None

  def _apply_teacher_flow(
      self, x_t: Array, time: Array, cond: CondDict | None
  ) -> Array:
    """Evaluates the frozen teacher `flow_model` at `(x_t, time)`."""
    del cond
    teacher_vars = _resolve_teacher_vars(self.params_flow)
    return cast(
        Array,
        self.flow_model.apply(
            teacher_vars, x=x_t, sigma=time, is_training=False
        ),
    )

  def _apply_flow_map(
      self,
      variables: models.PyTree,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None,
      *,
      is_training: bool,
      rngs: dict[str, Array] | None = None,
      mutable: Any = False,
  ) -> Any:
    """Applies `flow_map_model` with or without conditioning arguments."""
    del cond
    return self.flow_map_model.apply(
        variables,
        x_s,
        t,
        s,
        is_training=is_training,
        rngs=rngs,
        mutable=mutable,
    )

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the Lagrangian flow-map loss and updated mutables.

    Args:
      params: Differentiable parameters of `flow_map_model`.
      batch: Batch dictionary containing `x_0` and `x_1` arrays of shape
        `(batch, *spatial_dims, channels)`, plus any conditioning fields.
      rng: JAX PRNG key.
      mutables: Mutable non-differentiated state (e.g. `batch_stats`).

    Returns:
      Tuple `(loss, (metrics, updated_mutables))`.
    """
    batch_size = len(batch["x_0"])
    time_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_rng, batch_size, self.min_train_time, self.max_train_time
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)
    cond = self._extract_cond(batch)

    mutable_keys = list(mutables.keys())
    variables = {"params": params, **mutables}

    def _partial_flow_map(
        x_in: Array, t_in: Array, s_in: Array
    ) -> tuple[Array, Any]:
      out, updated_mut = self._apply_flow_map(
          variables,
          x_in,
          t_in,
          s_in,
          cond,
          is_training=True,
          rngs={"dropout": dropout_rng},
          mutable=mutable_keys,
      )
      return cast(Array, out), jax.lax.stop_gradient(updated_mut)

    x_st, dt_x_st, updated_mutables = jax.jvp(
        _partial_flow_map,
        (x_s, time_t, time_s),
        (jnp.zeros_like(x_s), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
        has_aux=True,
    )
    v_t = self._apply_teacher_flow(x_st, time_t, cond)
    loss = jnp.mean(jnp.abs(dt_x_st - jax.lax.stop_gradient(v_t)))
    return loss, ({"loss": loss}, updated_mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes Lagrangian eval loss and multi-step forward flow samples."""
    batch_size = len(batch["x_0"])
    time_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_rng, batch_size, self.min_train_time, self.max_train_time
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)
    cond = self._extract_cond(batch)

    def _partial_flow_map(x_in: Array, t_in: Array, s_in: Array) -> Array:
      return cast(
          Array,
          self._apply_flow_map(
              variables,
              x_in,
              t_in,
              s_in,
              cond,
              is_training=False,
              rngs={"dropout": dropout_rng},
          ),
      )

    x_st, dt_x_st = jax.jvp(
        _partial_flow_map,
        (x_s, time_t, time_s),
        (jnp.zeros_like(x_s), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )
    v_t = self._apply_teacher_flow(x_st, time_t, cond)
    loss = jnp.mean(jnp.abs(dt_x_st - v_t))

    eval_samples = {}
    for num_steps in self.number_of_eval_steps:
      samples = samplers.sample_forward_flow_map(
          _partial_flow_map, batch["x_0"], num_steps=num_steps
      )
      eval_samples[f"{num_steps}"] = jnp.asarray(samples)

    return {"eval_loss": loss, **eval_samples}

  @staticmethod
  def inference_fn(
      variables: models.PyTree, flow_map_model: nn.Module
  ) -> Callable[..., Array]:
    """Returns a bound forward flow-map inference function."""

    def _flow(
        x: Array,
        t: float | Array,
        s: float | Array,
        cond: CondDict | None = None,
    ) -> Array:
      t_arr = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float32), (x.shape[0],))
      s_arr = jnp.broadcast_to(jnp.asarray(s, dtype=jnp.float32), (x.shape[0],))
      if cond is None:
        return cast(
            Array,
            flow_map_model.apply(
                variables, x_s=x, t=t_arr, s=s_arr, is_training=False
            ),
        )
      return cast(
          Array,
          flow_map_model.apply(
              variables, x_s=x, t=t_arr, s=s_arr, cond=cond, is_training=False
          ),
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalLagrangianFlowMapModel(LagrangianFlowMapModel):
  """Conditional Lagrangian flow-map distillation model.

  Attributes:
    cond_shape: Mapping from conditioning key names to per-sample tensor shapes.
    number_of_eval_steps: Step counts evaluated via forward map composition.
  """

  cond_shape: ShapeDict | None = None
  number_of_eval_steps: tuple[int, ...] = (1,)

  def initialize(self, rng: Array) -> Mapping[str, Any]:
    """Initializes `flow_map_model` with dummy conditional inputs."""
    x = jnp.ones((1,) + self.input_shape)
    cond = cond_sample_from_shape(self.cond_shape, batch_dims=(1,))
    return self.flow_map_model.init(
        rng,
        x_s=x,
        t=jnp.ones((1,)),
        s=jnp.ones((1,)),
        cond=cond,
        is_training=False,
    )

  def _extract_cond(self, batch: models.BatchType) -> CondDict | None:
    """Extracts conditional entries specified in `self.cond_shape`."""
    if self.cond_shape is None:
      return None
    return {key: batch[key] for key in self.cond_shape.keys()}

  def _apply_teacher_flow(
      self, x_t: Array, time: Array, cond: CondDict | None
  ) -> Array:
    """Evaluates the conditional teacher `flow_model`."""
    teacher_vars = _resolve_teacher_vars(self.params_flow)
    return cast(
        Array,
        self.flow_model.apply(
            teacher_vars, x=x_t, sigma=time, cond=cond, is_training=False
        ),
    )

  def _apply_flow_map(
      self,
      variables: models.PyTree,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None,
      *,
      is_training: bool,
      rngs: dict[str, Array] | None = None,
      mutable: Any = False,
  ) -> Any:
    """Applies `flow_map_model` with conditional inputs."""
    return self.flow_map_model.apply(
        variables,
        x_s,
        t,
        s,
        cond=cond,
        is_training=is_training,
        rngs=rngs,
        mutable=mutable,
    )

  @staticmethod
  def inference_fn(
      variables: models.PyTree, flow_map_model: nn.Module
  ) -> Callable[..., Array]:
    """Returns a bound conditional forward flow-map inference function."""

    def _flow(
        x: Array,
        t: float | Array,
        s: float | Array,
        cond: CondDict | None = None,
    ) -> Array:
      t_arr = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float32), (x.shape[0],))
      s_arr = jnp.broadcast_to(jnp.asarray(s, dtype=jnp.float32), (x.shape[0],))
      return cast(
          Array,
          flow_map_model.apply(
              variables, x_s=x, t=t_arr, s=s_arr, cond=cond, is_training=False
          ),
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class LagrangianSelfDistilledFlowMapModel(models.BaseModel):
  """Teacher-free Lagrangian self-distilled flow-map model [3].

  Parameterizes the forward flow map as `X_{s, t}(x_s) = x_s + (t - s) *
  u_theta(x_s, t, s)` and minimizes a convex combination (`weight_v`) of the
  instantaneous velocity matching loss at `s = t` and the Lagrangian
  self-consistency loss `|d/dt X_{s, t}(x_s) - u_theta(X_{s, t}(x_s), t, t)|^2`.

  Attributes:
    input_shape: Tensor shape of a single sample (excluding batch dimension).
    mean_flow_model: Flax module representing `u_theta(x_s, t, s)`.
    interpolant: Stochastic interpolant linking `x_0` and `x_1`.
    noising_process: Callable sampling latent noise matching `x_0.shape`.
    time_sampling: Callable `(rng, batch_size, min_t, max_t) -> (t, s)`.
    min_train_time: Lower bound on sampled training times.
    max_train_time: Upper bound on sampled training times.
    number_of_eval_steps: Step counts evaluated via forward map composition.
    weight_v: Weight in `[0, 1]` on the velocity loss (`1 - weight_v` on map).
    use_adaptive_norm: Whether to apply adaptive loss weighting.
    norm_power: Exponent `p` for adaptive loss weighting.
    norm_regularization: Positive offset `c` in adaptive loss weighting.
    stop_gradient_target: Opt-in stop-gradient on the self-distillation target
      `u_theta(X_{s, t}(x_s), t, t)` in `loss_x`.
    per_sample_adaptive_norm: When `True` and `use_adaptive_norm=True`, applies
      `samplers.adaptive_weighted_loss` per sample instead of batch-level scalar
      normalization.
  """

  input_shape: tuple[int, ...]
  mean_flow_model: nn.Module
  interpolant: interpolants.StochasticInterpolant
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  time_sampling: Callable[[Array, int, float, float], tuple[Array, Array]] = (
      samplers.time_sampler_flow_map
  )
  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4
  number_of_eval_steps: tuple[int, ...] = (1,)
  weight_v: float = 0.5
  use_adaptive_norm: bool = False
  norm_power: float = 1.0
  norm_regularization: float = 1e-2
  stop_gradient_target: bool = False
  per_sample_adaptive_norm: bool = False

  def __post_init__(self) -> None:
    if self.weight_v < 0.0 or self.weight_v > 1.0:
      raise ValueError(
          "The weight of the v-term in the Lagrangian self-distillation loss"
          f" should be between 0 and 1, got {self.weight_v}."
      )

  def initialize(self, rng: Array) -> Mapping[str, Any]:
    """Initializes variables of `mean_flow_model`."""
    x = jnp.ones((1,) + self.input_shape)
    return self.mean_flow_model.init(
        rng,
        x_s=x,
        t=jnp.ones((1,)),
        s=jnp.ones((1,)),
        is_training=False,
    )

  def _extract_cond(self, batch: models.BatchType) -> CondDict | None:
    """Extracts optional conditioning inputs from `batch`."""
    del batch
    return None

  def _apply_mean_flow(
      self,
      variables: models.PyTree,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None,
      *,
      is_training: bool,
      rngs: dict[str, Array] | None = None,
      mutable: Any = False,
  ) -> Any:
    """Applies `mean_flow_model` with or without conditioning arguments."""
    del cond
    return self.mean_flow_model.apply(
        variables,
        x_s,
        t,
        s,
        is_training=is_training,
        rngs=rngs,
        mutable=mutable,
    )

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes Lagrangian self-distillation loss and updated mutables."""
    batch_size = len(batch["x_0"])
    time_rng, noise_rng, dropout_v_rng, dropout_x_rng, dropout_tgt_rng = (
        jax.random.split(rng, num=5)
    )
    time_t, time_s = self.time_sampling(
        time_rng, batch_size, self.min_train_time, self.max_train_time
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)
    cond = self._extract_cond(batch)

    mutable_keys = list(mutables.keys())
    updated_mutables = mutables

    if self.weight_v > 0.0:
      dinterp_dt = self.interpolant.calculate_time_derivative_interpolant(
          time_t, batch["x_0"], batch["x_1"], noise
      )
      v_tt, mut_v = self._apply_mean_flow(
          {"params": params, **updated_mutables},
          x_t,
          time_t,
          time_t,
          cond,
          is_training=True,
          rngs={"dropout": dropout_v_rng},
          mutable=mutable_keys,
      )
      updated_mutables = jax.lax.stop_gradient(mut_v)
      err_v = cast(Array, v_tt) - dinterp_dt
      if self.use_adaptive_norm and self.per_sample_adaptive_norm:
        loss_v, _ = samplers.adaptive_weighted_loss(
            err_v,
            norm_power=self.norm_power,
            norm_regularization=self.norm_regularization,
        )
      else:
        loss_v = jnp.mean(jnp.square(err_v))
    else:
      loss_v = jnp.zeros((), dtype=x_t.dtype)

    if self.weight_v < 1.0:
      variables_x = {"params": params, **updated_mutables}

      def _partial_flow_map(
          x_in: Array, t_in: Array, s_in: Array
      ) -> tuple[Array, Any]:
        delta_time = (t_in - s_in).reshape(
            (x_in.shape[0],) + (x_in.ndim - 1) * (1,)
        )
        u_st, mut_x = self._apply_mean_flow(
            variables_x,
            x_in,
            t_in,
            s_in,
            cond,
            is_training=True,
            rngs={"dropout": dropout_x_rng},
            mutable=mutable_keys,
        )
        return (
            x_in + delta_time.astype(x_in.dtype) * cast(Array, u_st),
            jax.lax.stop_gradient(mut_x),
        )

      x_st, dt_x_st, mut_x = jax.jvp(
          _partial_flow_map,
          (x_s, time_t, time_s),
          (jnp.zeros_like(x_s), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
          has_aux=True,
      )
      updated_mutables = mut_x
      v_tt_st, mut_tgt = self._apply_mean_flow(
          {"params": params, **updated_mutables},
          x_st,
          time_t,
          time_t,
          cond,
          is_training=True,
          rngs={"dropout": dropout_tgt_rng},
          mutable=mutable_keys,
      )
      updated_mutables = jax.lax.stop_gradient(mut_tgt)
      target = (
          jax.lax.stop_gradient(cast(Array, v_tt_st))
          if self.stop_gradient_target
          else cast(Array, v_tt_st)
      )
      err_x = dt_x_st - target
      if self.use_adaptive_norm and self.per_sample_adaptive_norm:
        loss_x, _ = samplers.adaptive_weighted_loss(
            err_x,
            norm_power=self.norm_power,
            norm_regularization=self.norm_regularization,
        )
      else:
        loss_x = jnp.mean(jnp.square(err_x))
    else:
      loss_x = jnp.zeros((), dtype=x_t.dtype)

    if self.use_adaptive_norm and not self.per_sample_adaptive_norm:
      adaptive_weight_x = (loss_x + self.norm_regularization) ** self.norm_power
      loss_x = loss_x / jax.lax.stop_gradient(adaptive_weight_x)
      adaptive_weight_v = (loss_v + self.norm_regularization) ** self.norm_power
      loss_v = loss_v / jax.lax.stop_gradient(adaptive_weight_v)

    loss = self.weight_v * loss_v + (1.0 - self.weight_v) * loss_x
    metric = {"loss": loss, "loss_v": loss_v, "loss_x": loss_x}
    return loss, (metric, updated_mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes LSD eval losses and multi-step forward flow samples."""
    batch_size = len(batch["x_0"])
    time_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_rng, batch_size, self.min_train_time, self.max_train_time
    )

    cond = self._extract_cond(batch)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)

    def _partial_flow_map(x_in: Array, t_in: Array, s_in: Array) -> Array:
      delta_time = (t_in - s_in).reshape(
          (x_in.shape[0],) + (x_in.ndim - 1) * (1,)
      )
      u_st = cast(
          Array,
          self._apply_mean_flow(
              variables,
              x_in,
              t_in,
              s_in,
              cond,
              is_training=False,
              rngs={"dropout": dropout_rng},
          ),
      )
      return x_in + delta_time.astype(x_in.dtype) * u_st

    x_st, dt_x_st = jax.jvp(
        _partial_flow_map,
        (x_s, time_t, time_s),
        (jnp.zeros_like(x_s), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )
    dinterp_dt = self.interpolant.calculate_time_derivative_interpolant(
        time_t, batch["x_0"], batch["x_1"], noise
    )
    v_tt_st = cast(
        Array,
        self._apply_mean_flow(
            variables,
            x_st,
            time_t,
            time_t,
            cond,
            is_training=False,
            rngs={"dropout": dropout_rng},
        ),
    )
    v_tt = cast(
        Array,
        self._apply_mean_flow(
            variables,
            x_t,
            time_t,
            time_t,
            cond,
            is_training=False,
            rngs={"dropout": dropout_rng},
        ),
    )

    loss_x = jnp.mean(jnp.square(dt_x_st - v_tt_st))
    loss_v = jnp.mean(jnp.square(v_tt - dinterp_dt))
    loss = self.weight_v * loss_v + (1.0 - self.weight_v) * loss_x

    eval_samples = {}
    for num_steps in self.number_of_eval_steps:
      samples = samplers.sample_forward_flow_map(
          _partial_flow_map, batch["x_0"], num_steps=num_steps
      )
      eval_samples[f"{num_steps}"] = jnp.asarray(samples)

    return {
        "eval_loss": loss,
        "eval_loss_v": loss_v,
        "eval_loss_x": loss_x,
        **eval_samples,
    }

  @staticmethod
  def inference_fn(
      variables: models.PyTree, mean_flow_model: nn.Module
  ) -> Callable[..., Array]:
    """Returns forward LSD flow map `x + (t - s) * u_theta(x, t, s[, cond])`."""

    def _flow(
        x: Array,
        t: float | Array,
        s: float | Array,
        cond: CondDict | None = None,
    ) -> Array:
      t_arr = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float32), (x.shape[0],))
      s_arr = jnp.broadcast_to(jnp.asarray(s, dtype=jnp.float32), (x.shape[0],))
      delta_time = (t_arr - s_arr).reshape((x.shape[0],) + (x.ndim - 1) * (1,))
      if cond is None:
        u_st = cast(
            Array,
            mean_flow_model.apply(
                variables, x_s=x, t=t_arr, s=s_arr, is_training=False
            ),
        )
      else:
        u_st = cast(
            Array,
            mean_flow_model.apply(
                variables,
                x_s=x,
                t=t_arr,
                s=s_arr,
                cond=cond,
                is_training=False,
            ),
        )
      return x + delta_time.astype(x.dtype) * u_st

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalLagrangianSelfDistilledFlowMapModel(
    LagrangianSelfDistilledFlowMapModel
):
  """Conditional teacher-free Lagrangian self-distilled flow-map model.

  Attributes:
    cond_shape: Mapping from conditioning key names to per-sample tensor shapes.
  """

  cond_shape: ShapeDict | None = None

  def initialize(self, rng: Array) -> Mapping[str, Any]:
    """Initializes `mean_flow_model` with dummy conditional inputs."""
    x = jnp.ones((1,) + self.input_shape)
    cond = cond_sample_from_shape(self.cond_shape, batch_dims=(1,))
    return self.mean_flow_model.init(
        rng,
        x_s=x,
        t=jnp.ones((1,)),
        s=jnp.ones((1,)),
        cond=cond,
        is_training=False,
    )

  def _extract_cond(self, batch: models.BatchType) -> CondDict | None:
    """Extracts conditional entries specified in `self.cond_shape`."""
    if self.cond_shape is None:
      return None
    return {key: batch[key] for key in self.cond_shape.keys()}

  def _apply_mean_flow(
      self,
      variables: models.PyTree,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None,
      *,
      is_training: bool,
      rngs: dict[str, Array] | None = None,
      mutable: Any = False,
  ) -> Any:
    """Applies `mean_flow_model` with conditional inputs."""
    return self.mean_flow_model.apply(
        variables,
        x_s,
        t,
        s,
        cond=cond,
        is_training=is_training,
        rngs=rngs,
        mutable=mutable,
    )

  @staticmethod
  def inference_fn(
      variables: models.PyTree, mean_flow_model: nn.Module
  ) -> Callable[..., Array]:
    """Returns conditional forward LSD map `x + (t - s) * u_theta`."""

    def _flow(
        x: Array,
        t: float | Array,
        s: float | Array,
        cond: CondDict | None = None,
    ) -> Array:
      t_arr = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float32), (x.shape[0],))
      s_arr = jnp.broadcast_to(jnp.asarray(s, dtype=jnp.float32), (x.shape[0],))
      delta_time = (t_arr - s_arr).reshape((x.shape[0],) + (x.ndim - 1) * (1,))
      u_st = cast(
          Array,
          mean_flow_model.apply(
              variables, x_s=x, t=t_arr, s=s_arr, cond=cond, is_training=False
          ),
      )
      return x + delta_time.astype(x.dtype) * u_st

    return _flow
