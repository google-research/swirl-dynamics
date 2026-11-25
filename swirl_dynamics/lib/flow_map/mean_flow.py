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

r"""Mean-flow models for one-step generative modeling and distillation.

References:
  [1] Mean Flows for One-step Generative Modeling. Zhengyang Geng, Mingyang
      Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. 2025.
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
class MeanFlowModel(models.BaseModel):
  """Trains a mean-flow model for distribution matching or distillation [1].

  Attributes:
    input_shape: Tensor shape of a single sample (excluding batch dimension).
    mean_flow_model: Flax module parameterizing average velocity `u(x_t, t, s)`.
    flow_model: Optional frozen teacher velocity module `v_phi(x, sigma)`. When
      `None`, the reference velocity is computed from `interpolant`.
    params_flow: Parameters or full Flax variable dictionary of `flow_model`
      when distilling a teacher velocity.
    interpolant: Stochastic interpolant linking `x_0` and `x_1`.
    noising_process: Callable sampling latent noise matching `x_0.shape`.
    time_sampling: Callable sampling ordered `(t, s)` time pairs.
    min_train_time: Lower bound on sampled training times.
    max_train_time: Upper bound on sampled training times.
    max_gap: Maximum time gap setting retained for config compatibility.
    number_of_eval_steps: Step counts used by downstream evaluation workflows.
    norm_power: Exponent `p` for adaptive per-sample loss reweighting.
    norm_regularization: Positive stabilizer `c` in adaptive loss reweighting.
    diagonal_sampling_rate: Probability of sampling `s = t` during training.
  """

  input_shape: tuple[int, ...]
  mean_flow_model: nn.Module
  flow_model: nn.Module | None = None
  params_flow: VariableDict | models.PyTree | None = None
  interpolant: interpolants.StochasticInterpolant = interpolants.RectifiedFlow()
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  time_sampling: Callable[..., tuple[Array, Array]] = (
      samplers.time_sampler_mean_flow
  )
  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4
  max_gap: float = 1.0
  number_of_eval_steps: tuple[int, ...] = (1, 4, 8)
  norm_power: float = 1.0
  norm_regularization: float = 1e-2
  diagonal_sampling_rate: float = 0.25

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

  def _apply_teacher_flow(
      self,
      flow_model: nn.Module,
      params_flow: VariableDict | models.PyTree,
      x_t: Array,
      time: Array,
      cond: CondDict | None,
  ) -> Array:
    """Evaluates the teacher `flow_model` at `(x_t, time)`."""
    del cond
    teacher_vars = _resolve_teacher_vars(params_flow)
    return cast(
        Array,
        flow_model.apply(teacher_vars, x=x_t, sigma=time, is_training=False),
    )

  def _compute_reference_velocity(
      self,
      time: Array,
      x_t: Array,
      batch: models.BatchType,
      noise: Array,
      cond: CondDict | None,
  ) -> Array:
    """Computes target velocity from teacher `flow_model` or `interpolant`."""
    if self.flow_model is not None:
      if self.params_flow is None:
        raise ValueError(
            "`params_flow` must be provided when `flow_model` is set."
        )
      return self._apply_teacher_flow(
          self.flow_model, self.params_flow, x_t, time, cond
      )
    return self.interpolant.calculate_time_derivative_interpolant(
        time, batch["x_0"], batch["x_1"], noise
    )

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
    """Applies `mean_flow_model` with or without conditioning kwargs."""
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
    """Computes the mean-flow training loss and updated mutables."""
    batch_size = len(batch["x_0"])
    time_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_rng,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    cond = self._extract_cond(batch)
    v_t = self._compute_reference_velocity(time_t, x_t, batch, noise, cond)

    mutable_keys = list(mutables.keys())
    variables = {"params": params, **mutables}

    def _partial_flow_map(
        x_in: Array, t_in: Array, s_in: Array
    ) -> tuple[Array, Any]:
      out, updated_mut = self._apply_mean_flow(
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

    v_tangent = jax.lax.stop_gradient(v_t).astype(x_t.dtype)
    u_ts, dt_u_st, updated_mutables = jax.jvp(
        _partial_flow_map,
        (x_t, time_t, time_s),
        (v_tangent, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
        has_aux=True,
    )

    dt = jnp.maximum(time_t - time_s, 0.0).reshape(
        (x_t.shape[0],) + (x_t.ndim - 1) * (1,)
    )
    u_target = v_t - dt.astype(u_ts.dtype) * dt_u_st
    loss, raw_mse = samplers.adaptive_weighted_loss(
        u_ts - jax.lax.stop_gradient(u_target),
        norm_power=self.norm_power,
        norm_regularization=self.norm_regularization,
    )
    metric = {"loss": loss, "raw_mse": raw_mse}
    return loss, (metric, updated_mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes `L^1` mean-flow error and one-step samples on an eval batch."""
    batch_size = len(batch["x_0"])
    time_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_rng,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    cond = self._extract_cond(batch)
    v_t = self._compute_reference_velocity(time_t, x_t, batch, noise, cond)

    def _partial_flow_map(x_in: Array, t_in: Array, s_in: Array) -> Array:
      return cast(
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

    u_ts, dt_u_st = jax.jvp(
        _partial_flow_map,
        (x_t, time_t, time_s),
        (v_t.astype(x_t.dtype), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    dt = jnp.maximum(time_t - time_s, 0.0).reshape(
        (x_t.shape[0],) + (x_t.ndim - 1) * (1,)
    )
    u_target = v_t - dt.astype(u_ts.dtype) * dt_u_st
    loss = jnp.mean(jnp.abs(u_ts - jax.lax.stop_gradient(u_target)))
    eval_samples = samplers.sample_backward_mean_flow(
        _partial_flow_map, batch["x_1"], num_steps=1
    )
    return {"eval_loss": loss, "eval_samples": eval_samples}

  @staticmethod
  def inference_fn(
      variables: models.PyTree, mean_flow_model: nn.Module
  ) -> Callable[..., Array]:
    """Returns bound mean-flow inference function accepting optional `cond`."""

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
            mean_flow_model.apply(
                variables, x_s=x, t=t_arr, s=s_arr, is_training=False
            ),
        )
      return cast(
          Array,
          mean_flow_model.apply(
              variables, x_s=x, t=t_arr, s=s_arr, cond=cond, is_training=False
          ),
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalMeanFlowModel(MeanFlowModel):
  """Trains a conditional mean-flow model for distribution matching.

  Attributes:
    cond_shape: Mapping of conditional input keys to sample shapes (excluding
      batch dimension). If `None`, no conditioning is extracted.
  """

  cond_shape: ShapeDict | None = None
  number_of_eval_steps: tuple[int, ...] = (1,)

  def initialize(self, rng: Array) -> Mapping[str, Any]:
    """Initializes variables of `mean_flow_model` with dummy conditioning."""
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
    if self.cond_shape is None:
      return None
    return {key: batch[key] for key in self.cond_shape.keys()}

  def _apply_teacher_flow(
      self,
      flow_model: nn.Module,
      params_flow: VariableDict | models.PyTree,
      x_t: Array,
      time: Array,
      cond: CondDict | None,
  ) -> Array:
    teacher_vars = _resolve_teacher_vars(params_flow)
    return cast(
        Array,
        flow_model.apply(
            teacher_vars, x=x_t, sigma=time, cond=cond, is_training=False
        ),
    )

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
    """Returns a bound conditional mean-flow inference function."""

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
          mean_flow_model.apply(
              variables, x_s=x, t=t_arr, s=s_arr, cond=cond, is_training=False
          ),
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalSymmetricMeanFlowModel(ConditionalMeanFlowModel):
  """Trains a conditional mean-flow model with symmetric `(t, s)` JVP loss."""

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the symmetric `(t, s)` mean-flow loss on a training batch."""
    batch_size = len(batch["x_0"])
    time_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_rng,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)

    cond = self._extract_cond(batch)
    v_t = self._compute_reference_velocity(time_t, x_t, batch, noise, cond)
    v_s = self._compute_reference_velocity(time_s, x_s, batch, noise, cond)

    mutable_keys = list(mutables.keys())
    variables = {"params": params, **mutables}

    def _partial_flow_map(
        x_in: Array, t_in: Array, s_in: Array
    ) -> tuple[Array, Any]:
      out, updated_mut = self._apply_mean_flow(
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

    u_ts, lin_fn, updated_mutables = jax.linearize(
        _partial_flow_map, x_t, time_t, time_s, has_aux=True
    )
    dt_u_st = lin_fn(
        jax.lax.stop_gradient(v_t).astype(x_t.dtype),
        jnp.ones_like(time_t),
        jnp.zeros_like(time_s),
    )
    ds_u_st = lin_fn(
        jnp.zeros_like(x_t), jnp.zeros_like(time_t), jnp.ones_like(time_s)
    )

    dt = jnp.maximum(time_t - time_s, 0.0).reshape(
        (x_t.shape[0],) + (x_t.ndim - 1) * (1,)
    )
    dt_cast = dt.astype(u_ts.dtype)
    u_target_t = v_t - dt_cast * dt_u_st
    u_target_s = v_s + dt_cast * ds_u_st

    err_t = jnp.asarray(u_ts - jax.lax.stop_gradient(u_target_t), jnp.float32)
    err_s = jnp.asarray(u_ts - jax.lax.stop_gradient(u_target_s), jnp.float32)
    reduce_axes = tuple(range(1, err_t.ndim))
    per_sample_mse_t = jnp.mean(jnp.square(err_t), axis=reduce_axes)
    per_sample_mse_s = jnp.mean(jnp.square(err_s), axis=reduce_axes)
    per_sample_mse = 0.5 * (per_sample_mse_t + per_sample_mse_s)
    weight = jnp.power(
        per_sample_mse + self.norm_regularization, self.norm_power
    )
    loss = jnp.mean(per_sample_mse / jax.lax.stop_gradient(weight))

    metric = {
        "loss": loss,
        "loss_t": jnp.mean(per_sample_mse_t),
        "loss_s": jnp.mean(per_sample_mse_s),
        "raw_mse": jnp.mean(per_sample_mse),
    }
    return loss, (metric, updated_mutables)
