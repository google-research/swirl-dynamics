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

r"""Training a mean_flow model (also for distillation).

References:
[1]: Mean Flows for One-step Generative Modeling: Zhengyang Geng, Mingyang Deng,
Xingjian Bai, J. Zico Kolter, and Kaiming He. (2025).
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
from typing import Any, TypeAlias

from clu import metrics as clu_metrics
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.diffusion import denoising_models
from swirl_dynamics.lib.flow_matching import interpolants
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers

Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jax.typing.ArrayLike
ArrayShape: TypeAlias = Sequence[int]
CondDict: TypeAlias = Mapping[str, Array]
DTypeLike: TypeAlias = jax.typing.DTypeLike
Metrics: TypeAlias = clu_metrics.Collection
ShapeDict: TypeAlias = Mapping[str, Any]
PyTree: TypeAlias = Any
VariableDict: TypeAlias = trainers.VariableDict
SamplingFn: TypeAlias = Callable[[ArrayLike, ArrayShape, Array], Array]

cond_sample_from_shape = denoising_models.cond_sample_from_shape


def logit_normal_dist(
    rng: Array,
    shape: ArrayShape,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: DTypeLike = jnp.float32,
) -> Array:
  """Samples logit-normal time steps in (0, 1)."""
  rnd_normal = jax.random.normal(rng, shape, dtype=dtype)
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
  """Samples the time for the mean flow model."""
  diagonal_rng, time_sample_rng_t, time_sample_rng_s = jax.random.split(
      rng, num=3
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

  diagonal_bool = jax.random.bernoulli(
      diagonal_rng, diagonal_sampling_rate, (batch_size,)
  )
  time_s = jnp.where(diagonal_bool, time_t, time_s)

  time_t, time_s = (
      jnp.where(time_t >= time_s, time_t, time_s),
      jnp.where(time_t < time_s, time_t, time_s),
  )
  return time_t, time_s


@dataclasses.dataclass(frozen=True, kw_only=True)
class MeanFlowModel(models.BaseModel):
  """Training a mean flow model for distribution matching."""

  input_shape: tuple[int, ...]
  mean_flow_model: nn.Module
  flow_model: nn.Module | None = None
  params_flow: VariableDict | None = None
  interpolant: interpolants.StochasticInterpolant = interpolants.RectifiedFlow()
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  time_sampling: Callable[..., tuple[Array, Array]] = time_sampler_mean_flow

  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4
  max_gap: float = 1.0
  number_of_eval_steps: tuple[int, ...] = (1, 4, 8)
  norm_power: float = 1.0
  norm_regularization: float = 1e-2
  diagonal_sampling_rate: float = 0.25

  def initialize(self, rng: Array):
    x = jnp.ones((1,) + self.input_shape)
    return self.mean_flow_model.init(
        rng,
        x_s=x,
        t=jnp.ones((1,)),
        s=jnp.ones((1,)),
        is_training=False,
    )

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the mean flow loss on a training batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    if self.flow_model is not None:
      if self.params_flow is None:
        raise ValueError(
            "`params_flow` must be provided when `flow_model` is set."
        )
      v_t = self.flow_model.apply(
          {"params": self.params_flow},
          x=x_t,
          sigma=time_t,
          is_training=False,
      )
    else:
      v_t = self.interpolant.calculate_time_derivative_interpolant(
          time_t, batch["x_0"], batch["x_1"], noise
      )

    def partial_flow_map(x_t, t, s):
      return self.mean_flow_model.apply(
          {"params": params},
          x_t,
          t,
          s,
          is_training=True,
          rngs={"dropout": dropout_rng},
      )

    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    dt = jnp.clip(time_t - time_s, min=0.0, max=1.0).reshape(
        (x_t.shape[0],) + (x_t.ndim - 1) * (1,)
    )
    u_target = v_t - dt * dt_u_st

    reduce_axes = tuple(range(1, u_ts.ndim))
    per_sample_mse = jnp.mean(
        jnp.square(u_ts - jax.lax.stop_gradient(u_target)),
        axis=reduce_axes,
    )
    adaptive_weight = (
        per_sample_mse + self.norm_regularization
    ) ** self.norm_power
    loss = jnp.mean(per_sample_mse / jax.lax.stop_gradient(adaptive_weight))
    metric = dict(loss=loss, raw_mse=jnp.mean(per_sample_mse))
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes metrics on an eval batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    if self.flow_model is not None:
      if self.params_flow is None:
        raise ValueError(
            "`params_flow` must be provided when `flow_model` is set."
        )
      v_t = self.flow_model.apply(
          {"params": self.params_flow},
          x=x_t,
          sigma=time_t,
          is_training=False,
      )
    else:
      v_t = self.interpolant.calculate_time_derivative_interpolant(
          time_t, batch["x_0"], batch["x_1"], noise
      )

    def partial_flow_map(x_t, t, s):
      return self.mean_flow_model.apply(
          variables,
          x_t,
          t,
          s,
          is_training=False,
          rngs={"dropout": dropout_rng},
      )

    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    dt = jnp.clip(time_t - time_s, min=0.0, max=1.0).reshape(
        (x_t.shape[0],) + (x_t.ndim - 1) * (1,)
    )
    u_target = v_t - dt * dt_u_st
    loss = jnp.mean(jnp.abs(u_ts - jax.lax.stop_gradient(u_target)))
    eval_samples = batch["x_1"] - partial_flow_map(
        batch["x_1"], jnp.ones_like(time_t), jnp.zeros_like(time_s)
    )
    return {"eval_loss": loss, "eval_samples": eval_samples}

  @classmethod
  def inference_fn(cls, variables: models.PyTree, mean_flow_model: nn.Module):  # pyrefly: ignore[bad-override]
    """Returns the inference flow function."""

    def _flow(x: Array, t: float | Array, s: float | Array) -> Array:
      t_arr = jnp.broadcast_to(jnp.asarray(t), (x.shape[0],))
      s_arr = jnp.broadcast_to(jnp.asarray(s), (x.shape[0],))
      return mean_flow_model.apply(  # pyrefly: ignore[bad-return]
          variables, x_s=x, t=t_arr, s=s_arr, is_training=False
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalMeanFlowModel(MeanFlowModel):
  """Training a conditional mean flow model for distribution matching."""

  cond_shape: ShapeDict | None = None
  number_of_eval_steps: tuple[int, ...] = (1,)

  def initialize(self, rng: Array) -> models.PyTree:
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

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the conditional mean flow loss on a training batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    if self.cond_shape is not None:
      cond = {key: batch[key] for key in self.cond_shape.keys()}
    else:
      cond = None

    if self.flow_model is not None:
      if self.params_flow is None:
        raise ValueError(
            "`params_flow` must be provided when `flow_model` is set."
        )
      v_t = self.flow_model.apply(
          {"params": self.params_flow},
          x=x_t,
          sigma=time_t,
          cond=cond,
          is_training=False,
      )
    else:
      v_t = self.interpolant.calculate_time_derivative_interpolant(
          time_t, batch["x_0"], batch["x_1"], noise
      )

    def partial_flow_map(x_t, t, s):
      return self.mean_flow_model.apply(
          {"params": params},
          x_t,
          t,
          s,
          cond=cond,
          is_training=True,
          rngs={"dropout": dropout_rng},
      )

    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    dt = jnp.clip(time_t - time_s, min=0.0, max=1.0).reshape(
        (x_t.shape[0],) + (x_t.ndim - 1) * (1,)
    )
    u_target = v_t - dt * dt_u_st

    reduce_axes = tuple(range(1, u_ts.ndim))
    per_sample_mse = jnp.mean(
        jnp.square(u_ts - jax.lax.stop_gradient(u_target)),
        axis=reduce_axes,
    )
    adaptive_weight = (
        per_sample_mse + self.norm_regularization
    ) ** self.norm_power
    loss = jnp.mean(per_sample_mse / jax.lax.stop_gradient(adaptive_weight))
    metric = dict(loss=loss, raw_mse=jnp.mean(per_sample_mse))
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes metrics on an eval batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    if self.cond_shape is not None:
      cond = {key: batch[key] for key in self.cond_shape.keys()}
    else:
      cond = None

    if self.flow_model is not None:
      if self.params_flow is None:
        raise ValueError(
            "`params_flow` must be provided when `flow_model` is set."
        )
      v_t = self.flow_model.apply(
          {"params": self.params_flow},
          x=x_t,
          sigma=time_t,
          cond=cond,
          is_training=False,
      )
    else:
      v_t = self.interpolant.calculate_time_derivative_interpolant(
          time_t, batch["x_0"], batch["x_1"], noise
      )

    def partial_flow_map(x_t, t, s):
      return self.mean_flow_model.apply(
          variables,
          x_t,
          t,
          s,
          cond=cond,
          is_training=False,
          rngs={"dropout": dropout_rng},
      )

    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    dt = jnp.clip(time_t - time_s, min=0.0, max=1.0).reshape(
        (x_t.shape[0],) + (x_t.ndim - 1) * (1,)
    )
    u_target = v_t - dt * dt_u_st
    loss = jnp.mean(jnp.abs(u_ts - jax.lax.stop_gradient(u_target)))
    eval_samples = batch["x_1"] - partial_flow_map(
        batch["x_1"], jnp.ones_like(time_t), jnp.zeros_like(time_s)
    )
    return {"eval_loss": loss, "eval_samples": eval_samples}

  @classmethod
  def inference_fn(cls, variables: models.PyTree, mean_flow_model: nn.Module):
    """Returns the inference conditional mean flow function."""

    def _flow(
        x: Array,
        t: float | Array,
        s: float | Array,
        cond: CondDict | None = None,
    ) -> Array:
      t_arr = jnp.broadcast_to(jnp.asarray(t), (x.shape[0],))
      s_arr = jnp.broadcast_to(jnp.asarray(s), (x.shape[0],))
      return mean_flow_model.apply(  # pyrefly: ignore[bad-return]
          variables, x_s=x, t=t_arr, s=s_arr, cond=cond, is_training=False
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalSymmetricMeanFlowModel(ConditionalMeanFlowModel):
  """Training a conditional symmetric mean flow model."""

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the symmetric mean flow loss on a training batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)

    if self.cond_shape is not None:
      cond = {key: batch[key] for key in self.cond_shape.keys()}
    else:
      cond = None

    if self.flow_model is not None:
      if self.params_flow is None:
        raise ValueError(
            "`params_flow` must be provided when `flow_model` is set."
        )
      v_t = self.flow_model.apply(
          {"params": self.params_flow},
          x=x_t,
          sigma=time_t,
          cond=cond,
          is_training=False,
      )
      v_s = self.flow_model.apply(
          {"params": self.params_flow},
          x=x_s,
          sigma=time_s,
          cond=cond,
          is_training=False,
      )
    else:
      v_t = self.interpolant.calculate_time_derivative_interpolant(
          time_t, batch["x_0"], batch["x_1"], noise
      )
      v_s = self.interpolant.calculate_time_derivative_interpolant(
          time_s, batch["x_0"], batch["x_1"], noise
      )

    def partial_flow_map(x_t, t, s):
      return self.mean_flow_model.apply(
          {"params": params},
          x_t,
          t,
          s,
          cond=cond,
          is_training=True,
          rngs={"dropout": dropout_rng},
      )

    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )
    _, ds_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (jnp.zeros_like(x_t), jnp.zeros_like(time_t), jnp.ones_like(time_s)),
    )

    dt = (time_t - time_s).reshape((x_t.shape[0],) + (x_t.ndim - 1) * (1,))
    u_target_t = v_t - dt * dt_u_st
    u_target_s = v_s + dt * ds_u_st

    reduce_axes = tuple(range(1, u_ts.ndim))
    per_sample_mse_t = jnp.mean(
        jnp.square(u_ts - jax.lax.stop_gradient(u_target_t)),
        axis=reduce_axes,
    )
    per_sample_mse_s = jnp.mean(
        jnp.square(u_ts - jax.lax.stop_gradient(u_target_s)),
        axis=reduce_axes,
    )
    per_sample_mse = 0.5 * (per_sample_mse_t + per_sample_mse_s)
    adaptive_weight = (
        per_sample_mse + self.norm_regularization
    ) ** self.norm_power
    loss = jnp.mean(per_sample_mse / jax.lax.stop_gradient(adaptive_weight))
    loss_t = jnp.mean(per_sample_mse_t)
    loss_s = jnp.mean(per_sample_mse_s)

    metric = dict(
        loss=loss,
        loss_t=loss_t,
        loss_s=loss_s,
        raw_mse=jnp.mean(per_sample_mse),
    )
    return loss, (metric, mutables)
