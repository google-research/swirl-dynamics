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

r"""Training a mean_flow model (also for distillation).

Mean flow models seeks to approximate the mean flow between two distributions,
namely, $\rho_0$ and $\rho_1$. Here we assume that the mean flow is given by a
stochastic interpolant of the form:

$$x_t = \alpha(t) x_0 + \beta(t) x_1,$$

where $\alpha(t)$ and $\beta(t)$ are the interpolation weights, and $x_0$ and
$x_1$ are the samples from $\rho_0$ and $\rho_1$ respectively.

Thus the flow map is given by:

$$u(x_t, t, s) =  \frac{1}{t-s} int_{s}^t v(x_{\tau}, \tau}) d\tau,$$

where u(x_t, t, s) is the flow map, x_t is a sample from the interpolant,
t and s are the times, and v(x, \tau) is the flow. The latter can be given by
a trained networks, or it can be given by the interpolant itself, in the form

$$\dot{x}_t = \alpha'(t) x_0 + \beta'(t) x_1.$$

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
from swirl_dynamics.projects.debiasing.stochastic_interpolants import interpolants
from swirl_dynamics.projects.distillation.flow_map_matching import backbones
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers


# Defining aliases for the types.
Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jax.typing.ArrayLike  # This is for the keys.
ArrayShape: TypeAlias = Sequence[int]
CondDict: TypeAlias = Mapping[str, Array]
Metrics: TypeAlias = clu_metrics.Collection
ShapeDict: TypeAlias = Mapping[str, Any]  # may be nested
PyTree: TypeAlias = Any
VariableDict: TypeAlias = trainers.VariableDict
SamplingFn: TypeAlias = Callable[[ArrayLike, ArrayShape, Array], Array]
# StochasticInterpolantLossFn: TypeAlias = losses.StochasticInterpolantLossFn


def cond_sample_from_shape(
    shape: ShapeDict | None, batch_dims: tuple[int, ...] = (1,)
) -> PyTree | None:
  """Instantiates a conditional input sample based on shape specifications."""
  if shape is None:
    return None
  elif isinstance(shape, tuple):
    return jnp.ones(batch_dims + shape)
  elif isinstance(shape, dict):
    return {k: cond_sample_from_shape(v, batch_dims) for k, v in shape.items()}
  else:
    raise TypeError(f"Cannot initialize shape: {shape}")


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
    diagonal_sampling_rate: float = 0.25,
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
    diagonal_sampling_rate: The rate at which we sample the diagonal of the s
      and t, thus recovering the conditional flow.
    time_sampling: The function to use for the time sampling of t and s.

  Returns:
    The sampled times t and s.
  """
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

  # We follow 4.3 in [1], and we sample the diagonal of the s and t.
  diagonal_bool = jax.random.bernoulli(
      diagonal_rng, diagonal_sampling_rate, (batch_size,)
  )

  # If diagonal bool is true, then time_s is equal to time_t.
  time_s = jnp.where(diagonal_bool, time_t, time_s)

  # Ensures that t>s.
  time_t, time_s = (
      jnp.where(time_t >= time_s, time_t, time_s),
      jnp.where(time_t < time_s, time_t, time_s),
  )

  return time_t, time_s


@dataclasses.dataclass(frozen=True, kw_only=True)
class MeanFlowModel(models.BaseModel):
  """Training a mean flow model for distribution matching.

  Attributes:
    input_shape: The tensor shape of a single sample (i.e. without any batch
      dimension).
    mean_flow_model: The flax module to use for the flow.
    flow_model: The flax module to use for the teacher model. If None, we use
      the interpolant to compute the flow.
    params_flow: The parameters of the flow model used as a teacher model. If
      None, we use the interpolant to compute the flow.
    interpolant: The interpolant to use for the training. In most of the cases,
      we use a RectifiedFlow interpolant.
    noising_process: A Callable that samples noise from the noising process.
    time_sampling: A Callable that samples the interpolation times at training.
    min_train_time: Minimum time at which the flow map and flow are sampled at
      training.
    max_train_time: Maximum time at which the flow map and flow are sampled at
      training.
    max_gap: Maximum gap between the times at which the flow map is sampled.
    weighted_norm: The norm to use for the loss, if None we use euclidean norm,
      otherwise we use weighted norm.
    norm_power: The exponent to use for the norm re-weighting.
    norm_regularization: The regularization term to use for the norm
      re-weighting.
    diagonal_sampling_rate: The rate at which we sample the diagonal of the s
      and t, thus recovering the conditional flow.
  """

  input_shape: tuple[int, ...]
  mean_flow_model: nn.Module
  flow_model: nn.Module | None = None
  params_flow: VariableDict | None = None
  interpolant: interpolants.StochasticInterpolant = interpolants.RectifiedFlow()
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  # TODO: Transfer the time sampling logic to the time_sampling
  # function instead of being handled inside the loss_fn.
  time_sampling: Callable[..., tuple[Array, Array]] = time_sampler_mean_flow

  min_train_time: float = 1e-4  # This should be close to 0.
  max_train_time: float = 1.0 - 1e-4  # It should be close to 1.
  # Refactor the sampling time logic to be handled in the time_sampling
  # function.
  max_gap: float = 1.0
  number_of_eval_steps: tuple[int, ...] = (1, 4, 8)
  norm_power: float = 1.0
  norm_regularization: float = 1e-2
  diagonal_sampling_rate: float = 0.25

  def initialize(self, rng: Array):
    # TODO: Add a dtype object to ensure consistency of types.
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
    """Computes the flow matching loss on a training batch.

    Args:
      params: The parameters of the flow model to differentiate against.
      batch: A batch of training data expected to contain two fields, namely
        `x_0` and `x_1` corresponding to samples from different sets. Each field
        is expected to have a shape of `(batch, *spatial_dims, channels)`.
      rng: A Jax random key.
      mutables: The mutable (non-differentiated) parameters of the flow model
        (e.g. batch stats); *currently assumed empty*.

    Returns:
      The loss value and a tuple of training metric and mutables.
    """
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    # Sampling the time.
    # TODO: encapsulate this in a function.
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    # Interpolation between x_0 and x_1 (check the interpolant)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    # Computes the velocity at time t. If we are distilling a flow map, we use
    # the flow map model, otherwise we use the interpolant.
    if self.flow_model is not None:
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

    # Partial flow map to compute the gradient vector product.
    def partial_flow_map(x_t, t, s):
      return self.mean_flow_model.apply(
          {"params": params},
          x_t,
          t,
          s,
          is_training=True,
          rngs={"dropout": dropout_rng},
      )

    # Partial derivate with respect to t of the flow map.
    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    u_target = v_t - jnp.clip(time_t - time_s, min=0.0, max=1.0) * dt_u_st

    loss = jnp.mean(jnp.square(u_ts - jax.lax.stop_gradient(u_target)))

    # Using the adaptive norm in section 4.3 of [1].
    adaptive_weight = (loss + self.norm_regularization) ** self.norm_power
    loss = loss / jax.lax.stop_gradient(adaptive_weight)

    metric = dict(loss=loss)

    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Compute metrics on an eval batch.

    Randomly selects members of the batch and interpolates them at different
    points between 0 and 1 and computed the flow. The error of the flow at each
    point is aggregated.

    Args:
      variables: The full model variables for the flow module.
      batch: A batch of evaluation data expected to contain two fields `x_0` and
        `x_1` fields, representing samples of each set. Both fields are expected
        to have shape of `(batch, *spatial_dims, channels)`.
      rng: A Jax random key.

    Returns:
      A dictionary of evaluation metrics.
    """
    # We bootstrap the samples from the batch, but we keep them paired by using
    # the same random number generator seed.
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    # Sampling the time.
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    # Interpolation between x_0 and x_1 (check the interpolant)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    # Shall we add constraint that x_tt = x_t?
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    # Computes the velocity at time t. If we are distilling a flow, we use
    # the flow model, otherwise we use the interpolant.
    if self.flow_model is not None:
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

    # Partial flow map to compute the gradient vector product.
    def partial_flow_map(x_t, t, s):
      return self.mean_flow_model.apply(
          variables,
          x_t,
          t,
          s,
          is_training=False,
          rngs={"dropout": dropout_rng},
      )

    # Partial derivate with respect to t of the flow map.
    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    u_target = v_t - jnp.clip(time_t - time_s, min=0.0, max=1.0) * dt_u_st
    loss = jnp.mean(jnp.abs(u_ts - jax.lax.stop_gradient(u_target)))

    # Samples from one-step generation. Following [1], \rho_0 is the target
    # distribution and \rho_1 is the noise.
    eval_samples = batch["x_1"] - partial_flow_map(
        batch["x_1"], jnp.ones_like(time_t), jnp.zeros_like(time_s)
    )

    eval_losses = {"eval_loss": loss, "eval_samples": eval_samples}

    return eval_losses  # pytype: disable=bad-return-type

  @classmethod
  def inference_fn(cls, variables: models.PyTree, mean_flow_model: nn.Module):
    """Returns the inference flow function."""

    def _flow(x: Array, t: float | Array, s: float | Array) -> Array:
      # This is a wrapper to vectorize time if it is a float.
      if not jnp.shape(jnp.asarray(t)):
        t *= jnp.ones((x.shape[0],))
        s *= jnp.ones((x.shape[0],))
      return mean_flow_model.apply(
          variables, x_s=x, t=t, s=s, is_training=False
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalMeanFlowModel(MeanFlowModel):
  """Training a conditional flow-based model for distribution matching.

  Attributes:
    cond_shape: Shape of the conditional input.
  """

  cond_shape: ShapeDict | None = None
  cond_keys: tuple[str, ...] = ("emb:label",)
  number_of_eval_steps: tuple[int, ...] = (1,)

  def initialize(self, rng: Array) -> models.PyTree:
    x = jnp.ones((1,) + self.input_shape)
    cond = cond_sample_from_shape(self.cond_shape, batch_dims=(1,))

    return self.mean_flow_model.init(  # add conditional input here.
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
    """Computes the flow matching loss on a training batch.

    Args:
      params: The parameters of the flow model to differentiate against.
      batch: A batch of training data expected to contain two fields, namely
        `x_0` and `x_1` corresponding to samples from different sets. Each field
        is expected to have a shape of `(batch, *spatial_dims, channels)`.
      rng: A Jax random key.
      mutables: The mutable (non-differentiated) parameters of the flow model
        (e.g. batch stats); *currently assumed empty*.

    Returns:
      The loss value and a tuple of training metric and mutables.
    """
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    # Sampling the times.
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    # Interpolation between x_0 and x_1.
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    # Extracting the conditioning. In this case it is just the label.
    cond = {key: batch[key] for key in self.cond_keys}

    # Computes the velocity at time t. If we are distilling a flow map, we use
    # the flow map model, otherwise we use the interpolant.
    if self.flow_model is not None:
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

    # Partial flow map to compute the gradient vector product.
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

    # Partial derivate with respect to t of the flow map.
    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    u_target = v_t - jnp.clip(time_t - time_s, min=0.0, max=1.0) * dt_u_st

    # Using the adaptive norm in section 4.3 of [1].
    loss = jnp.mean(jnp.square(u_ts - jax.lax.stop_gradient(u_target)))
    adaptive_weight = (loss + self.norm_regularization) ** self.norm_power
    loss = loss / jax.lax.stop_gradient(adaptive_weight)

    metric = dict(loss=loss)
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Compute metrics on an eval batch.

    Randomly selects members of the batch and interpolates them at different
    points between 0 and 1 and computed the flow. The error of the flow at each
    point is aggregated.

    Args:
      variables: The full model variables for the flow module.
      batch: A batch of evaluation data expected to contain two fields `x_0` and
        `x_1` fields, representing samples of each set. Both fields are expected
        to have shape of `(batch, *spatial_dims, channels)`.
      rng: A Jax random key.

    Returns:
      A dictionary of evaluation metrics.
    """
    # We bootstrap the samples from the batch, but we keep them paired by using
    # the same random number generator seed.
    # TODO: Avoid repeated code in here.
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    # Sampling the time.
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    # Interpolation between x_0 and x_1 (check the interpolant)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    # Extracting the conditioning.
    cond = {key: batch[key] for key in self.cond_keys}

    # Computes the velocity at time t. If we are distilling a flow map, we use
    # the flow map model, otherwise we use the interpolant.
    if self.flow_model is not None:
      if self.params_flow is None:
        raise ValueError("Params flow is None.")
      else:
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

    # Partial flow map to compute the gradient vector product.
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

    # Partial derivate with respect to t of the flow map.
    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    u_target = v_t - jnp.clip(time_t - time_s, min=0.0, max=1.0) * dt_u_st

    # We don't use the adaptive norm here.
    loss = jnp.mean(jnp.abs(u_ts - jax.lax.stop_gradient(u_target)))

    # Samples from one-step generation. Following [1], \rho_0 is the target
    # distribution and \rho_1 is the noise.
    eval_samples = batch["x_1"] - partial_flow_map(
        batch["x_1"], jnp.ones_like(time_t), jnp.zeros_like(time_s)
    )

    eval_losses = {"eval_loss": loss, "eval_samples": eval_samples}

    return eval_losses  # pytype: disable=bad-return-type

  @classmethod
  def inference_fn(cls, variables: models.PyTree, mean_flow_model: nn.Module):
    """Returns the inference conditional mean flow function."""

    def _flow(
        x: Array,
        t: float | Array,
        s: float | Array,
        cond: CondDict | None = None,
    ) -> Array:
      # This is a wrapper to vectorize time if it is a float.
      if not jnp.shape(jnp.asarray(t)):
        t *= jnp.ones((x.shape[0],))
        s *= jnp.ones((x.shape[0],))
      return mean_flow_model.apply(
          variables, x_s=x, t=t, s=s, cond=cond, is_training=False
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalSymmetricMeanFlowModel(ConditionalMeanFlowModel):
  """Training a conditional flow-based model for distribution matching.

  We preserve most of the code from the ConditionalMeanFlowModel, but we
  replace the loss function with a symmetric version. In which we compute the
  derivate with respect to both t and s.
  """

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the flow matching loss on a training batch.

    Args:
      params: The parameters of the flow model to differentiate against.
      batch: A batch of training data expected to contain two fields, namely
        `x_0` and `x_1` corresponding to samples from different sets. Each field
        is expected to have a shape of `(batch, *spatial_dims, channels)`.
      rng: A Jax random key.
      mutables: The mutable (non-differentiated) parameters of the flow model
        (e.g. batch stats); *currently assumed empty*.

    Returns:
      The loss value and a tuple of training metric and mutables.
    """
    batch_size = len(batch["x_0"])
    time_sample_rng_t, dropout_rng, noise_rng = jax.random.split(rng, num=3)
    # Sampling the time.
    time_t, time_s = self.time_sampling(
        time_sample_rng_t,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    # Interpolation between x_0 and x_1 (check the interpolant)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    # Shall we add constraint that x_tt = x_t?
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)

    # Extracting the conditioning. In this case it is just the label.
    cond = {key: batch[key] for key in self.cond_keys}

    # Computes the velocity at time t. If we are distilling a flow map, we use
    # the flow map model, otherwise we use the interpolant.
    if self.flow_model is not None:
      v_t = self.flow_model.apply(
          {"params": self.params_flow},
          x=x_t,
          sigma=time_t,
          cond=cond,
          is_training=False,
      )
      v_s = self.flow_model.apply(
          {"params": self.params_flow},
          x=x_t,
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

    # Partial flow map to compute the gradient vector product.
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

    # Partial derivate with respect to t of the flow map.
    u_ts, dt_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (v_t, jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    _, ds_u_st = jax.jvp(
        partial_flow_map,
        (x_t, time_t, time_s),
        (jnp.zeros_like(time_t), jnp.zeros_like(time_t), jnp.ones_like(time_s)),
    )

    u_target_t = v_t - (time_t - time_s) * dt_u_st
    u_target_s = - v_s + (time_t - time_s) * ds_u_st

    # Using the adaptive norm in section 4.3 of [1].
    loss_t = jnp.mean(jnp.square(u_ts - jax.lax.stop_gradient(u_target_t)))
    loss_s = jnp.mean(jnp.square(u_ts - jax.lax.stop_gradient(u_target_s)))

    loss = 0.5 * (loss_t + loss_s)
    adaptive_weight = (loss + self.norm_regularization) ** self.norm_power
    loss = loss / jax.lax.stop_gradient(adaptive_weight)

    metric = dict(loss=loss, loss_t=loss_t, loss_s=loss_s)
    return loss, (metric, mutables)


class RescaledMeanFlowUNet(backbones.FlowMapUNet):
  """Rescaled flow map model.

  Attributes:
    time_rescale: Factor for rescaling the time, which normally is in [0, 1] and
      the input to the UNet which is a noise level, which has a much wider range
      of values.
  """

  time_rescale: float = 1.0

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
    """Runs rescaled Unet with noise input."""
    if x_s.shape[-1] != self.out_channels:
      raise ValueError(
          f"Number of channels in the input ({x_s.shape[-1]}) must "
          "match the number of channels in the output "
          f"{self.out_channels})."
      )
    if t.ndim != s.ndim:
      raise ValueError(f"Time dimensions must be equal ({t.ndim} != {s.ndim}).")

    if s.ndim < 1:
      s = jnp.broadcast_to(s, (x_s.shape[0],))
      t = jnp.broadcast_to(t, (x_s.shape[0],))

    if s.ndim != 1 or x_s.shape[0] != s.shape[0] or x_s.shape[0] != t.shape[0]:
      raise ValueError(
          "s must be 1D and have the same leading (batch) dimension as x and t."
          f"Instead x_s has leading dimension ({x_s.shape[0]}),"
          f"s has leading dimension ({s.shape[0]}), and "
          f"t has leading dimension ({t.shape[0]})."
      )

    time_s = s * self.time_rescale
    time_t = t * self.time_rescale

    u_theta = super().__call__(
        x_s, time_t, time_s, cond, is_training=is_training
    )

    return u_theta


# TODO: Create a three-dimensional version of this model.
