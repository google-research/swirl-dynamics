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

r"""Training a flow map matching model.

Starting from a flow induced by stochastic interpolants [1], which take the form
  X_t = \alpha(t) * X_0 + \beta(t) * X_1 + \gamma(t) * Z
where X_0 \sim \mu_0, X_1 \sim \mu_1, and Z \sim N(0, 1).
We seek to lean the flow map X_{s,t} such that X_t = X_{s, t}(X_s) for any
s<t.

References:
[1]: Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.
Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.
[2]: Flow map matching with stochastic interpolants: A mathematical framework
for consistency models. Nicholas M. Boffi,  Michael S. Albergo, and Eric
Vanden-Eijnden.

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class LagrangianFlowMapModel(models.BaseModel):
  """Training a flow-based model for distribution matching.

  Attributes:
    input_shape: The tensor shape of a single sample (i.e. without any batch
      dimension).
    flow_map_model: The flax module to use for the flow.
    flow_model: The flax module to use for the teacher model.
    params_flow: The parameters of the flow model used as a teacher model.
    interpolant: The interpolant to use for the training.
    noising_process: A Callable that samples noise from the noising process.
    time_sampling: A Callable that samples the interpolation times at training.
    num_eval_cases_per_lvl: int = 1
    min_train_time: Minimum time at which the flow map and flow are sampled at
      training.
    max_train_time: Maximum time at which the flow map and flow are sampled at
      training.
    max_gap: Maximum gap between the times at which the flow map is sampled.
    number_of_eval_steps: Number of steps for the evaluation.
    weighted_norm: The norm to use for the loss, if None we use euclidean norm,
      otherwise we use weighted norm.
  """

  input_shape: tuple[int, ...]
  flow_map_model: nn.Module
  flow_model: nn.Module
  params_flow: VariableDict
  interpolant: interpolants.StochasticInterpolant
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  # TODO: Transfer the time sampling logic to the time_sampling
  # function instead of being handled inside the loss_fn.
  time_sampling: Callable[[Array, tuple[int, ...]], Array] = functools.partial(
      jax.random.uniform, dtype=jnp.float32
  )

  min_train_time: float = 1e-4  # This should be close to 0.
  max_train_time: float = 1.0 - 1e-4  # It should be close to 1.
  # Refactor the sampling time logic to be handled in the time_sampling
  # function.
  max_gap: float = 1.0

  num_eval_cases_per_lvl: int = 1
  number_of_eval_steps: tuple[int, ...] = (1, 4, 8)

  def initialize(self, rng: Array):
    # TODO: Add a dtype object to ensure consistency of types.
    x = jnp.ones((1,) + self.input_shape)

    return self.flow_map_model.init(
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
    time_sample_rng_t, time_sample_rng_s, dropout_rng, noise_rng = (
        jax.random.split(rng, num=4)
    )
    # Sampling the time.
    # TODO: encapsulate this in a function.
    time_range = self.max_train_time - self.min_train_time
    time_t = (
        time_range * self.time_sampling(time_sample_rng_t, (batch_size,))
        + self.min_train_time
    )
    time_s = (
        time_range * self.time_sampling(time_sample_rng_s, (batch_size,))
        + self.min_train_time
    )

    # Ensures that t>s.
    time_t, time_s = (jnp.where(time_t > time_s, time_t, time_s),
                      jnp.where(time_t < time_s, time_t, time_s))

    # Interpolation between x_0 and x_1 (check the interpolant)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    # Shall we add constraint that x_tt = x_t?
    # x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)

    # Partial flow map to compute the gradient vector product.
    def partial_flow_map(x_s, t, s):
      return self.flow_map_model.apply(
          {"params": params},
          x_s,
          t,
          s,
          is_training=True,
          rngs={"dropout": dropout_rng},
      )

    # Partial derivate with respect to t of the flow map.
    x_st, dt_x_st = jax.jvp(
        partial_flow_map,
        (x_s, time_t, time_s),
        (jnp.zeros_like(x_s), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    v_t = self.flow_model.apply(
        {"params": self.params_flow},
        x=x_st,
        sigma=time_t,
        is_training=False,
    )

    loss = jnp.mean(jnp.abs(dt_x_st - v_t))

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
    time_sample_rng_t, time_sample_rng_s, dropout_rng, noise_rng = (
        jax.random.split(rng, num=4)
    )
    # Sampling the time.
    # TODO: encapsulate this in a function.
    time_range = self.max_train_time - self.min_train_time
    time_t = (
        time_range * self.time_sampling(time_sample_rng_t, (batch_size,))
        + self.min_train_time
    )
    time_s = (
        time_range * self.time_sampling(time_sample_rng_s, (batch_size,))
        + self.min_train_time
    )

    # Ensures that t>s.
    time_t, time_s = (jnp.where(time_t > time_s, time_t, time_s),
                      jnp.where(time_t < time_s, time_t, time_s))

    # Interpolation between x_0 and x_1 (check the interpolant)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    # Shall we add constraint that x_tt = x_t?
    # x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)

    # Partial flow map to compute the gradient vector product.
    def partial_flow_map(x_s, t, s):
      return self.flow_map_model.apply(
          variables,
          x_s,
          t,
          s,
          is_training=False,
          rngs={"dropout": dropout_rng},
      )

    # Partial derivate with respect to t of the flow map.
    x_st, dt_x_st = jax.jvp(
        partial_flow_map,
        (x_s, time_t, time_s),
        (jnp.zeros_like(x_s), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    v_t = self.flow_model.apply(
        {"params": self.params_flow},
        x=x_st,
        sigma=time_t,
        is_training=False,
    )

    loss = jnp.mean(jnp.abs(dt_x_st - v_t))

    # TODO: Perform an evaluation and save the samples to be
    # visualized in tensorboard.
    eval_samples = {}
    def body_for_loop(i, x, delta_t):
      return partial_flow_map(
          x,
          delta_t * (i + 1) * jnp.ones((x.shape[0],)),
          delta_t * i * jnp.ones((x.shape[0],)),
      )

    for num_steps in self.number_of_eval_steps:
      delta_t = 1./ num_steps
      body_for_loop = functools.partial(body_for_loop, delta_t=delta_t)
      samples = jax.lax.fori_loop(0, num_steps, body_for_loop, batch["x_0"])
      # Using the casting to jnp.array to avoid a type error.
      eval_samples[f"{num_steps}"] = jnp.array(samples)

    eval_losses = {"eval_loss": loss, **eval_samples}

    # In order to plot the samples, we return a dictionary of the form
    # Mapping[str, jax.Array | Mapping[str, jax.Array]],
    # which is different from the type annotation. We disable the pytype error.
    return eval_losses  # pytype: disable=bad-return-type

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_map_model: nn.Module):
    """Returns the inference flow function."""

    def _flow(x: Array, t: float | Array, s: float | Array) -> Array:
      # This is a wrapper to vectorize time if it is a float.
      if not jnp.shape(jnp.asarray(t)):
        t *= jnp.ones((x.shape[0],))
        s *= jnp.ones((x.shape[0],))
      return flow_map_model.apply(variables, x_s=x, t=t, s=s, is_training=False)

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalLagrangianFlowMapModel(LagrangianFlowMapModel):
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

    return self.flow_map_model.init(  # add conditional input here.
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
    time_sample_rng_t, time_sample_rng_s, dropout_rng, noise_rng = (
        jax.random.split(rng, num=4)
    )
    # Sampling the time.
    # TODO: encapsulate this in a function.
    time_range = self.max_train_time - self.min_train_time
    time_t = (
        time_range * self.time_sampling(time_sample_rng_t, (batch_size,))
        + self.min_train_time
    )
    time_s = (
        time_range * self.time_sampling(time_sample_rng_s, (batch_size,))
        + self.min_train_time
    )

    # Ensures that t>s.
    # TODO: learn this even if t<s. Would it make sense to add
    # symmetries here?
    time_t, time_s = (jnp.where(time_t > time_s, time_t, time_s),
                      jnp.where(time_t < time_s, time_t, time_s))

    # Interpolation between x_0 and x_1 (check the interpolant)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    # Shall we add constraint that x_tt = x_t?
    # x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)

    # Extracting the conditioning. In this case it is just the label.
    cond = {key: batch[key] for key in self.cond_keys}

    # Partial flow map to compute the gradient vector product.
    def partial_flow_map(x_s, t, s):
      return self.flow_map_model.apply(
          {"params": params},
          x_s,
          t,
          s,
          cond=cond,
          is_training=True,
          rngs={"dropout": dropout_rng},
      )

    # Partial derivate with respect to s of the flow map.
    x_st, dt_x_st = jax.jvp(
        partial_flow_map,
        (x_s, time_t, time_s),
        (jnp.zeros_like(x_s), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    v_t = self.flow_model.apply(
        {"params": self.params_flow},
        x=x_st,
        sigma=time_t,
        cond=cond,
        is_training=False,
    )

    loss = jnp.mean(jnp.abs(dt_x_st - v_t))

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
    time_sample_rng_t, time_sample_rng_s, dropout_rng, noise_rng = (
        jax.random.split(rng, num=4)
    )
    # Sampling the time.
    # TODO: encapsulate this in a function.
    time_range = self.max_train_time - self.min_train_time
    time_t = (
        time_range * self.time_sampling(time_sample_rng_t, (batch_size,))
        + self.min_train_time
    )
    time_s = (
        time_range * self.time_sampling(time_sample_rng_s, (batch_size,))
        + self.min_train_time
    )

    # # Ensures that t>s.
    time_t, time_s = (jnp.where(time_t > time_s, time_t, time_s),
                      jnp.where(time_t < time_s, time_t, time_s))

    # Extracting the conditioning.
    cond = {key: batch[key] for key in self.cond_keys}

    # Interpolation between x_0 and x_1 (check the interpolant)
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    # Shall we add constraint that x_tt = x_t?
    # x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    x_s = self.interpolant(time_s, batch["x_0"], batch["x_1"], noise)

    # Partial flow map to compute the gradient vector product.
    def partial_flow_map(x_s, t, s):
      return self.flow_map_model.apply(
          variables,
          x_s,
          t,
          s,
          cond=cond,
          is_training=True,
          rngs={"dropout": dropout_rng},
      )

    # Partial derivate with respect to t of the flow map.
    x_st, dt_x_st = jax.jvp(
        partial_flow_map,
        (x_s, time_t, time_s),
        (jnp.zeros_like(x_s), jnp.ones_like(time_t), jnp.zeros_like(time_s)),
    )

    v_t = self.flow_model.apply(
        {"params": self.params_flow},
        x=x_st,
        sigma=time_t,
        cond=cond,
        is_training=False,
    )

    loss = jnp.mean(jnp.abs(dt_x_st - v_t))

    eval_samples = {}
    def body_for_loop(i, x, delta_t):
      return partial_flow_map(
          x,
          delta_t * (i + 1) * jnp.ones((x.shape[0],)),
          delta_t * i * jnp.ones((x.shape[0],)),
      )

    for num_steps in self.number_of_eval_steps:
      delta_t = 1./ num_steps
      body_for_loop = functools.partial(body_for_loop, delta_t=delta_t)
      samples = jax.lax.fori_loop(0, num_steps, body_for_loop, batch["x_0"])
      # Using the casting to jnp.array to avoid a type error.
      eval_samples[f"{num_steps}"] = jnp.array(samples)

    eval_losses = {"eval_loss": loss, **eval_samples}

    return eval_losses  # pytype: disable=bad-return-type

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_map_model: nn.Module):
    """Returns the inference conditional flow function."""

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
      return flow_map_model.apply(
          variables, x_s=x, t=t, s=s, cond=cond, is_training=False
      )

    return _flow


class RescaledFlowMapUNet(backbones.FlowMapUNet):
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
      raise ValueError(
          f"Time dimensions must be equal ({t.ndim} != {s.ndim})."
      )

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

    s_theta = super().__call__(
        x_s, time_t, time_s, cond, is_training=is_training
    )

    delta_time = (t - s).reshape((s.shape[0],) + (x_s.ndim - 1) * (1,))
    # Imposes the boundary condition exactly. See Eq. 4.1 in [2].
    x_t = x_s * (1 - delta_time) + delta_time * s_theta
    return x_t

# TODO: Create a three-dimensional version of this model.
