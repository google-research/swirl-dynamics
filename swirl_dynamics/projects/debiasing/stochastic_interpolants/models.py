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

r"""Training a flow model for stochastic interpolants.

Stochastic interpolants  flow [1] seeks to train a flow model
  X_t = \alpha(t) * X_0 + \beta(t) * X_1 + \gamma(t) * Z
where X_0 \sim \mu_0, X_1 \sim \mu_1, and Z \sim N(0, 1).
This expression links two distributions $\mu_0$ and $\mu_1$, such that the
initial and final conditions are drawn from such distributions.

References:
[1]: Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
from typing import Any, ClassVar, TypeAlias

from clu import metrics as clu_metrics
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.diffusion import unets3d
from swirl_dynamics.projects.debiasing.stochastic_interpolants import backbones
from swirl_dynamics.projects.debiasing.stochastic_interpolants import interpolants
from swirl_dynamics.projects.debiasing.stochastic_interpolants import losses
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers


# Defining aliases for the types.
Array: TypeAlias = jax.Array
ArrayShape: TypeAlias = Sequence[int]
CondDict: TypeAlias = Mapping[str, Array]
Metrics: TypeAlias = clu_metrics.Collection
ShapeDict: TypeAlias = Mapping[str, Any]  # may be nested
PyTree: TypeAlias = Any
VariableDict: TypeAlias = trainers.VariableDict
StochasticInterpolantLossFn: TypeAlias = losses.StochasticInterpolantLossFn


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
class StochasticInterpolantModel(models.BaseModel):
  """Training a flow-based model for distribution matching.

  Attributes:
    input_shape: The tensor shape of a single sample (i.e. without any batch
      dimension).
    flow_model: The flax module to use for the flow. The required interface for
      its `__call__` function is specified by `FlowFlaxModule`.
    interpolant: The interpolant to use for the training.
    noising_process_interp: A Callable that samples noise from the noising
      process for the interpolant.
    noising_process_flow: A Callable that samples noise from the noising process
      for computing the time derivative of the interpolant.
    loss_stochastic_interpolant: A Callable that computes the loss for the
      stochastic interpolant.
    use_same_noise_rng: Whether to use the same random number generator for
      sampling the noise for the interpolant and the flow. If False, we use
      different random number generators for each.
    time_sampling: A Callable that samples the interpolation times at training.
    num_eval_cases_per_lvl: int = 1
    min_eval_time_lvl: Minimum time at which the flow is sampled at evaluation.
      This should be close to 0.
    max_eval_time_lvl: Maximum time at which the flow is sampled at evaluation.
      This should be close to 1.
    num_eval_time_levels: Number of times at which the flow will be sampled for
      each trajectory between x_0 and x_1.
    weighted_norm: The norm to use for the loss, if None we use euclidean norm,
      otherwise we use weighted norm.
  """

  input_shape: tuple[int, ...]
  flow_model: nn.Module
  interpolant: interpolants.StochasticInterpolant
  # TODO: refactor the noising process it is performed outside the
  # loss function.
  noising_process_interp: Callable[[Array, ArrayShape], Array] = (
      jax.random.normal
  )
  noising_process_flow: Callable[[Array, ArrayShape], Array] = jax.random.normal
  loss_stochastic_interpolant: StochasticInterpolantLossFn
  time_sampling: Callable[[Array, tuple[int, ...]], Array] = functools.partial(
      jax.random.uniform, dtype=jnp.float32
  )

  use_same_noise_rng: bool = True
  min_train_time: float = 1e-4  # This should be close to 0.
  max_train_time: float = 1.0 - 1e-4  # It should be close to 1.

  num_eval_cases_per_lvl: int = 1
  min_eval_time_lvl: float = 1e-4  # This should be close to 0.
  max_eval_time_lvl: float = 1.0 - 1e-4  # It should be close to 1.
  num_eval_time_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
    # TODO: Add a dtype object to ensure consistency of types.
    x = jnp.ones((1,) + self.input_shape)

    return self.flow_model.init(
        rng, x=x, sigma=jnp.ones((1,)), is_training=False
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
    time_sample_rng, dropout_rng, noise_rng, noise_flow_rng = jax.random.split(
        rng, num=4
    )

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    # Interpolation between x_0 and x_1 (check the interpolant)
    # Check the dimensions here.
    noise_interp = self.noising_process_interp(noise_rng, batch["x_0"].shape)
    if self.use_same_noise_rng:
      noise_flow_rng = noise_rng
    noise_flow = self.noising_process_flow(noise_flow_rng, batch["x_0"].shape)
    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp)

    v_t = self.flow_model.apply(
        {"params": params},
        x=x_t,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng},
    )

    loss = self.loss_stochastic_interpolant(
        v_t,
        time,
        batch["x_0"],
        batch["x_1"],
        noise_flow,
        self.interpolant,
    )
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
    choice_rng, noise_rng = jax.random.split(rng)
    # Shape: (num_eval_time_levels, num_eval_cases_per_lvl, *input_shape)
    x_0 = jax.random.choice(
        key=choice_rng,
        a=batch["x_0"],
        shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
    )

    x_1 = jax.random.choice(
        key=choice_rng,
        a=batch["x_1"],
        shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
    )

    noise = self.noising_process_interp(noise_rng, x_1.shape)

    # The intervals for evaluation of time are linear.
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_t = self.interpolant(time_eval, x_0, x_1, noise)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None), out_axes=1)(x_t, time_eval)

    # Tile the time to have the same shape as the batch and the number of
    # evaluation cases.
    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))

    # Eq. (1) in [1].
    int_losses = jax.vmap(
        self.loss_stochastic_interpolant, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise, self.interpolant)

    eval_losses = {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

    return eval_losses

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):
    """Returns the inference flow function."""

    def _flow(x: Array, time: float | Array) -> Array:
      # This is a wrapper to vectorize time if it is a float.
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolantCRPSModel(models.BaseModel):
  """Training a flow-based model for distribution matching.

  Attributes:
    input_shape: The tensor shape of a single sample (i.e. without any batch
      dimension).
    flow_model: The flax module to use for the flow. The required interface for
      its `__call__` function is specified by `FlowFlaxModule`.
    interpolant: The interpolant to use for the training.
    noising_process_interp: A Callable that samples noise from the noising
      process for the interpolant.
    noising_process_flow: A Callable that samples noise from the noising process
      for computing the time derivative of the interpolant.
    loss_stochastic_interpolant: A Callable that computes the loss for the
      stochastic interpolant.
    use_same_noise_rng: Whether to use the same random number generator for
      sampling the noise for the interpolant and the flow. If False, we use
      different random number generators for each.
    time_sampling: A Callable that samples the interpolation times at training.
    num_eval_cases_per_lvl: int = 1
    min_eval_time_lvl: Minimum time at which the flow is sampled at evaluation.
      This should be close to 0.
    max_eval_time_lvl: Maximum time at which the flow is sampled at evaluation.
      This should be close to 1.
    num_eval_time_levels: Number of times at which the flow will be sampled for
      each trajectory between x_0 and x_1.
    weighted_norm: The norm to use for the loss, if None we use euclidean norm,
      otherwise we use weighted norm.
  """

  input_shape: tuple[int, ...]
  flow_model: nn.Module
  interpolant: interpolants.StochasticInterpolant
  # TODO: refactor the noising process it is performed outside the
  # loss function.
  noising_process_interp: Callable[[Array, ArrayShape], Array] = (
      jax.random.normal
  )
  noising_process_flow: Callable[[Array, ArrayShape], Array] = jax.random.normal
  loss_stochastic_interpolant: StochasticInterpolantLossFn
  time_sampling: Callable[[Array, tuple[int, ...]], Array] = functools.partial(
      jax.random.uniform, dtype=jnp.float32
  )

  use_same_noise_rng: bool = False
  min_train_time: float = 1e-4  # This should be close to 0.
  max_train_time: float = 1.0 - 1e-4  # It should be close to 1.

  num_eval_cases_per_lvl: int = 1
  min_eval_time_lvl: float = 1e-4  # This should be close to 0.
  max_eval_time_lvl: float = 1.0 - 1e-4  # It should be close to 1.
  num_eval_time_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
    # TODO: Add a dtype object to ensure consistency of types.
    x = jnp.ones((1,) + self.input_shape)

    return self.flow_model.init(
        rng, x=x, sigma=jnp.ones((1,)), is_training=False
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
    time_sample_rng, dropout_rng, noise_rng, noise_rng_2, noise_flow_rng = (
        jax.random.split(rng, num=5)
    )

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    # Interpolation between x_0 and x_1 (check the interpolant)
    # Check the dimensions here.
    noise_interp_1 = self.noising_process_interp(noise_rng, batch["x_0"].shape)
    noise_interp_2 = self.noising_process_interp(
        noise_rng_2, batch["x_1"].shape
    )
    if self.use_same_noise_rng:
      noise_flow_rng = noise_rng
    noise_flow = self.noising_process_flow(noise_flow_rng, batch["x_0"].shape)

    # We prepate two samples for the CRPS loss.
    # TODO: Add a feature of adding more samples to CRPS loss.
    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp_1)
    x_t_2 = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp_2)

    v_t = self.flow_model.apply(
        {"params": params},
        x=x_t,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng},
    )
    v_t_2 = self.flow_model.apply(
        {"params": params},
        x=x_t_2,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng},
    )

    # Compute the reference flow.
    v_ref = self.interpolant.calculate_time_derivative_interpolant(
        time, batch["x_0"], batch["x_1"], noise_flow
    )

    # Compute CRPS loss
    loss = 0.5 * jnp.mean(
        jnp.abs(v_t - v_ref) + jnp.abs(v_t_2 - v_ref)
    ) - 0.25 * jnp.mean(jnp.abs(v_t - v_t_2))

    metric = dict(loss=loss)
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Compute metrics on an eval batch using the L^2 loss.

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
    choice_rng, noise_rng = jax.random.split(rng)
    # Shape: (num_eval_time_levels, num_eval_cases_per_lvl, *input_shape)
    x_0 = jax.random.choice(
        key=choice_rng,
        a=batch["x_0"],
        shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
    )

    x_1 = jax.random.choice(
        key=choice_rng,
        a=batch["x_1"],
        shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
    )

    noise = self.noising_process_interp(noise_rng, x_1.shape)

    # The intervals for evaluation of time are linear.
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_t = self.interpolant(time_eval, x_0, x_1, noise)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None), out_axes=1)(x_t, time_eval)

    # Tile the time to have the same shape as the batch and the number of
    # evaluation cases.
    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))

    # Eq. (1) in [1].
    int_losses = jax.vmap(
        self.loss_stochastic_interpolant, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise, self.interpolant)

    eval_losses = {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

    return eval_losses

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):
    """Returns the inference flow function."""

    def _flow(x: Array, time: float | Array) -> Array:
      # This is a wrapper to vectorize time if it is a float.
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolantFlowScoreModel(models.BaseModel):
  """Training a flow-based model for both flow and score.

  Attributes:
    input_shape: The tensor shape of a single sample (i.e. without any batch
      dimension).
    flow_score_model: The flax module to use for the flow and score. The
      required interface for its `__call__` function is specified by
      `FlowScoreFlaxModule`.
    interpolant: The interpolant to use for the training.
    noising_process: A Callable that samples noise from the noising process.
    loss_stochastic_interpolant: A Callable that computes the loss for the
      stochastic interpolant.
    time_sampling: A Callable that samples the interpolation times at training.
    num_eval_cases_per_lvl: int = 1
    min_eval_time_lvl: Minimum time at which the flow is sampled at evaluation.
      This should be close to 0.
    max_eval_time_lvl: Maximum time at which the flow is sampled at evaluation.
      This should be close to 1.
    num_eval_time_levels: Number of times at which the flow will be sampled for
      each trajectory between x_0 and x_1.
    weighted_norm: The norm to use for the loss, if None we use euclidean norm,
      otherwise we use weighted norm.
  """

  input_shape: tuple[int, ...]
  flow_score_model: nn.Module
  interpolant: interpolants.StochasticInterpolant
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  loss_stochastic_interpolant_flow: StochasticInterpolantLossFn
  loss_stochastic_interpolant_score: StochasticInterpolantLossFn
  time_sampling: Callable[[Array, tuple[int, ...]], Array] = functools.partial(
      jax.random.uniform, dtype=jnp.float32
  )

  min_train_time: float = 1e-4  # This should be close to 0.
  max_train_time: float = 1.0 - 1e-4  # It should be close to 1.

  num_eval_cases_per_lvl: int = 1
  min_eval_time_lvl: float = 1e-4  # This should be close to 0.
  max_eval_time_lvl: float = 1.0 - 1e-4  # It should be close to 1.
  num_eval_time_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
    # TODO: Add a dtype object to ensure consistency of types.
    x = jnp.ones((1,) + self.input_shape)

    return self.flow_score_model.init(
        rng, x=x, sigma=jnp.ones((1,)), is_training=False
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
      mutables: The mutable (non-differentiated) parameters of the flow-score
        model (e.g. batch stats).

    Returns:
      The loss value and a tuple of training metric and mutables.
    """
    batch_size = len(batch["x_0"])
    time_sample_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    # Interpolation between x_0 and x_1 (check the interpolant)
    # Check the dimensions here.
    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise)

    # Here we assume that the flow model has two outputs, one for the flow and
    # one for the score (or the denoised sample if using a denoising loss).
    v_t, s_t = self.flow_score_model.apply(
        {"params": params},
        x=x_t,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng},
    )

    loss_flow = self.loss_stochastic_interpolant_flow(
        v_t,
        time,
        batch["x_0"],
        batch["x_1"],
        noise,
        self.interpolant,
    )
    loss_score = self.loss_stochastic_interpolant_score(
        s_t,
        time,
        batch["x_0"],
        batch["x_1"],
        noise,
        self.interpolant,
    )

    loss = loss_flow + loss_score
    metric = dict(loss=loss, loss_flow=loss_flow, loss_score=loss_score)
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes metrics on an eval batch.

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
    choice_rng, noise_rng = jax.random.split(rng)
    # Shape: (num_eval_time_levels, num_eval_cases_per_lvl, *input_shape)
    x_0 = jax.random.choice(
        key=choice_rng,
        a=batch["x_0"],
        shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
    )

    x_1 = jax.random.choice(
        key=choice_rng,
        a=batch["x_1"],
        shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
    )

    noise = self.noising_process(noise_rng, x_1.shape)

    # The intervals for evaluation of time are linear.
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_t = self.interpolant(time_eval, x_0, x_1, noise)
    flow_fn = self.inference_fn(variables, self.flow_score_model)
    v_t, s_t = jax.vmap(flow_fn, in_axes=(1, None), out_axes=1)(x_t, time_eval)

    # Tile the time to have the same shape as the batch and the number of
    # evaluation cases.
    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))

    # Eq. (1) in [1].
    int_losses_flow = jax.vmap(
        self.loss_stochastic_interpolant_flow, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise, self.interpolant)

    int_losses_score = jax.vmap(
        self.loss_stochastic_interpolant_score, in_axes=(0, 0, 0, 0, 0, None)
    )(s_t, time_tiled, x_0, x_1, noise, self.interpolant)

    # We sum the losses to get the total loss.
    int_losses = int_losses_flow + int_losses_score

    eval_losse = {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

    return eval_losse

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):
    """Returns the inference flow function."""

    def _flow(x: Array, time: float | Array) -> Array:
      # This is a wrapper to vectorize time if it is a float.
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalStochasticInterpolantModel(StochasticInterpolantModel):
  """Training a conditional flow-based model for distribution matching.

  Attributes:
    cond_shape: Dictionary containing the keys and shapes of the conditional
      input.
  """

  cond_shape: ShapeDict | None = None

  def initialize(self, rng: Array) -> models.PyTree:
    x = jnp.ones((1,) + self.input_shape)
    cond = cond_sample_from_shape(self.cond_shape, batch_dims=(1,))

    return self.flow_model.init(  # add conditional input here.
        rng, x=x, sigma=jnp.ones((1,)), cond=cond, is_training=False
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
    time_sample_rng, dropout_rng, noise_rng, noise_flow_rng = jax.random.split(
        rng, num=4
    )

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    # Interpolation between x_0 and x_1 (check the interpolant)
    # Check the dimensions here.
    noise_interp = self.noising_process_interp(noise_rng, batch["x_0"].shape)
    if self.use_same_noise_rng:
      noise_flow_rng = noise_rng
    noise_flow = self.noising_process_flow(noise_flow_rng, batch["x_0"].shape)
    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp)

    # Extracting the conditioning.
    if self.cond_shape is not None:
      cond = {key: batch[key] for key in self.cond_shape.keys()}
    else:
      cond = None

    v_t = self.flow_model.apply(
        {"params": params},
        x=x_t,
        sigma=time,
        cond=cond,
        is_training=True,
        rngs={"dropout": dropout_rng},
    )

    # Eq. (1) in [1], but with a possibly weighted norm.
    loss = self.loss_stochastic_interpolant(
        v_t,
        time,
        batch["x_0"],
        batch["x_1"],
        noise_flow,
        self.interpolant,
    )

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
    choice_rng, noise_rng = jax.random.split(rng)

    # Shuffling the batch (but keeping the pairs together).
    batch_reorg = jax.tree.map(
        lambda x: jax.random.choice(
            key=choice_rng,
            a=x,
            shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
        ),
        batch,
    )

    # The intervals for evaluation of time are linear.
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_0 = batch_reorg["x_0"]
    x_1 = batch_reorg["x_1"]
    noise = self.noising_process_interp(noise_rng, x_1.shape)

    # Extracting the conditioning.
    if self.cond_shape is not None:
      cond = {key: batch_reorg[key] for key in self.cond_shape.keys()}
    else:
      cond = None

    x_t = self.interpolant(time_eval, x_0, x_1, noise)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None, 1), out_axes=1)(
        x_t, time_eval, cond
    )

    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))
    int_losses = jax.vmap(
        self.loss_stochastic_interpolant, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise, self.interpolant)

    eval_losses = {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

    return eval_losses

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):
    """Returns the inference conditional flow function."""

    def _flow(
        x: Array, time: float | Array, cond: CondDict | None = None
    ) -> Array:
      # This is a wrapper to vectorize time if it is a float.
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(
          variables, x=x, sigma=time, cond=cond, is_training=False
      )

    return _flow


class RescaledUnet(backbones.UNet):
  """Rescaled flow model.

  Attributes:
    time_rescale: Factor for rescaling the time, which normally is in [0, 1] and
      the input to the UNet which is a noise level, which has a much wider range
      of values.
  """

  time_rescale: float = 1000.0

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Runs rescaled Unet with noise input."""
    if x.shape[-1] != self.out_channels:
      raise ValueError(
          f"Number of channels in the input ({x.shape[-1]}) must "
          "match the number of channels in the output "
          f"{self.out_channels})."
      )

    if sigma.ndim < 1:
      sigma = jnp.broadcast_to(sigma, (x.shape[0],))

    if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
          "sigma must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
      )

    time = sigma * self.time_rescale

    f_x = super().__call__(x, time, cond, is_training=is_training)

    return f_x


class RescaledUnet3d(unets3d.UNet3d):
  """Rescaled flow model for the 3D Unet.

  Attributes:
    time_rescale: Factor for rescaling the time, which normally is in [0, 1] and
      the input to the UNet which is a noise level, which has a much wider range
      of values.
  """

  time_rescale: float = 1000.0

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Runs rescaled Unet3d with noise input."""
    if x.shape[-1] != self.out_channels:
      raise ValueError(
          f"Number of channels in the input ({x.shape[-1]}) must "
          "match the number of channels in the output "
          f"{self.out_channels})."
      )

    if sigma.ndim < 1:
      sigma = jnp.broadcast_to(sigma, (x.shape[0],))

    if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
          "sigma must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
      )

    time = sigma * self.time_rescale

    f_x = super().__call__(x, time, cond, is_training=is_training)

    return f_x
