# Copyright 2024 The swirl_dynamics Authors.
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

r"""Training a flow model for Rectified flow.

Rectified flow [1] seeks to train a flow model
  dX_t = v(X_t, t) dt,
which links two distributions $\mu_0$ and $\mu_1$, such that the initial and
final conditions are drawn from such distributions, namely, $X_0 \sim \mu_0$
and $X_1 \sim \mu_1$.

Rectified flow achieves this by by minimizing the loss
  $E_{t ~ U[\eps, 1-\eps]} E_{X_0, X_1} [ \| X_1 - X_0 - v(X_t, t)  \|^2]$,
where $X_t = t * X_1 + (1 - t) X_0$, for $t \in [0,1]$.
Basically, the loss tries to obtain a straight line between x_0 and x_1, i.e.,
$X_1 = X_0 + \Delta t v_0$, for $\Delta t = 1$. However, one can
evaluate the speed v at any point in the trajectory, and if v is constant it
should yield $X_1 = X_0 + \Delta t v_t$ for $t \in [0, 1]$, which is exactly the
expression that the loss is pursuing.

References:
[1] Xingchao Liu, Chengyue Gong and Qiang Liu. "Flow Straight and Fast:
  Learning to Generate and Transfer Data with Rectified Flow" NeurIPS 2022,
  Workshop on Score-Based Methods.
"""

from collections.abc import Callable, Mapping
import dataclasses
import functools
from typing import Any, ClassVar, Protocol

from clu import metrics as clu_metrics
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers

Array = jax.Array
CondDict = Mapping[str, Array]
Metrics = clu_metrics.Collection
ShapeDict = Mapping[str, Any]  # may be nested
PyTree = Any
VariableDict = trainers.VariableDict


class FlowFlaxModule(Protocol):
  """Expected interface of the flax module compatible with `ReFlowModel`.

  NOTE: This protocol is for reference only and not statically checked.
  """

  def __call__(self, x: Array, t: Array, is_training: bool) -> Array:
    ...


# TODO add a test function.
def lognormal_sampler(
    mean: float = 0.0, std: float = 1.0
) -> Callable[[Array, tuple[int, ...]], Array]:
  return lambda x, y: jax.lax.logistic(
      std * jax.random.uniform(x, y, dtype=jnp.float32) + mean
  )


def cond_sample_from_shape(
    shape: ShapeDict | None, batch_dims: tuple[int, ...] = (1,)
) -> PyTree:
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
class ReFlowModel(models.BaseModel):
  """Training a flow-based model for distribution matching.

  Attributes:
    input_shape: The tensor shape of a single sample (i.e. without any batch
      dimension).
    flow_model: The flax module to use for the flow. The required interface for
      its `__call__` function is specified by `FlowFlaxModule`.
    time_sampling: A Callable that samples the interpolation times at training.
    num_eval_cases_per_lvl: int = 1
    min_eval_time_lvl: Minimum time at which the flow is sampled at evaluation.
      This should be close to 0.
    max_eval_time_lvl: Maximum time at which the flow is sampled at evaluation.
      This should be close to 1.
    num_eval_time_levels: Number of times at which the flow will be sampled for
      each trajectory between x_0 and x_1.
  """

  input_shape: tuple[int, ...]
  flow_model: nn.Module
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
    time_sample_rng, dropout_rng = jax.random.split(rng, num=2)

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))

    # Interpolation between x_0 and x_1.
    x_t = vmap_mult(batch["x_1"], time) + vmap_mult(batch["x_0"], 1 - time)

    v_t = self.flow_model.apply(
        {"params": params},
        x=x_t,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng},
    )
    # Eq. (1) in [1].
    loss = jnp.mean(jnp.square((batch["x_1"] - batch["x_0"]) - v_t))
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
    choice_rng, _ = jax.random.split(rng)
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

    # The intervals for evaluation of time are linear.
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))
    x_t = vmap_mult(x_1, time_eval) + vmap_mult(x_0, 1 - time_eval)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None), out_axes=1)(x_t, time_eval)

    # Eq. (1) in [1]. (by default in_axes=0 and out_axes=0 in vmap)
    int_losses = jax.vmap(jnp.mean)(jnp.square((x_1 - x_0 - v_t)))
    eval_losses = {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

    return eval_losses

  @staticmethod
  def inference_fn(variables: models.PyTree, flow_model: nn.Module):
    """Returns the inference flow function."""

    def _flow(x: Array, time: float | Array) -> Array:
      # This is a wrapper to vectorize time if it is a float.
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalReFlowModel(ReFlowModel):
  """Training a conditional flow-based model for distribution matching.

  Attributes:
    cond_shape: Shape of the conditional input.
  """
  cond_shape: ShapeDict | None = None

  def initialize(self, rng: Array):
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
    time_sample_rng, dropout_rng = jax.random.split(rng, num=2)

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    # Extracting the conditioning.
    cond = {
        "channel:mean": batch["channel:mean"],
        "channel:std": batch["channel:std"],
    }

    vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))

    # Interpolation between x_0 and x_1.
    x_t = vmap_mult(batch["x_1"], time) + vmap_mult(batch["x_0"], 1 - time)

    v_t = self.flow_model.apply(
        {"params": params},
        x=x_t,
        sigma=time,
        cond=cond,
        is_training=True,
        rngs={"dropout": dropout_rng},
    )
    # Eq. (1) in [1].
    loss = jnp.mean(jnp.square((batch["x_1"] - batch["x_0"]) - v_t))
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
    choice_rng, _ = jax.random.split(rng)

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
    cond = {
        "channel:mean": batch_reorg["channel:mean"],
        "channel:std": batch_reorg["channel:std"],
    }

    vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))
    x_t = vmap_mult(x_1, time_eval) + vmap_mult(x_0, 1 - time_eval)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None, 1), out_axes=1)(
        x_t, time_eval, cond
    )

    # Eq. (1) in [1]. (by default in_axes=0 and out_axes=0 in vmap)
    int_losses = jax.vmap(jnp.mean)(jnp.square((x_1 - x_0 - v_t)))
    eval_losses = {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

    return eval_losses

  @staticmethod
  def inference_fn(variables: models.PyTree, flow_model: nn.Module):
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


class RescaledUnet(unets.UNet):
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
