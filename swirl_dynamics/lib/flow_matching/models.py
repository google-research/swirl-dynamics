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

r"""Training a flow model for stochastic interpolants.

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
from swirl_dynamics.lib.diffusion import denoising_models
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.lib.diffusion import unets3d
from swirl_dynamics.lib.flow_matching import interpolants
from swirl_dynamics.lib.flow_matching import losses
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers

Array: TypeAlias = jax.Array
ArrayShape: TypeAlias = Sequence[int]
CondDict: TypeAlias = Mapping[str, Array]
Metrics: TypeAlias = clu_metrics.Collection
ShapeDict: TypeAlias = Mapping[str, Any]
PyTree: TypeAlias = Any
VariableDict: TypeAlias = trainers.VariableDict
StochasticInterpolantLossFn: TypeAlias = losses.StochasticInterpolantLossFn

cond_sample_from_shape = denoising_models.cond_sample_from_shape


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolantModel(models.BaseModel):
  """Training a flow-based model for distribution matching."""

  input_shape: tuple[int, ...]
  flow_model: nn.Module
  interpolant: interpolants.StochasticInterpolant
  noising_process_interp: Callable[[Array, ArrayShape], Array] = (
      jax.random.normal
  )
  noising_process_flow: Callable[[Array, ArrayShape], Array] = jax.random.normal
  loss_stochastic_interpolant: StochasticInterpolantLossFn
  time_sampling: Callable[[Array, tuple[int, ...]], Array] = functools.partial(
      jax.random.uniform, dtype=jnp.float32
  )

  use_same_noise_rng: bool = True
  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4

  num_eval_cases_per_lvl: int = 1
  min_eval_time_lvl: float = 1e-4
  max_eval_time_lvl: float = 1.0 - 1e-4
  num_eval_time_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
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
    """Computes the flow matching loss on a training batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng, dropout_rng, noise_rng, noise_flow_rng = jax.random.split(
        rng, num=4
    )

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    noise_interp = self.noising_process_interp(noise_rng, batch["x_0"].shape)
    if self.use_same_noise_rng:
      noise_flow_rng = noise_rng
    noise_flow = self.noising_process_flow(noise_flow_rng, batch["x_0"].shape)
    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp)

    v_t, mutables = self.flow_model.apply(
        {"params": params, **mutables},
        x=x_t,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng},
        mutable=list(mutables.keys()),
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
    """Computes metrics on an eval batch."""
    choice_rng, noise_rng = jax.random.split(rng)
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
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_t = self.interpolant(time_eval, x_0, x_1, noise)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None), out_axes=1)(x_t, time_eval)
    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))
    int_losses = jax.vmap(
        self.loss_stochastic_interpolant, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise, self.interpolant)

    return {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):  # pyrefly: ignore[bad-override]
    """Returns the inference flow function."""

    def _flow(x: Array, time: float | Array) -> Array:
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)  # pyrefly: ignore[bad-return]

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolantCRPSModel(models.BaseModel):
  """Training a flow-based model for distribution matching with CRPS loss."""

  input_shape: tuple[int, ...]
  flow_model: nn.Module
  interpolant: interpolants.StochasticInterpolant
  noising_process_interp: Callable[[Array, ArrayShape], Array] = (
      jax.random.normal
  )
  noising_process_flow: Callable[[Array, ArrayShape], Array] = jax.random.normal
  loss_stochastic_interpolant: StochasticInterpolantLossFn
  time_sampling: Callable[[Array, tuple[int, ...]], Array] = functools.partial(
      jax.random.uniform, dtype=jnp.float32
  )

  use_same_noise_rng: bool = False
  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4

  num_eval_cases_per_lvl: int = 1
  min_eval_time_lvl: float = 1e-4
  max_eval_time_lvl: float = 1.0 - 1e-4
  num_eval_time_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
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
    """Computes the flow matching CRPS loss on a training batch."""
    batch_size = len(batch["x_0"])
    (
        time_sample_rng,
        dropout_rng,
        noise_rng,
        noise_rng_2,
        noise_flow_rng_1,
        noise_flow_rng_2,
    ) = jax.random.split(rng, num=6)
    dropout_rng_1, dropout_rng_2 = jax.random.split(dropout_rng)

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    noise_interp_1 = self.noising_process_interp(noise_rng, batch["x_0"].shape)
    noise_interp_2 = self.noising_process_interp(
        noise_rng_2, batch["x_1"].shape
    )
    noise_flow_1 = (
        noise_interp_1
        if self.use_same_noise_rng
        else self.noising_process_flow(noise_flow_rng_1, batch["x_0"].shape)
    )
    noise_flow_2 = (
        noise_interp_2
        if self.use_same_noise_rng
        else self.noising_process_flow(noise_flow_rng_2, batch["x_1"].shape)
    )

    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp_1)
    x_t_2 = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp_2)

    v_t, mutables = self.flow_model.apply(
        {"params": params, **mutables},
        x=x_t,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng_1},
        mutable=list(mutables.keys()),
    )
    v_t_2, mutables = self.flow_model.apply(
        {"params": params, **mutables},
        x=x_t_2,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng_2},
        mutable=list(mutables.keys()),
    )

    v_ref_1 = self.interpolant.calculate_time_derivative_interpolant(
        time, batch["x_0"], batch["x_1"], noise_flow_1
    )
    v_ref_2 = self.interpolant.calculate_time_derivative_interpolant(
        time, batch["x_0"], batch["x_1"], noise_flow_2
    )

    loss = 0.5 * jnp.mean(
        jnp.abs(v_t - v_ref_1) + jnp.abs(v_t_2 - v_ref_2)
    ) - 0.5 * jnp.mean(jnp.abs(v_t - v_t_2))

    metric = dict(loss=loss)
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes metrics on an eval batch using the L^2 loss."""
    choice_rng, noise_rng = jax.random.split(rng)
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
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_t = self.interpolant(time_eval, x_0, x_1, noise)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None), out_axes=1)(x_t, time_eval)
    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))
    int_losses = jax.vmap(
        self.loss_stochastic_interpolant, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise, self.interpolant)

    return {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):  # pyrefly: ignore[bad-override]
    """Returns the inference flow function."""

    def _flow(x: Array, time: float | Array) -> Array:
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)  # pyrefly: ignore[bad-return]

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolantFlowScoreModel(models.BaseModel):
  """Training a flow-based model for both flow and score."""

  input_shape: tuple[int, ...]
  flow_score_model: nn.Module
  interpolant: interpolants.StochasticInterpolant
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  loss_stochastic_interpolant_flow: StochasticInterpolantLossFn
  loss_stochastic_interpolant_score: StochasticInterpolantLossFn
  time_sampling: Callable[[Array, tuple[int, ...]], Array] = functools.partial(
      jax.random.uniform, dtype=jnp.float32
  )

  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4

  num_eval_cases_per_lvl: int = 1
  min_eval_time_lvl: float = 1e-4
  max_eval_time_lvl: float = 1.0 - 1e-4
  num_eval_time_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
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
    """Computes the flow-score matching loss on a training batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng, dropout_rng, noise_rng = jax.random.split(rng, num=3)

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise)

    (v_t, s_t), mutables = self.flow_score_model.apply(
        {"params": params, **mutables},
        x=x_t,
        sigma=time,
        is_training=True,
        rngs={"dropout": dropout_rng},
        mutable=list(mutables.keys()),
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
    """Computes metrics on an eval batch."""
    choice_rng, noise_rng = jax.random.split(rng)
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
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_t = self.interpolant(time_eval, x_0, x_1, noise)
    flow_fn = self.inference_fn(variables, self.flow_score_model)
    v_t, s_t = jax.vmap(flow_fn, in_axes=(1, None), out_axes=1)(x_t, time_eval)
    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))
    int_losses_flow = jax.vmap(
        self.loss_stochastic_interpolant_flow, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise, self.interpolant)
    int_losses_score = jax.vmap(
        self.loss_stochastic_interpolant_score, in_axes=(0, 0, 0, 0, 0, None)
    )(s_t, time_tiled, x_0, x_1, noise, self.interpolant)
    int_losses = int_losses_flow + int_losses_score
    return {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):  # pyrefly: ignore[bad-override]
    """Returns the inference flow function."""

    def _flow(x: Array, time: float | Array) -> Array:
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)  # pyrefly: ignore[bad-return]

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalStochasticInterpolantModel(StochasticInterpolantModel):
  """Training a conditional flow-based model for distribution matching."""

  cond_shape: ShapeDict | None = None

  def initialize(self, rng: Array) -> models.PyTree:
    x = jnp.ones((1,) + self.input_shape)
    cond = cond_sample_from_shape(self.cond_shape, batch_dims=(1,))
    return self.flow_model.init(
        rng, x=x, sigma=jnp.ones((1,)), cond=cond, is_training=False
    )

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the conditional flow matching loss on a training batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng, dropout_rng, noise_rng, noise_flow_rng = jax.random.split(
        rng, num=4
    )

    time_range = self.max_train_time - self.min_train_time
    time = (
        time_range * self.time_sampling(time_sample_rng, (batch_size,))
        + self.min_train_time
    )

    noise_interp = self.noising_process_interp(noise_rng, batch["x_0"].shape)
    if self.use_same_noise_rng:
      noise_flow_rng = noise_rng
    noise_flow = self.noising_process_flow(noise_flow_rng, batch["x_0"].shape)
    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp)

    if self.cond_shape is not None:
      cond = {key: batch[key] for key in self.cond_shape.keys()}
    else:
      cond = None

    v_t, mutables = self.flow_model.apply(
        {"params": params, **mutables},
        x=x_t,
        sigma=time,
        cond=cond,
        is_training=True,
        rngs={"dropout": dropout_rng},
        mutable=list(mutables.keys()),
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
    """Computes metrics on an eval batch."""
    choice_rng, noise_rng = jax.random.split(rng)
    batch_reorg = jax.tree.map(
        lambda x: jax.random.choice(
            key=choice_rng,
            a=x,
            shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
        ),
        batch,
    )
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_0 = batch_reorg["x_0"]
    x_1 = batch_reorg["x_1"]
    noise = self.noising_process_interp(noise_rng, x_1.shape)

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

    return {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):
    """Returns the inference conditional flow function."""

    def _flow(
        x: Array, time: float | Array, cond: CondDict | None = None
    ) -> Array:
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(  # pyrefly: ignore[bad-return]
          variables, x=x, sigma=time, cond=cond, is_training=False
      )

    return _flow


class TimeRescaled(nn.Module):
  """Wraps a flow network and rescales the time input."""

  network: nn.Module
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
    """Runs wrapped network with rescaled time input."""
    if sigma.ndim < 1:
      sigma = jnp.broadcast_to(sigma, (x.shape[0],))
    if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
          "sigma must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
      )
    time = sigma * self.time_rescale
    return self.network(x, time, cond, is_training=is_training)


class RescaledUnet(unets.UNet):
  """Rescaled flow model for the 2D UNet."""

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
    """Runs rescaled UNet with noise input."""
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
    return super().__call__(x, time, cond, is_training=is_training)


class RescaledUnet3d(unets3d.UNet3d):
  """Rescaled flow model for the 3D Unet."""

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
    return super().__call__(x, time, cond, is_training=is_training)
