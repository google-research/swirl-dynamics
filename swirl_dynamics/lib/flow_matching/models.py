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

r"""Flow-matching models for the stochastic interpolants framework.

References:
  [1] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.
      Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.
"""

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Any, ClassVar

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.diffusion import denoising_models
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.lib.diffusion import unets3d
from swirl_dynamics.lib.flow_matching import interpolants
from swirl_dynamics.lib.flow_matching import losses
from swirl_dynamics.templates import models

Array = jax.Array
ArrayShape = Sequence[int]
CondDict = denoising_models.CondDict
ShapeDict = denoising_models.ShapeDict
PyTree = Any
StochasticInterpolantLossFn = losses.StochasticInterpolantLossFn

cond_sample_from_shape = denoising_models.cond_sample_from_shape


def _rescale_sigma(x: Array, sigma: Array, time_rescale: float) -> Array:
  """Broadcasts scalar `sigma`, validates 1D batch alignment, and rescales."""
  if sigma.ndim < 1:
    sigma = jnp.broadcast_to(sigma, (x.shape[0],))
  if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
    raise ValueError(
        "sigma must be 1D and have the same leading (batch) dimension as x"
        f" ({x.shape[0]})!"
    )
  return sigma * time_rescale


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolantModel(models.BaseModel):
  """Trains a flow-based model for distribution matching via interpolants.

  Attributes:
    input_shape: Tensor shape of a single sample (excluding batch dimension).
    flow_model: Flax module parameterizing the velocity field.
    interpolant: Stochastic interpolant linking `x_0` and `x_1`.
    noising_process_interp: Callable sampling noise for the interpolant state.
    noising_process_flow: Callable sampling noise for the target velocity.
    loss_stochastic_interpolant: Loss callable comparing predicted and target
      quantities.
    time_sampling: Callable sampling training interpolation times in `[0, 1]`.
    use_same_noise_rng: Whether to share the noise PRNG key between the
      interpolant state and target time derivative.
    min_train_time: Lower bound on sampled training times.
    max_train_time: Upper bound on sampled training times.
    num_eval_cases_per_lvl: Number of evaluation samples per time level.
    min_eval_time_lvl: Minimum evaluation time level.
    max_eval_time_lvl: Maximum evaluation time level.
    num_eval_time_levels: Number of linearly spaced evaluation time levels.
  """

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
    """Initializes variables of `flow_model`."""
    x = jnp.ones((1,) + self.input_shape)
    return self.flow_model.init(
        rng, x=x, sigma=jnp.ones((1,)), is_training=False
    )

  def _extract_cond(self, batch: models.BatchType) -> CondDict | None:
    """Extracts optional conditioning inputs from `batch`."""
    del batch
    return None

  def _apply_flow(
      self,
      variables: models.PyTree,
      *,
      x: Array,
      sigma: Array,
      cond: CondDict | None,
      is_training: bool,
      rngs: dict[str, Array] | None = None,
      mutable: Any = False,
  ) -> Any:
    """Applies `flow_model` with or without conditioning kwargs."""
    del cond
    return self.flow_model.apply(
        variables,
        x=x,
        sigma=sigma,
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
    """Computes the stochastic interpolant loss on a training batch.

    Args:
      params: Differentiable parameters of `flow_model`.
      batch: Training batch containing `x_0` and `x_1` arrays with shape
        `(batch, *spatial_dims, channels)`, plus any conditional keys.
      rng: JAX PRNG key.
      mutables: Non-differentiated mutable model state (e.g. `batch_stats`).

    Returns:
      Tuple `(loss, (metrics, updated_mutables))`.
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

    noise_interp = self.noising_process_interp(noise_rng, batch["x_0"].shape)
    if self.use_same_noise_rng:
      noise_flow_rng = noise_rng
    noise_flow = self.noising_process_flow(noise_flow_rng, batch["x_0"].shape)
    x_t = self.interpolant(time, batch["x_0"], batch["x_1"], noise_interp)
    cond = self._extract_cond(batch)

    v_t, mutables = self._apply_flow(
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
    """Computes time-stratified interpolant losses on an evaluation batch.

    Args:
      variables: Complete model variables for `flow_model`.
      batch: Evaluation batch containing `x_0` and `x_1` arrays.
      rng: JAX PRNG key.

    Returns:
      Dictionary of evaluation losses keyed by `time_lvl{i}`.
    """
    choice_rng, noise_rng, noise_flow_rng = jax.random.split(rng, num=3)
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
    noise_interp = self.noising_process_interp(noise_rng, x_1.shape)
    if self.use_same_noise_rng:
      noise_flow_rng = noise_rng
    noise_flow = self.noising_process_flow(noise_flow_rng, x_1.shape)
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_t = self.interpolant(time_eval, x_0, x_1, noise_interp)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None), out_axes=1)(x_t, time_eval)
    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))
    int_losses = jax.vmap(
        self.loss_stochastic_interpolant, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise_flow, self.interpolant)

    return {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):  # pyrefly: ignore[bad-override]
    """Returns the unconditional inference velocity function."""

    def _flow(x: Array, time: float | Array) -> Array:
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)  # pyrefly: ignore[bad-return]

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolantCRPSModel(StochasticInterpolantModel):
  """Trains a stochastic interpolant flow model with symmetric fair-CRPS loss.

  Inherits initialization, `L^2` time-level evaluation, and inference from
  `StochasticInterpolantModel`, while evaluating two independent stochastic
  ensemble draws at a shared `(x_t, v_ref)` during `loss_fn`.
  """

  use_same_noise_rng: bool = False

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the symmetric two-member fair-CRPS loss on a training batch."""
    batch_size = len(batch["x_0"])
    time_sample_rng, dropout_rng, noise_rng, noise_flow_rng = jax.random.split(
        rng, num=4
    )
    dropout_rng_1, dropout_rng_2 = jax.random.split(dropout_rng)

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
    cond = self._extract_cond(batch)

    mutable_keys = list(mutables.keys())
    v_t, mutables_1 = self._apply_flow(
        {"params": params, **mutables},
        x=x_t,
        sigma=time,
        cond=cond,
        is_training=True,
        rngs={"dropout": dropout_rng_1},
        mutable=mutable_keys,
    )
    v_t_2, mutables_2 = self._apply_flow(
        {"params": params, **mutables},
        x=x_t,
        sigma=time,
        cond=cond,
        is_training=True,
        rngs={"dropout": dropout_rng_2},
        mutable=mutable_keys,
    )
    mutables = jax.tree.map(
        lambda a, b: (
            (0.5 * (a + b)).astype(a.dtype)
            if jnp.issubdtype(a.dtype, jnp.floating)
            else a
        ),
        mutables_1,
        mutables_2,
    )

    v_ref = self.interpolant.calculate_time_derivative_interpolant(
        time, batch["x_0"], batch["x_1"], noise_flow
    )

    loss = 0.5 * jnp.mean(
        jnp.abs(v_t - v_ref) + jnp.abs(v_t_2 - v_ref)
    ) - 0.5 * jnp.mean(jnp.abs(v_t - v_t_2))

    metric = dict(loss=loss)
    return loss, (metric, mutables)


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolantFlowScoreModel(models.BaseModel):
  """Trains a joint velocity and score model for stochastic interpolants.

  Attributes:
    input_shape: Tensor shape of a single sample (excluding batch dimension).
    flow_score_model: Flax module returning `(velocity, score)` at `(x, sigma)`.
    interpolant: Stochastic interpolant linking `x_0` and `x_1`.
    noising_process: Callable sampling noise for the interpolant.
    loss_stochastic_interpolant_flow: Loss callable for the velocity output.
    loss_stochastic_interpolant_score: Loss callable for the score/denoiser
      output.
    time_sampling: Callable sampling training interpolation times in `[0, 1]`.
    min_train_time: Lower bound on sampled training times.
    max_train_time: Upper bound on sampled training times.
    num_eval_cases_per_lvl: Number of evaluation samples per time level.
    min_eval_time_lvl: Minimum evaluation time level.
    max_eval_time_lvl: Maximum evaluation time level.
    num_eval_time_levels: Number of linearly spaced evaluation time levels.
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

  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4

  num_eval_cases_per_lvl: int = 1
  min_eval_time_lvl: float = 1e-4
  max_eval_time_lvl: float = 1.0 - 1e-4
  num_eval_time_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
    """Initializes variables of `flow_score_model`."""
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
    """Computes the combined flow and score loss on a training batch."""
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
    """Computes combined flow and score losses across evaluation time levels."""
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
    """Returns the joint `(velocity, score)` inference function."""

    def _flow(x: Array, time: float | Array) -> tuple[Array, Array]:
      if not jnp.shape(jnp.asarray(time)):
        time *= jnp.ones((x.shape[0],))
      return flow_model.apply(variables, x=x, sigma=time, is_training=False)  # pyrefly: ignore[bad-return]

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalStochasticInterpolantModel(StochasticInterpolantModel):
  """Trains a conditional flow-based model for distribution matching.

  Attributes:
    cond_shape: Mapping of conditional input keys to sample shapes (excluding
      batch dimension). If `None`, no conditioning is passed.
  """

  cond_shape: ShapeDict | None = None

  def initialize(self, rng: Array):
    """Initializes variables of `flow_model` with dummy conditional inputs."""
    x = jnp.ones((1,) + self.input_shape)
    cond = cond_sample_from_shape(self.cond_shape, batch_dims=(1,))
    return self.flow_model.init(
        rng, x=x, sigma=jnp.ones((1,)), cond=cond, is_training=False
    )

  def _extract_cond(self, batch: models.BatchType) -> CondDict | None:
    if self.cond_shape is None:
      return None
    return {key: batch[key] for key in self.cond_shape.keys()}

  def _apply_flow(
      self,
      variables: models.PyTree,
      *,
      x: Array,
      sigma: Array,
      cond: CondDict | None,
      is_training: bool,
      rngs: dict[str, Array] | None = None,
      mutable: Any = False,
  ) -> Any:
    return self.flow_model.apply(
        variables,
        x=x,
        sigma=sigma,
        cond=cond,
        is_training=is_training,
        rngs=rngs,
        mutable=mutable,
    )

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes time-stratified conditional interpolant evaluation losses."""
    choice_rng, noise_rng, noise_flow_rng = jax.random.split(rng, num=3)
    rand_idx = jax.random.choice(
        key=choice_rng,
        a=batch["x_0"].shape[0],
        shape=(self.num_eval_time_levels, self.num_eval_cases_per_lvl),
    )
    time_eval = jnp.linspace(
        self.min_eval_time_lvl,
        self.max_eval_time_lvl,
        self.num_eval_time_levels,
    )

    x_0 = jnp.asarray(batch["x_0"])[rand_idx]
    x_1 = jnp.asarray(batch["x_1"])[rand_idx]
    noise_interp = self.noising_process_interp(noise_rng, x_1.shape)
    if self.use_same_noise_rng:
      noise_flow_rng = noise_rng
    noise_flow = self.noising_process_flow(noise_flow_rng, x_1.shape)
    cond = jax.tree.map(
        lambda y: jnp.asarray(y)[rand_idx], self._extract_cond(batch)
    )

    x_t = self.interpolant(time_eval, x_0, x_1, noise_interp)
    flow_fn = self.inference_fn(variables, self.flow_model)
    v_t = jax.vmap(flow_fn, in_axes=(1, None, 1), out_axes=1)(
        x_t, time_eval, cond
    )
    time_tiled = jnp.tile(time_eval[:, None], (1, self.num_eval_cases_per_lvl))
    int_losses = jax.vmap(
        self.loss_stochastic_interpolant, in_axes=(0, 0, 0, 0, 0, None)
    )(v_t, time_tiled, x_0, x_1, noise_flow, self.interpolant)

    return {f"time_lvl{i}": loss for i, loss in enumerate(int_losses)}

  @classmethod
  def inference_fn(cls, variables: models.PyTree, flow_model: nn.Module):
    """Returns the conditional inference velocity function."""

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
  """Wraps a flow network and rescales the `[0, 1]` time input.

  Attributes:
    network: Wrapped Flax module accepting `(x, sigma, cond, is_training=...)`
      or `(x, sigma, is_training=...)`.
    time_rescale: Multiplicative scale factor applied to `sigma`.
  """

  network: nn.Module
  time_rescale: float = 1000.0

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: CondDict | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Evaluates `network` with rescaled time input."""
    time = _rescale_sigma(x, sigma, self.time_rescale)
    if cond is None:
      return self.network(x, time, is_training=is_training)
    return self.network(x, time, dict(cond), is_training=is_training)


class RescaledUnet(unets.UNet):
  """2D UNet flow backbone with rescaled `[0, 1]` time input.

  Attributes:
    time_rescale: Multiplicative scale factor mapping `sigma` in `[0, 1]` to the
      noise-embedding scale expected by `UNet`.
  """

  time_rescale: float = 1000.0

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: CondDict | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Evaluates the rescaled 2D UNet."""
    if x.shape[-1] != self.out_channels:
      raise ValueError(
          f"Number of channels in the input ({x.shape[-1]}) must match the"
          f" number of channels in the output ({self.out_channels})."
      )
    time = _rescale_sigma(x, sigma, self.time_rescale)
    cond_dict = dict(cond) if cond is not None else None
    return super().__call__(x, time, cond_dict, is_training=is_training)


class RescaledUnet3d(unets3d.UNet3d):
  """3D UNet flow backbone with rescaled `[0, 1]` time input.

  Attributes:
    time_rescale: Multiplicative scale factor mapping `sigma` in `[0, 1]` to the
      noise-embedding scale expected by `UNet3d`.
  """

  time_rescale: float = 1000.0

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: CondDict | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Evaluates the rescaled 3D UNet."""
    if x.shape[-1] != self.out_channels:
      raise ValueError(
          f"Number of channels in the input ({x.shape[-1]}) must match the"
          f" number of channels in the output ({self.out_channels})."
      )
    time = _rescale_sigma(x, sigma, self.time_rescale)
    cond_dict = dict(cond) if cond is not None else None
    return super().__call__(x, time, cond_dict, is_training=is_training)
