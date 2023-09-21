# Copyright 2023 The swirl_dynamics Authors.
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

"""Training diffusion-based generative models via denoising."""

from collections.abc import Callable
import dataclasses
from typing import ClassVar, Protocol

from clu import metrics as clu_metrics
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array = jax.Array
Metrics = clu_metrics.Collection
VariableDict = trainers.VariableDict


class DenoisingFlaxModule(Protocol):
  """Interface for the flax module expected by `DenoisingModel`."""

  def __call__(self, x: Array, sigma: Array, is_training: bool) -> Array:
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoisingModel(models.BaseModel):
  """Training a denoiser for diffusion-based generative models.

  Attributes:
    input_shape: The tensor shape of a single sample (i.e. without any batch
      dimension).
    denoiser: The flax module to use for denoising. The required interface for
      its `__call__` function is specified by `DenoisingFlaxModule` (not
      statically checked).
    noise_sampling: A callale that samples the noise levels for training.
    noise_weighting: A callable that computes the loss weighting corresponding
      to given noise levels.
    num_eval_cases_per_lvl: The number of evaluation samples to generate (by
      noising randomly chosen members of the evaluation batch) for each noise
      level.
    min_eval_noise_lvl: The minimum noise level for evaluation.
    max_eval_noise_lvl: The maximum noise level for evaluation.
    num_eval_noise_levels: The number of noise levels to evaluate on
      (log-uniformly distributed between the minimum and maximum values).
  """

  input_shape: tuple[int, ...]
  denoiser: nn.Module
  noise_sampling: diffusion.NoiseLevelSampling
  noise_weighting: diffusion.NoiseLossWeighting
  num_eval_cases_per_lvl: int = 1
  min_eval_noise_lvl: float = 1e-3
  max_eval_noise_lvl: float = 50.0
  num_eval_noise_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
    x_sample = jnp.ones((1,) + self.input_shape)
    return self.denoiser.init(
        rng, x=x_sample, sigma=jnp.ones((1,)), is_training=False
    )

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the denoising loss on a training batch.

    Args:
      params: The parameters of the denoising model to differentiate against.
      batch: A batch of training data; expected to contain an `x` field with
        shape `(batch, *spatial_dims, channels)` representing the unnoised
        samples.
      rng: A Jax random key.
      mutables: The mutable (non-diffenretiated) parameters of the denoising
        model (e.g. batch stats); currently assumed emtpy.

    Returns:
      The loss value and a tuple of training metric and mutables.
    """
    batch_size = len(batch["x"])
    rng1, rng2 = jax.random.split(rng)
    sigma = self.noise_sampling(rng=rng1, shape=(batch_size,))
    weights = self.noise_weighting(sigma)
    noise = jax.random.normal(rng2, batch["x"].shape)
    vmapped_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))
    noised = batch["x"] + vmapped_mult(noise, sigma)
    denoised = self.denoiser.apply(
        {"params": params}, x=noised, sigma=sigma, is_training=True
    )
    loss = jnp.mean(vmapped_mult(weights, jnp.square(denoised - batch["x"])))
    metric = dict(loss=loss)
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Compute metrics on an eval batch.

    Randomly selects members of the batch and noise them to a number of fixed
    levels. Each level is aggregated in terms of the average L2 error.

    Args:
      variables: The full model variables for the denoising module.
      batch: A batch of evaluation data; expected to contain an `x` field with
        shape `(batch, *spatial_dims, channels)` representing the unnoised
        samples.
      rng: A Jax random key.

    Returns:
      A dictionary of evaluation metrics.
    """
    choice_rng, noise_rng = jax.random.split(rng)
    x = jax.random.choice(
        key=choice_rng,
        a=batch["x"],
        shape=(self.num_eval_noise_levels, self.num_eval_cases_per_lvl),
    )
    sigma = jnp.exp(
        jnp.linspace(
            self.min_eval_noise_lvl,
            self.max_eval_noise_lvl,
            self.num_eval_noise_levels,
        )
    )
    noise = jax.random.normal(noise_rng, x.shape)
    noised = x + jax.vmap(jnp.multiply, in_axes=(0, 0))(noise, sigma)
    denoise_fn = self.inference_fn(variables, self.denoiser)
    denoised = jax.vmap(denoise_fn, in_axes=(1, None), out_axes=1)(
        noised, sigma
    )
    ema_losses = jax.vmap(jnp.mean)(jnp.square(denoised - x))
    eval_losses = {f"sigma_lvl{i}": loss for i, loss in enumerate(ema_losses)}
    return eval_losses

  @staticmethod
  def inference_fn(variables: models.PyTree, denoiser: nn.Module):
    """Returns the inference denoising function."""

    def _denoise(x: Array, sigma: float | Array) -> Array:
      if not jnp.shape(jnp.asarray(sigma)):
        sigma *= jnp.ones((x.shape[0],))
      return denoiser.apply(variables, x=x, sigma=sigma, is_training=False)

    return _denoise


class DenoisingModelTrainState(train_states.BasicTrainState):
  """Train state with an additional field tracking the EMA params."""

  # ema params is accessed through `ema_state.ema`
  ema_state: optax.EmaState | None = None

  @property
  def ema_variables(self) -> flax.core.FrozenDict:
    if self.ema_state:
      return flax.core.FrozenDict({"params": self.ema_state.ema})
    else:
      raise ValueError("EMA state is none.")


TrainState = DenoisingModelTrainState


class DenoisingTrainer(trainers.BasicTrainer[DenoisingModel, TrainState]):
  """Trainer for diffusion model."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    train_loss: clu_metrics.Average.from_output("loss")
    train_loss_std: clu_metrics.Std.from_output("loss")

  EvalMetrics = clu_metrics.Collection.create(  # pylint: disable=invalid-name
      **{
          f"eval_denoise_lvl{i}": clu_metrics.Average.from_output(
              f"sigma_lvl{i}"
          )
          for i in range(DenoisingModel.num_eval_noise_levels)
      }
  )

  def __init__(self, ema_decay: float, *args, **kwargs):
    self.ema = optax.ema(ema_decay)
    super().__init__(*args, **kwargs)

  def initialize_train_state(self, rng: Array) -> TrainState:
    init_vars = self.model.initialize(rng)
    mutables, params = flax.core.pop(init_vars, "params")
    return TrainState.create(
        replicate=self.is_distributed,
        params=params,
        opt_state=self.optimizer.init(params),
        flax_mutables=mutables,
        ema_state=self.ema.init(params),
    )

  @property
  def update_train_state(
      self,
  ) -> Callable[[TrainState, VariableDict, VariableDict], TrainState]:
    """Returns function that updates the train state."""

    def _update_train_state(
        train_state: TrainState,
        grads: VariableDict,
        mutables: VariableDict,
    ) -> TrainState:
      updates, new_opt_state = self.optimizer.update(
          grads, train_state.opt_state, train_state.params
      )
      new_params = optax.apply_updates(train_state.params, updates)
      _, new_ema_state = self.ema.update(new_params, train_state.ema_state)
      return train_state.replace(
          step=train_state.step + 1,
          opt_state=new_opt_state,
          params=new_params,
          flax_mutables=mutables,
          ema_state=new_ema_state,
      )

    return _update_train_state

  @staticmethod
  def inference_fn_from_state_dict(
      state: TrainState, *args, use_ema: bool = True, **kwargs
  ):
    if use_ema:
      if isinstance(state.ema_state, dict):
        variables = flax.core.FrozenDict({"params": state.ema_state["ema"]})
      else:
        variables = state.ema_variables
    else:
      variables = state.model_variables
    return DenoisingModel.inference_fn(variables, *args, **kwargs)


class DistributedDenoisingTrainer(
    DenoisingTrainer,
    trainers.BasicDistributedTrainer[DenoisingModel, TrainState],
):
  # MRO: DenoisingTrainer > BasicDistributedTrainer > BasicTrainer
  ...
