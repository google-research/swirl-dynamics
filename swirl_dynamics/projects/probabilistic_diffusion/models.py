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

"""Training a denoising model for diffusion-based generation."""

from collections.abc import Mapping
import dataclasses
from typing import Any, ClassVar, Protocol

from clu import metrics as clu_metrics
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers

Array = jax.Array
CondDict = Mapping[str, Array]
Metrics = clu_metrics.Collection
ShapeDict = Mapping[str, Any]  # may be nested
PyTree = Any
VariableDict = trainers.VariableDict


class DenoisingFlaxModule(Protocol):
  """Expected interface of the flax module compatible with `DenoisingModel`.

  NOTE: This protocol is for reference only and not statically checked.
  """

  def __call__(
      self, x: Array, sigma: Array, cond: CondDict | None, is_training: bool
  ) -> Array:
    ...


def _cond_sample_from_shape(
    shape: ShapeDict | None, batch_dims: tuple[int, ...] = (1,)
) -> PyTree:
  """Instantiates a conditional input sample based on shape specifications."""
  if shape is None:
    return None
  elif isinstance(shape, tuple):
    return jnp.ones(batch_dims + shape)
  elif isinstance(shape, dict):
    return {k: _cond_sample_from_shape(v) for k, v in shape.items()}
  else:
    raise TypeError(f"Cannot initialize shape: {shape}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoisingModel(models.BaseModel):
  """Training a model to remove Gaussian noise from samples.

  Attributes:
    input_shape: The tensor shape of a single sample (i.e. without any batch
      dimension).
    denoiser: The flax module to use for denoising. The required interface for
      its `__call__` function is specified by `DenoisingFlaxModule`.
    noise_sampling: A Callable that samples the noise levels for training.
    noise_weighting: A Callable that computes the loss weighting corresponding
      to given noise levels.
    cond_shape: The tensor shapes (as tuples) of conditional inputs as a
      (possibly nested) dict.
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
  cond_shape: ShapeDict | None = None

  num_eval_cases_per_lvl: int = 1
  min_eval_noise_lvl: float = 1e-3
  max_eval_noise_lvl: float = 50.0
  num_eval_noise_levels: ClassVar[int] = 10

  def initialize(self, rng: Array):
    x = jnp.ones((1,) + self.input_shape)
    cond = _cond_sample_from_shape(self.cond_shape, (1,))
    return self.denoiser.init(
        rng, x=x, sigma=jnp.ones((1,)), cond=cond, is_training=False
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
      batch: A batch of training data expected to contain an `x` field with a
        shape of `(batch, *spatial_dims, channels)`, representing the unnoised
        samples. Optionally, it may also contain a `cond` field, which is a
        dictionary of conditional inputs.
      rng: A Jax random key.
      mutables: The mutable (non-diffenretiated) parameters of the denoising
        model (e.g. batch stats); *currently assumed emtpy*.

    Returns:
      The loss value and a tuple of training metric and mutables.
    """
    batch_size = len(batch["x"])
    rng1, rng2, rng3 = jax.random.split(rng, num=3)
    sigma = self.noise_sampling(rng=rng1, shape=(batch_size,))
    weights = self.noise_weighting(sigma)
    noise = jax.random.normal(rng2, batch["x"].shape)
    vmapped_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))
    noised = batch["x"] + vmapped_mult(noise, sigma)
    cond = batch["cond"] if self.cond_shape else None
    denoised = self.denoiser.apply(
        {"params": params},
        x=noised,
        sigma=sigma,
        cond=cond,
        is_training=True,
        rngs={"dropout": rng3},  # TODO(lzepedanunez): refactor this.
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
      batch: A batch of evaluation data expected to contain an `x` field with a
        shape of `(batch, *spatial_dims, channels)`, representing the unnoised
        samples. Optionally, it may also contain a `cond` field, which is a
        dictionary of conditional inputs.
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
            jnp.log(self.min_eval_noise_lvl),
            jnp.log(self.max_eval_noise_lvl),
            self.num_eval_noise_levels,
        )
    )
    noise = jax.random.normal(noise_rng, x.shape)
    noised = x + jax.vmap(jnp.multiply, in_axes=(0, 0))(noise, sigma)
    cond = batch["cond"] if self.cond_shape else None
    denoise_fn = self.inference_fn(variables, self.denoiser)
    denoised = jax.vmap(denoise_fn, in_axes=(1, None, None), out_axes=1)(
        noised, sigma, cond
    )
    ema_losses = jax.vmap(jnp.mean)(jnp.square(denoised - x))
    eval_losses = {f"sigma_lvl{i}": loss for i, loss in enumerate(ema_losses)}
    return eval_losses

  @staticmethod
  def inference_fn(variables: models.PyTree, denoiser: nn.Module):
    """Returns the inference denoising function."""

    def _denoise(
        x: Array, sigma: float | Array, cond: CondDict | None = None
    ) -> Array:
      if not jnp.shape(jnp.asarray(sigma)):
        sigma *= jnp.ones((x.shape[0],))
      return denoiser.apply(
          variables, x=x, sigma=sigma, cond=cond, is_training=False
      )

    return _denoise
