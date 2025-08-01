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

"""Training modules for generative super-resolution."""

from collections.abc import Callable, Mapping
import dataclasses
import functools
from typing import Any, TypeAlias

import clu.metrics as clu_metrics
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from swirl_dynamics import templates
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import metrics as metric_lib
from swirl_dynamics.lib import solvers as solver_lib


Array: TypeAlias = jax.Array
ArrayDict: TypeAlias = Mapping[str, Array]
BatchType: TypeAlias = Mapping[str, Any]
LossAndAux: TypeAlias = tuple[jax.Array, tuple[ArrayDict, Any]]
PyTree: TypeAlias = Any
ShapeDict: TypeAlias = Mapping[str, Any]
VariableDict: TypeAlias = flax.core.scope.FrozenVariableDict


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
class DenoisingModel(templates.BaseModel):
  """Denoising diffusion model with additional eval metrics for super-resolution.

  The model is trained to remove noise from a noisy input sample, given the
  corresponding noise level and corresponding conditional inputs representing
  the low-resolution information.

  This class also implements various optional evaluation metrics that help
  monitor progress during training. Besides denoising mean squared error (MSE)
  at different noise levels, additional eval metrics include continuous ranked
  probability scores (CRPS) of the samples generated by a sampling SDE and eval
  data log likelihood computed from the probability flow ODE. Some generated
  samples are also returned from evaluation for visualization.

  Attributes:
    sample_shape: The tensor shape of the high-resolution samples (excluding
      batch dimensions).
    denoiser: A flax neural network module that makes the denoising computation.
      Its `__call__` method should have the signature: `(x, sigma, cond,
      is_training) -> y, where `x` and `y` are the noisy and clean samples
      respectively, `sigma` is thenoise level, `cond` is the conditional input
      dictionary, and `is_training` is a flag indicating whether the method is
      being called during training.
    noise_sampling: A function that samples noise levels for training.
    noise_weighting: A function that computes the weighting for a given noise
      level in the loss.
    eval_denoise: Whether to evaluate denoising metrics (MSE at a few fixed
      noise levels).
    cond_shape: Dictionary mapping conditional input names to their shapes (as
      tuples). This allows for nested structures if needed.
    num_eval_noise_levels: The number of noise levels to evaluate the denoising
      L2 error.
    num_eval_cases_per_lvl: The number of evaluation samples created per noise
      level. These are generated by adding noise to random members of the
      evaluation batch.
    min_eval_noise_lvl: The minimum noise level to evaluate denoising. Should be
      close to zero.
    max_eval_noise_lvl: The maximum noise level to evaluate denoising. Should be
      close to the maximum noise level used during training.
    eval_sampling: Whether to evaluate sampling metrics (CRPS).
    diffusion_scheme: The diffusion scheme (i.e. noise and scale schedules) for
      the SDE and ODE samplers.
    cfg_strength: Classifier-free guidance strength. See Ho and Salimans
      (https://arxiv.org/abs/2207.12598) for details.
    num_sde_steps: The number of solver steps for solving the sampling SDE.
    num_samples_per_condition: The number of samples to generate per condition
      during evaluation. These samples will be generated in one single batch.
    eval_likelihood: Whether to evaluate the log likelihood of the eval samples
      using the ODE probability flow.
    num_ode_steps: The number of solver steps for likelihood evaluation using
      the ODE probability flow.
    num_likelihood_probes: The number of probes for approximating the log
      likelihood of eval samples.
  """

  sample_shape: tuple[int, ...]
  denoiser: nn.Module
  noise_sampling: dfn_lib.NoiseLevelSampling
  noise_weighting: dfn_lib.NoiseLossWeighting
  cond_shape: ShapeDict | None = None
  eval_denoise: bool = True
  num_eval_noise_levels: int = 5
  num_eval_cases_per_lvl: int = 1
  min_eval_noise_lvl: float = 1e-3
  max_eval_noise_lvl: float = 50.0
  eval_sampling: bool = True
  diffusion_scheme: dfn_lib.Diffusion
  cfg_strength: float = 0.3
  num_sde_steps: int = 128
  num_samples_per_condition: int = 4
  eval_likelihood: bool = True
  num_ode_steps: int = 64
  num_likelihood_probes: int = 4

  def initialize(self, rng: Array):
    """Initializes the denoising neural network parameters."""
    x = jnp.ones((1,) + self.sample_shape)
    cond = cond_sample_from_shape(self.cond_shape, batch_dims=(1,))
    return self.denoiser.init(
        rng, x=x, sigma=jnp.ones((1,)), cond=cond, is_training=False
    )

  def loss_fn(
      self, params: PyTree, batch: BatchType, rng: Array, mutables: PyTree
  ) -> LossAndAux:
    """Computes the denoising loss on a training batch.

    Args:
      params: The parameters (differentiated) of the denoising model.
      batch: A batch of training data, expected to contain an `x` field with a
        shape of `(batch, *spatial_dims, channels)`, representing the unnoised
        samples. Optionally, it may also contain a `cond` field, which is a
        dictionary of conditional inputs.
      rng: Key for random number generation.
      mutables: The mutable (non-diffenretiated) parameters of the denoising
        model (e.g. batch stats).

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
        rngs={"dropout": rng3},
    )
    loss = jnp.mean(vmapped_mult(weights, jnp.square(denoised - batch["x"])))
    metric = dict(loss=loss)
    return loss, (metric, mutables)

  def eval_fn(
      self, variables: PyTree, batch: BatchType, rng: Array
  ) -> Mapping[str, jax.Array]:
    """Compute metrics on an evaluation batch."""
    rng_denoise, rng_sample, rng_likelihood = jax.random.split(rng, num=3)
    eval_metrics = {}
    if self.eval_denoise:
      eval_metrics.update(self.denoising_eval(variables, batch, rng_denoise))
    if self.eval_sampling:
      eval_metrics.update(self.sampling_eval(variables, batch, rng_sample))
    if self.eval_likelihood:
      eval_metrics.update(
          self.likelihood_eval(variables, batch, rng_likelihood)
      )
    return eval_metrics

  def denoising_eval(
      self, variables: PyTree, batch: BatchType, rng: Array
  ) -> ArrayDict:
    """Compute denoising metrics on an eval batch.

    Randomly selects members of the batch and noise them to a number of fixed
    levels. Each level is aggregated in terms of the average L2 error.

    Args:
      variables: Variables for the denoising module.
      batch: A batch of evaluation data expected to contain an `x` field with a
        shape of `(batch, *spatial_dims, channels)`, representing the unnoised
        samples. Optionally, it may also contain a `cond` field, which is a
        dictionary of conditional inputs.
      rng: Key for random number generation.

    Returns:
      A dictionary of denoising-based evaluation metrics.
    """
    choice_rng, noise_rng = jax.random.split(rng)
    rand_idx = jax.random.choice(
        key=choice_rng,
        a=jnp.arange(batch["x"].shape[0]),
        shape=(self.num_eval_noise_levels, self.num_eval_cases_per_lvl),
    )
    x = batch["x"][rand_idx]
    cond = (
        jax.tree.map(lambda y: y[rand_idx], batch["cond"])
        if self.cond_shape
        else None
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
    denoise_fn = self.inference_fn(variables, self.denoiser)
    denoised = jax.vmap(denoise_fn, in_axes=(1, None, 1), out_axes=1)(
        noised, sigma, cond
    )
    ema_losses = jax.vmap(jnp.mean)(jnp.square(denoised - x))
    return {f"sigma_lvl{i}": loss for i, loss in enumerate(ema_losses)}

  def sampling_eval(
      self, variables: PyTree, batch: BatchType, rng: Array
  ) -> Mapping[str, jax.Array]:
    """Compute sampling metrics on an eval batch."""
    sde_sampler = self.get_sde_sampler(variables)
    sampling_fn = functools.partial(
        sde_sampler.generate, self.num_samples_per_condition
    )
    samples = jax.vmap(sampling_fn, in_axes=(0, 0, 0))(
        jax.random.split(rng, batch["x"].shape[0]),
        batch["cond"],
        batch.get("guidance_inputs", {}),
    )  # ~ (batch, samples, *sample_dims)
    crps = jnp.mean(metric_lib.crps(forecasts=samples, observations=batch["x"]))
    return {
        # Take first batch element and one sample only. The batch axis is kept
        # to work with `CollectingMetric` in clu.
        "example_sample": jnp.asarray(samples)[:1, 0],
        "example_input": batch["cond"]["channel:daily_mean"][:1],
        "example_obs": batch["x"][:1],
        "mean_crps": crps,
    }  # pytype: disable=bad-return-type

  def likelihood_eval(
      self, variables: PyTree, batch: BatchType, rng: Array
  ) -> Mapping[str, jax.Array]:
    """Evaluates the log likelihood using the ODE probability flow.

    The likelihood of eval data is approximated using the instantaneous change
    of variable formula (see `dfn_lib.OdeSampler.compute_log_likelihood()` for
    more details), computed with the probability flow ODE.

    Args:
      variables: Variables for the denoising model.
      batch: Same as in `.denoising_eval()`.
      rng: Key for random number generation.

    Returns:
      Likelihood of the eval samples per dimension.
    """
    ode_sampler = self.get_ode_sampler(variables)
    sample_log_likelihood = ode_sampler.compute_log_likelihood(
        inputs=batch["x"],
        cond=batch["cond"],
        guidance_inputs=batch.get("guidance_inputs", {}),
        rng=rng,
    )
    return {
        "sample_log_likelihood_per_dim": jnp.mean(
            sample_log_likelihood / np.prod(self.sample_shape)
        )
    }

  def get_sde_sampler(self, variables: PyTree) -> dfn_lib.SdeSampler:
    """Constructs a SDE sampler from given denoising model variables."""
    return dfn_lib.SdeSampler(
        input_shape=self.sample_shape,
        denoise_fn=self.inference_fn(variables, self.denoiser),
        integrator=solver_lib.EulerMaruyama(iter_type="loop"),  # Save memory.
        scheme=self.diffusion_scheme,
        guidance_transforms=(
            dfn_lib.ClassifierFreeHybrid(guidance_strength=self.cfg_strength),
        ),
        tspan=dfn_lib.edm_noise_decay(
            self.diffusion_scheme, num_steps=self.num_sde_steps
        ),
    )

  def get_ode_sampler(self, variables: PyTree) -> dfn_lib.OdeSampler:
    """Constructs an ODE sampler from given denoising model variables."""
    return dfn_lib.OdeSampler(
        input_shape=self.sample_shape,
        denoise_fn=self.inference_fn(variables, self.denoiser),
        integrator=solver_lib.HeunsMethod(),
        scheme=self.diffusion_scheme,
        guidance_transforms=(
            dfn_lib.ClassifierFreeHybrid(guidance_strength=self.cfg_strength),
        ),
        tspan=dfn_lib.edm_noise_decay(
            self.diffusion_scheme, num_steps=self.num_ode_steps
        ),
        num_probes=self.num_likelihood_probes,
    )

  @classmethod
  def inference_fn(cls, variables: PyTree, denoiser: nn.Module):
    """Constructs the inference denoising function from variables and denoiser."""

    def _denoise(
        x: Array, sigma: float | Array, cond: ArrayDict | None = None
    ) -> Array:
      if not jnp.shape(jnp.asarray(sigma)):
        sigma *= jnp.ones((x.shape[0],))
      return denoiser.apply(
          variables, x=x, sigma=sigma, cond=cond, is_training=False
      )

    return _denoise


class DenoisingModelTrainState(templates.BasicTrainState):
  """Train state keeping track of model params and optimizer states."""

  # EMA params is accessed through `ema_state.ema`.
  ema_state: optax.EmaState | None = None

  @property
  def ema_variables(self) -> flax.core.FrozenDict:
    if self.ema_state:
      return flax.core.FrozenDict({"params": self.ema_state.ema})
    else:
      raise ValueError("EMA state is none.")


Model: TypeAlias = DenoisingModel
State: TypeAlias = DenoisingModelTrainState


class DenoisingTrainer(templates.BasicDistributedTrainer[Model, State]):
  """A data-parallel trainer for the super-resolution denoising model."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    """Training metrics for the denoising model."""

    train_loss: clu_metrics.Average.from_output("loss")
    train_loss_std: clu_metrics.Std.from_output("loss")

  @functools.cached_property
  def EvalMetrics(self):
    """Evaluation metrics for the denoising model."""
    denoising_metrics = {
        f"eval_denoise_lvl{i}": clu_metrics.Average.from_output(f"sigma_lvl{i}")
        for i in range(self.model.num_eval_noise_levels)
    }
    sampling_metrics = {
        "eval_mean_crps": clu_metrics.Average.from_output("mean_crps"),
        "eval_plot_data": clu_metrics.CollectingMetric.from_outputs(
            ("example_sample", "example_input", "example_obs")
        ),
    }
    likelihood_metrics = {
        "eval_ll_per_dim": clu_metrics.Average.from_output(
            "sample_log_likelihood_per_dim"
        ),
        "eval_ll_per_dim_std": clu_metrics.Std.from_output(
            "sample_log_likelihood_per_dim"
        ),
    }
    return clu_metrics.Collection.create(
        **denoising_metrics, **sampling_metrics, **likelihood_metrics
    )

  def __init__(self, ema_decay: float, *args, **kwargs):
    """Trainer constructor."""
    self.ema = optax.ema(ema_decay)
    super().__init__(*args, **kwargs)

  def initialize_train_state(self, rng: Array) -> State:
    """Initializes the train state."""
    init_vars = self.model.initialize(rng)
    mutables, params = flax.core.pop(init_vars, "params")
    return DenoisingModelTrainState.create(
        replicate=self.is_distributed,
        params=params,
        opt_state=self.optimizer.init(params),
        flax_mutables=mutables,
        ema_state=self.ema.init(params),
    )

  @property
  def update_train_state(
      self,
  ) -> Callable[[State, VariableDict, VariableDict], State]:
    """Returns function that updates the train state."""

    def _update_train_state(
        train_state: State, grads: VariableDict, mutables: VariableDict
    ) -> State:
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
      state: State, *args, use_ema: bool = True, **kwargs
  ):
    """Constructs the inference denoising function from a train state."""
    if use_ema:
      if isinstance(state.ema_state, dict):
        variables = flax.core.FrozenDict({"params": state.ema_state["ema"]})
      else:
        variables = state.ema_variables
    else:
      variables = state.model_variables
    return DenoisingModel.inference_fn(variables, *args, **kwargs)
