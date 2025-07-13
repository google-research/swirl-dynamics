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

"""Conditional denoising model with sampling CRPS evaluation."""

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import metrics as metric_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects.probabilistic_diffusion import models as dfn_models
from swirl_dynamics.templates import models

Array = jax.Array
ArrayMapping = Mapping[str, Array]
BatchType = Mapping[str, Any]
PyTree = Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class DenoisingModel(dfn_models.DenoisingModel):
  """Trains a conditional denoising model for GCM-WRF data.

  Evaluation consists of (a) denoising mean squared errors (b) sampling CRPS and
  (c) log likelihood per dimension of the eval data samples.

  Attributes:
    diffusion_scheme: The diffusion scheme for SDE and ODE samplers.
    cg_strength: Classifier guidance strength.
    num_sde_steps: The number of steps for the SDE sampler.
    num_samples_per_condition: The number of samples to generate per condition
      during evaluation. These samples will be generated in one batch.
    num_ode_steps: The number of steps for the ODE probability flow.
    num_likelihood_probes: The number of probes for approximating the log
      likelihood of eval samples.
  """

  diffusion_scheme: dfn_lib.Diffusion
  cg_strength: float = 0.3
  num_sde_steps: int = 256
  num_samples_per_condition: int = 4
  num_ode_steps: int = 128
  num_likelihood_probes: int = 4

  def eval_fn(
      self, variables: PyTree, batch: BatchType, rng: Array
  ) -> models.ArrayDict:
    """Computes denoising, sampling and likelihood metrics on an eval batch."""
    rng_denoise, rng_sample, rng_likelihood = jax.random.split(rng, num=3)
    # The denoising metrics include the one-step denoising mean squared errors
    # at a few different noise levels (see `dfn_models.DenoisingModel` for more
    # details ).
    denoising_metrics = self.denoising_eval(variables, batch, rng_denoise)
    sampling_metrics = self.sampling_eval(variables, batch, rng_sample)
    likelihood_metrics = self.likelihood_eval(variables, batch, rng_likelihood)
    return dict(**denoising_metrics, **sampling_metrics, **likelihood_metrics)

  def sampling_eval(
      self, variables: models.PyTree, batch: BatchType, rng: Array
  ) -> models.ArrayDict:
    """Computes sampling metrics on an eval batch.

    This function runs end-to-end SDE-based sampling and evaluates the quality
    of final samples. Currently, only the continuous ranked probability score
    (CRPS) is computed.

    Args:
      variables: Variables for the denoising model.
      batch: A batch of evaluation data.
      rng: A Jax random key.

    Returns:
      Dictionary of sampling metrics.
    """
    sde_sampler = self.get_sde_sampler(variables)
    sampling_fn = functools.partial(
        sde_sampler.generate, self.num_samples_per_condition
    )
    def sampling_map(args):
      return sampling_fn(*args)

    # We use lax.map as a memory efficient alternative to jax.vmap
    samples = jax.lax.map(
        sampling_map,
        (
            jax.random.split(rng, batch["x"].shape[0]),
            batch["cond"],
            batch.get("guidance_inputs", {}),
        ),
    )

    crps = jnp.mean(metric_lib.crps(forecasts=samples, observations=batch["x"]))
    rmse = metric_lib.mean_squared_error(
        jnp.mean(samples, axis=1), batch["x"], squared=False
    )
    return {
        # Take first batch element and one sample only. The batch axis is kept
        # to work with `CollectingMetric` in clu.
        "gen_sample": jnp.asarray(samples)[:1, 0],
        "example_input": batch["cond"]["channel:input"][:1],
        "example_obs": batch["x"][:1],
        "mean_crps": crps,
        "rmse_ens_mean": rmse,
    }  # pytype: disable=bad-return-type

  def likelihood_eval(
      self, variables: PyTree, batch: BatchType, rng: Array
  ) -> models.ArrayDict:
    """Evaluates the log likelihood using the ODE probability flow.

    The likelihood is evaluated using the instantaneous change of variable
    formula (see `dfn_lib.OdeSampler.compute_log_likelihood()` for more
    details). The result represents the probability density of eval data under
    the current distribution parametrized by the probability flow.

    Args:
      variables: Variables for the denoising model.
      batch: A batch of evaluation data.
      rng: A Jax random key.

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
            sample_log_likelihood / np.prod(self.input_shape)
        )
    }

  def get_sde_sampler(self, variables: PyTree) -> dfn_lib.SdeSampler:
    """Constructs a SDE sampler from given denoising model variables."""
    return dfn_lib.SdeSampler(
        input_shape=self.input_shape,
        denoise_fn=self.inference_fn(variables, self.denoiser),
        integrator=solver_lib.EulerMaruyama(iter_type="loop"),  # Save memory.
        scheme=self.diffusion_scheme,
        guidance_transforms=(
            dfn_lib.ClassifierFreeHybrid(guidance_strength=self.cg_strength),
        ),
        tspan=dfn_lib.edm_noise_decay(
            self.diffusion_scheme, num_steps=self.num_sde_steps
        ),
    )

  def get_ode_sampler(self, variables: PyTree) -> dfn_lib.OdeSampler:
    """Constructs an ODE sampler from given denoising model variables."""
    return dfn_lib.OdeSampler(
        input_shape=self.input_shape,
        denoise_fn=self.inference_fn(variables, self.denoiser),
        integrator=solver_lib.HeunsMethod(),
        scheme=self.diffusion_scheme,
        guidance_transforms=(
            dfn_lib.ClassifierFreeHybrid(guidance_strength=self.cg_strength),
        ),
        tspan=dfn_lib.edm_noise_decay(
            self.diffusion_scheme, num_steps=self.num_ode_steps
        ),
        num_probes=self.num_likelihood_probes,
    )


@flax.struct.dataclass
class SdeSampler(dfn_lib.SdeSampler):
  """SDE sampling class with de-normalizing functionality.

  Attributes:
    mean: The scaling bias applied to the generated samples.
    std: The scaling factor applied to the generated samples.
  """

  rescale_mean: np.ndarray = dataclasses.field(
      default_factory=functools.partial(np.zeros, ())
  )
  rescale_std: np.ndarray = dataclasses.field(
      default_factory=functools.partial(np.ones, ())
  )

  def generate_and_denormalize(
      self,
      num_samples: int,
      rng: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Generates and denormalizes a batch of diffusion samples from scratch.

    Args:
      num_samples: The number of samples to generate in a single batch.
      rng: The base rng for the generation process.
      cond: Explicit conditioning inputs for the denoising function. These
        should be provided **without** batch dimensions (one should be added
        inside this function based on `num_samples`).
      guidance_inputs: Inputs used to construct the guided denoising function.
        These also should in principle not include a batch dimension.

    Returns:
      The generated denormalized samples.
    """
    generated = self.generate(num_samples, rng, cond, guidance_inputs)
    return generated * self.rescale_std + self.rescale_mean

  def generate_denormalize_and_add_input(
      self,
      input_indices: list[int],
      input_mean: Array,
      input_std: Array,
      num_samples: int,
      rng: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Generates, denormalizes and adds the input to a batch of samples.

    Args:
      input_indices: The index of the input channels corresponding to the output
        variables.
      input_mean: The scaling bias applied to the full conditioning inputs.
        Indexing of the output variables is done a posteriori.
      input_std: The scaling factor applied to the full conditioning inputs.
        Indexing of the output variables is done a posteriori.
      num_samples: The number of samples to generate in a single batch.
      rng: The base rng for the generation process.
      cond: Explicit conditioning inputs for the denoising function. These
        should be provided **without** batch dimensions (one should be added
        inside this function based on `num_samples`).
      guidance_inputs: Inputs used to construct the guided denoising function.
        These also should in principle not include a batch dimension.

    Returns:
      The generated denormalized samples.
    """
    generated = self.generate_and_denormalize(
        num_samples, rng, cond, guidance_inputs
    )

    broadcast_shape = (1,) * (len(cond["channel:input"].shape) - 1)
    added_input = cond["channel:input"] * jax.lax.broadcast(
        jnp.array(input_std), broadcast_shape
    ) + jax.lax.broadcast(jnp.array(input_mean), broadcast_shape)

    return generated + added_input[..., input_indices]
