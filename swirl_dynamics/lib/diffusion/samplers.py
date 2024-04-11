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

"""Diffusion samplers."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import flax
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.lib.diffusion import guidance
from swirl_dynamics.lib.solvers import ode
from swirl_dynamics.lib.solvers import sde

Array = jax.Array
ArrayMapping = Mapping[str, Array]
Params = Mapping[str, Any]


class DenoiseFn(Protocol):

  def __call__(
      self, x: Array, sigma: Array, cond: ArrayMapping | None
  ) -> Array:
    ...


ScoreFn = DenoiseFn


def dlog_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
  """Returns d/dt log(f(t)) = ḟ(t)/f(t) given f(t)."""
  return jax.grad(lambda t: jnp.log(f(t)))


def dsquare_dt(f: diffusion.ScheduleFn) -> diffusion.ScheduleFn:
  """Returns d/dt (f(t))^2 = 2ḟ(t)f(t) given f(t)."""
  return jax.grad(lambda t: jnp.square(f(t)))


def denoiser2score(
    denoise_fn: DenoiseFn, scheme: diffusion.Diffusion
) -> ScoreFn:
  """Converts a denoiser to the corresponding score function."""

  def _score(x: Array, sigma: Array, cond: ArrayMapping | None = None) -> Array:
    # Reference: eq. 74 in Karras et al. (https://arxiv.org/abs/2206.00364).
    scale = scheme.scale(scheme.sigma.inverse(sigma))
    x_hat = jnp.divide(x, scale)
    target = denoise_fn(x_hat, sigma, cond)
    return jnp.divide(target - x_hat, scale * jnp.square(sigma))

  return _score


# ********************
# Time step schedulers
# ********************


class TimeStepScheduler(Protocol):

  def __call__(self, scheme: diffusion.Diffusion, *args, **kwargs) -> Array:
    """Outputs the time steps based on diffusion noise schedule."""
    ...


def uniform_time(
    scheme: diffusion.Diffusion,
    num_steps: int = 256,
    end_time: float | None = 1e-3,
    end_sigma: float | None = None,
) -> Array:
  """Time steps uniform in [t_min, t_max]."""
  if (end_time is None and end_sigma is None) or (
      end_time is not None and end_sigma is not None
  ):
    raise ValueError(
        "Exactly one of `end_time` and `end_sigma` must be specified."
    )

  start = diffusion.MAX_DIFFUSION_TIME
  end = end_time or scheme.sigma.inverse(end_sigma)
  return jnp.linspace(start, end, num_steps)


def exponential_noise_decay(
    scheme: diffusion.Diffusion,
    num_steps: int = 256,
    end_sigma: float | None = 1e-3,
) -> Array:
  """Time steps corresponding to exponentially decaying sigma."""
  exponent = jnp.arange(num_steps) / (num_steps - 1)
  r = end_sigma / scheme.sigma_max
  sigma_schedule = scheme.sigma_max * jnp.power(r, exponent)
  return jnp.asarray(scheme.sigma.inverse(sigma_schedule))


def edm_noise_decay(
    scheme: diffusion.Diffusion,
    rho: int = 7,
    num_steps: int = 256,
    end_sigma: float | None = 1e-3,
) -> Array:
  """Time steps corresponding to Eq. 5 in Karras et al."""
  rho_inv = 1 / rho
  sigma_schedule = jnp.arange(num_steps) / (num_steps - 1)
  sigma_schedule *= jnp.power(end_sigma, rho_inv) - jnp.power(
      scheme.sigma_max, rho_inv
  )
  sigma_schedule += jnp.power(scheme.sigma_max, rho_inv)
  sigma_schedule = jnp.power(sigma_schedule, rho)
  return jnp.asarray(scheme.sigma.inverse(sigma_schedule))


# ********************
# Samplers
# ********************


@flax.struct.dataclass
class Sampler:
  """Base class for denoising-based diffusion samplers.

  Attributes:
    input_shape: The tensor shape of a sample (excluding any batch dimensions).
    scheme: The diffusion scheme which contains the scale and noise schedules.
    denoise_fn: A function to remove noise from input data. Must handle batched
      inputs, noise levels and conditions.
    tspan: Full diffusion time steps for iterative denoising, decreasing from 1
      to (approximately) 0.
    guidance_transforms: An optional sequence of guidance transforms that
      modifies the denoising function in a post-process fashion.
    apply_denoise_at_end: If `True`, applies the denoise function another time
      to the terminal states, which are typically at a small but non-zero noise
      level.
    return_full_paths: If `True`, the output of `.generate()` and `.denoise()`
      will contain the complete sampling paths. Otherwise only the terminal
      states are returned.
  """

  input_shape: tuple[int, ...]
  scheme: diffusion.Diffusion
  denoise_fn: DenoiseFn
  tspan: Array
  guidance_transforms: Sequence[guidance.Transform] = ()
  apply_denoise_at_end: bool = True
  return_full_paths: bool = False

  def generate(
      self,
      num_samples: int,
      rng: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Generates a batch of diffusion samples from scratch.

    Args:
      num_samples: The number of samples to generate in a single batch.
      rng: The base rng for the generation process.
      cond: Explicit conditioning inputs for the denoising function. These
        should be provided **without** batch dimensions (one should be added
        inside this function based on `num_samples`).
      guidance_inputs: Inputs used to construct the guided denoising function.
        These also should in principle not include a batch dimension.

    Returns:
      The generated samples.
    """
    if self.tspan is None or self.tspan.ndim != 1:
      raise ValueError("`tspan` must be a 1-d array.")

    init_rng, denoise_rng = jax.random.split(rng)
    x_shape = (num_samples,) + self.input_shape
    x1 = jax.random.normal(init_rng, x_shape)
    x1 *= self.scheme.sigma(self.tspan[0]) * self.scheme.scale(self.tspan[0])

    if cond is not None:
      cond = jax.tree.map(lambda x: jnp.stack([x] * num_samples, axis=0), cond)

    denoised = self.denoise(x1, denoise_rng, self.tspan, cond, guidance_inputs)

    samples = denoised[-1] if self.return_full_paths else denoised
    if self.apply_denoise_at_end:
      denoise_fn = self.get_guided_denoise_fn(guidance_inputs=guidance_inputs)
      samples = denoise_fn(
          jnp.divide(samples, self.scheme.scale(self.tspan[-1])),
          self.scheme.sigma(self.tspan[-1]),
          cond,
      )
      if self.return_full_paths:
        denoised = jnp.concatenate([denoised, samples[None]], axis=0)

    return denoised if self.return_full_paths else samples

  def denoise(
      self,
      noisy: Array,
      rng: Array,
      tspan: Array,
      cond: ArrayMapping | None,
      guidance_inputs: ArrayMapping | None,
  ) -> Array:
    """Applies iterative denoising to given noisy states.

    Args:
      noisy: A batch of noisy states (all at the same noise level). Can be fully
        noisy or partially denoised.
      rng: Base Jax rng for denoising.
      tspan: A decreasing sequence of diffusion time steps within the interval
        [1, 0). The first element aligns with the time step of the `noisy`
        input.
      cond: (Optional) Conditioning inputs for the denoise function. The batch
        dimension should match that of `noisy`.
      guidance_inputs: Inputs for constructing the guided denoising function.

    Returns:
      The denoised output.
    """
    raise NotImplementedError

  def get_guided_denoise_fn(
      self, guidance_inputs: Mapping[str, Array]
  ) -> DenoiseFn:
    """Returns a guided denoise function."""
    denoise_fn = self.denoise_fn
    for transform in self.guidance_transforms:
      denoise_fn = transform(denoise_fn, guidance_inputs)
    return denoise_fn


@flax.struct.dataclass
class OdeSampler(Sampler):
  """Draw samples by solving an probability flow ODE.

  Attributes:
    integrator: The ODE solver for solving the sampling ODE.
  """

  integrator: ode.OdeSolver = ode.HeunsMethod()

  def denoise(
      self,
      noisy: Array,
      rng: Array,
      tspan: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Applies iterative denoising to given noisy states."""
    del rng
    params = dict(cond=cond, guidance_inputs=guidance_inputs)
    # Currently all ODE integrators return full paths. The lead axis should
    # always be time.
    denoised = self.integrator(self.dynamics, noisy, tspan, params)
    return denoised if self.return_full_paths else denoised[-1]

  @property
  def dynamics(self) -> ode.OdeDynamics:
    """The RHS of the sampling ODE.

    In score function (eq. 3 in Karras et al. https://arxiv.org/abs/2206.00364):

      dx = [ṡ(t)/s(t) x - s(t)² σ̇(t)σ(t) ∇pₜ(x)] dt,

    or, in terms of denoise function (eq. 81):

      dx = [σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) dt

    where s(t), σ(t) are the scale and noise schedule of the diffusion scheme.
    """

    def _dynamics(x: Array, t: Array, params: Params) -> Array:
      assert not t.ndim, "`t` must be a scalar."
      denoise_fn = self.get_guided_denoise_fn(
          guidance_inputs=params["guidance_inputs"]
      )
      s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
      x_hat = jnp.divide(x, s)
      dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
      dlog_s_dt = dlog_dt(self.scheme.scale)(t)
      target = denoise_fn(x_hat, sigma, params["cond"])
      return (dlog_sigma_dt + dlog_s_dt) * x - dlog_sigma_dt * s * target

    return _dynamics

  def compute_log_likelihood(
      self,
      inputs: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
      eps: Array | None = None,
  ) -> Array:
    """Computes the log likelihood of given data using the probability flow.

    This is done by integrating the probability flow ODE forward (from zero to
    max noise), using `inputs` as the initial condition. The log likelihood is
    then calculated from that of the terminal state using the change of variable
    formula. This formula involves computing the trace of the Jacobian of the
    dynamics, which can be done either exactly or approximately via Hutchinson's
    trace estimator.

    Args:
      inputs: Data whose log likelihood is computed. Expected to have shape
        `(batch, *input_shape)`.
      cond: Conditioning inputs for the denoise function. The batch dimension
        should match that of `data`.
      guidance_inputs: Inputs for constructing the guided denoising function.
      eps: The 'probes' to use for Hutchinson's trace estimator (see
        `.log_likelihood_augmented_dynamics_approximate`). Expected to have
        shape (batch, num_probes, *input_shape). If `None`, the trace is
        computed exactly (see `.log_likelihood_augmented_dynamics_exact`).

    Returns:
      The computed log likelihood.
    """
    if inputs.shape[1:] != self.input_shape:
      raise ValueError(
          "`inputs` is expected to have shape (batch_size,) +"
          f" {self.input_shape}, but has shape {inputs.shape} instead."
      )

    if eps is not None and (
        eps.shape[2:] != self.input_shape or eps.shape[0] != inputs.shape[0]
    ):
      raise ValueError(
          "`eps` is expected to have shape (batch_size, num_probes,"
          f" *input_shape) but has shape {eps.shape} instead."
      )

    batch_size = inputs.shape[0]
    dim = np.prod(self.input_shape)
    params = dict(cond=cond, guidance_inputs=guidance_inputs)

    if eps is not None:
      params["eps"] = jnp.reshape(eps, (*eps.shape[:2], dim))
      dynamics_fn = self.log_likelihood_augmented_dynamics_approximate
    else:
      dynamics_fn = self.log_likelihood_augmented_dynamics_exact

    q0 = jnp.concatenate(
        [jnp.reshape(inputs, (batch_size, -1)), jnp.zeros((batch_size, 1))],
        axis=-1,
    )
    q_paths = self.integrator(
        func=dynamics_fn, x0=q0, tspan=self.tspan[::-1], params=params
    )
    x1, dlogp01 = q_paths[-1, :, :-1], q_paths[-1, :, -1]

    sigma1 = self.scheme.sigma(self.tspan[0]) * self.scheme.sigma(self.tspan[0])
    log_p1 = (
        -dim / 2 * jnp.log(2 * jnp.pi)
        - dim * jnp.log(sigma1)
        - 0.5 * jnp.einsum("nd,nd->n", x1, x1) / jnp.square(sigma1)
    )
    # The sign before `dlogp01` is minus because the integration limits are
    # reversed compared to eq. 3 in https://arxiv.org/abs/1810.01367
    return log_p1 - dlogp01

  @property
  def log_likelihood_augmented_dynamics_exact(self) -> ode.OdeDynamics:
    """Exact augmented dynamics for computing the log likelihood.

    The probability flow ODE dynamics is augmented with the dynamics of the log
    density, described by the instantaneous change of variable formula (see
    Grathwohl et al. https://arxiv.org/abs/1810.01367, eq. 2):

      d(log p(x))/dt = - Trace(dF/dx),

    where F denotes RHS of the probability flow. The trace is evaluated exactly
    using vector-Jacobian products vmapped over rows of an identity matrix.
    The cost is proportional to the number of sample dimensions, which can be
    expensive.
    """

    def _exact_dynamics(x: Array, t: Array, params: Params) -> Array:
      x = x[:, :-1]
      fn = lambda x: self.dynamics(
          x.reshape(x.shape[0], *self.input_shape), t, params
      ).reshape(x.shape[0], -1)
      dxdt, vjp_fn = jax.vjp(fn, x)
      eye = jnp.stack([jnp.eye(x.shape[-1])] * x.shape[0], axis=0)
      (dfndx,) = jax.vmap(vjp_fn, in_axes=1, out_axes=1)(eye)
      dlogp = -jnp.trace(dfndx, axis1=1, axis2=2)
      return jnp.concatenate([dxdt, dlogp[:, None]], axis=-1)

    return _exact_dynamics

  @property
  def log_likelihood_augmented_dynamics_approximate(self) -> ode.OdeDynamics:
    """Approximate augmented dynamics for computing the log likelihood.

    The trace in the augmented log density dynamics (see
    `.log_likelihood_augmented_dynamics_exact`) is approximated with
    Hutchinson's trace estimator (eq. 7, https://arxiv.org/abs/1810.01367):

      Trace(dF/dx) = E [εᵀ(dF/dx)ε],

    where E denotes expection over ε, which are typically sampled from standard
    Gaussian distributions. The cost of this estimator is proportional to the
    number of probes (i.e. ε samples) instead of the sample dimension.
    """

    def _approximate_dynamics(x: Array, t: Array, params: Params) -> Array:
      x = x[:, :-1]
      fn = lambda x: self.dynamics(
          x.reshape(x.shape[0], *self.input_shape), t, params
      ).reshape(x.shape[0], -1)
      dxdt, vjp_fn = jax.vjp(fn, x)
      (eps_vjp,) = jax.vmap(vjp_fn, in_axes=1, out_axes=1)(params["eps"])
      dlogp = -jnp.mean(
          jnp.einsum("bnd,bnd->bn", eps_vjp, params["eps"]), axis=1
      )
      return jnp.concatenate([dxdt, dlogp[:, None]], axis=-1)

    return _approximate_dynamics


@flax.struct.dataclass
class SdeSampler(Sampler):
  """Draws samples by solving an SDE.

  Attributes:
    integrator: The SDE solver for solving the sampling SDE.
  """

  integrator: sde.SdeSolver = sde.EulerMaruyama(iter_type="scan")

  def denoise(
      self,
      noisy: Array,
      rng: Array,
      tspan: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Applies iterative denoising to given noisy states."""
    if self.integrator.terminal_only and self.return_full_paths:
      raise ValueError(
          f"Integrator type `{type(self.integrator)}` does not support"
          " returning full paths."
      )

    params = dict(
        drift=dict(guidance_inputs=guidance_inputs, cond=cond), diffusion={}
    )
    denoised = self.integrator(self.dynamics, noisy, tspan, rng, params)
    # SDE solvers may return either the full paths or the terminal state only.
    # If the former, the lead axis should be time.
    samples = denoised if self.integrator.terminal_only else denoised[-1]
    return denoised if self.return_full_paths else samples

  @property
  def dynamics(self) -> sde.SdeDynamics:
    """Drift and diffusion terms of the sampling SDE.

    In score function:

      dx = [ṡ(t)/s(t) x - 2 s(t)²σ̇(t)σ(t) ∇pₜ(x)] dt + s(t) √[2σ̇(t)σ(t)] dωₜ,

    obtained by substituting eq. 28, 34 of Karras et al.
    (https://arxiv.org/abs/2206.00364) into the reverse SDE formula - eq. 6 in
    Song et al. (https://arxiv.org/abs/2011.13456). Alternatively, it may be
    rewritten in terms of the denoise function (plugging in eq. 74 of
    Karras et al.) as:

      dx = [2 σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [2 s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) dt
        + s(t) √[2σ̇(t)σ(t)] dωₜ

    where s(t), σ(t) are the scale and noise schedule of the diffusion scheme
    respectively.
    """

    def _drift(x: Array, t: Array, params: Params) -> Array:
      assert not t.ndim, "`t` must be a scalar."
      denoise_fn = self.get_guided_denoise_fn(
          guidance_inputs=params["guidance_inputs"]
      )
      s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
      x_hat = jnp.divide(x, s)
      dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
      dlog_s_dt = dlog_dt(self.scheme.scale)(t)
      drift = (2 * dlog_sigma_dt + dlog_s_dt) * x
      drift -= 2 * dlog_sigma_dt * s * denoise_fn(x_hat, sigma, params["cond"])
      return drift

    def _diffusion(x: Array, t: Array, params: Params) -> Array:
      del x, params
      assert not t.ndim, "`t` must be a scalar."
      dsquare_sigma_dt = dsquare_dt(self.scheme.sigma)(t)
      return jnp.sqrt(dsquare_sigma_dt) * self.scheme.scale(t)

    return sde.SdeDynamics(_drift, _diffusion)
