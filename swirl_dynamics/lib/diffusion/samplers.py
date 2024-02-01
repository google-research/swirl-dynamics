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
    # reference: eq. 74 in Karras et al. (https://arxiv.org/abs/2206.00364).
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
      inputs and noise levels.
    guidance_transforms: An optional sequence of guidance transforms that
      modifies the denoising function in a post-process fashion.
  """

  input_shape: tuple[int, ...]
  scheme: diffusion.Diffusion
  denoise_fn: DenoiseFn
  guidance_transforms: Sequence[guidance.Transform]

  def generate(
      self,
      num_samples: int,
      rng: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Generates a batch of diffusion samples.

    Args:
      num_samples: The number of samples to generate in a single batch.
      rng: The base rng for the generation process.
      cond: Explicit conditioning inputs for the denoising function. These
        should be provided without any batch dimensions (one should be added
        inside this function based on `num_samples`).
      guidance_inputs: Inputs used to construct the guided denoising function.
        These also should in principle not include a batch dimension.

    Returns:
      The generated samples.
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
  """Draw samples by solving an probabilistic flow ODE.

  Attributes:
    input_shape: The tensor shape of a sample (excluding any batch dimensions).
    scheme: The diffusion scheme which contains the scale and noise schedules.
    denoise_fn: A function to remove noise from input data. Must handle batched
      inputs and noise levels.
    integrator: The ODE solver for solving the sampling ODE.
    tspan: The time steps for the ODE solver (decreasing typically from 1 to 0).
    guidance_transforms: An optional sequence of guidance transforms that
      modifies the denoising function in a post-process fashion.
    apply_denoise_at_end: Whether to apply the denoise function for another time
      to the terminal state.
    return_full_paths: If `True`, the output will contain the complete sampling
      paths with axis 0 corresponding to diffusion times specified by `tspan`.
  """

  input_shape: tuple[int, ...]
  scheme: diffusion.Diffusion
  denoise_fn: DenoiseFn
  integrator: ode.OdeSolver
  tspan: Array
  guidance_transforms: Sequence[guidance.Transform]
  apply_denoise_at_end: bool = True
  return_full_paths: bool = False

  def generate(
      self,
      num_samples: int,
      rng: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Generate a batch of samples by solving the sampling ODE."""
    if self.tspan.ndim != 1:
      raise ValueError("`tspan` must be a 1-d array.")

    x_shape = (num_samples,) + self.input_shape
    t0, t1 = self.tspan[0], self.tspan[-1]
    x1 = jax.random.normal(rng, x_shape)
    x1 *= self.scheme.sigma(t0) * self.scheme.scale(t0)

    if cond is not None:
      rep_fn = lambda x: jnp.tile(x[None], (num_samples,) + (1,) * x.ndim)
      cond = jax.tree_map(rep_fn, cond)

    params = dict(cond=cond, guidance_inputs=guidance_inputs)
    # Output of integrator must have time at axis 0.
    paths = self.integrator(self.dynamics, x1, self.tspan, params)

    if self.apply_denoise_at_end:
      denoise_fn = self.get_guided_denoise_fn(guidance_inputs=guidance_inputs)
      final = denoise_fn(
          jnp.divide(paths[-1], self.scheme.scale(t1)),
          self.scheme.sigma(t1),
          cond,
      )
      paths = jnp.concatenate([paths, final[None]], axis=0)

    return paths if self.return_full_paths else paths[-1]

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


@flax.struct.dataclass
class SdeSampler(Sampler):
  """Draws samples by solving an SDE.

  Attributes:
    input_shape: The tensor shape of a sample (excluding any batch dimensions).
    scheme: The diffusion scheme which contains the scale and noise schedules.
    denoise_fn: A function to remove noise from input data. Must handle batched
      inputs and noise levels.
    integrator: The SDE solver for solving the sampling SDE.
    tspan: The time steps for the SDE solver  (decreasing typically from 1 to
      0).
    guidance_transforms: An optional sequence of guidance transforms that
      modifies the denoising function in a post-process fashion.
    apply_denoise_at_end: Whether to apply the denoise function for another time
      to the terminal state.
    return_full_paths: If `True`, the output will contain the complete sampling
      paths with axis 0 corresponding to diffusion times specified by `tspan`.
  """

  input_shape: tuple[int, ...]
  scheme: diffusion.Diffusion
  denoise_fn: DenoiseFn
  integrator: sde.SdeSolver
  tspan: Array
  guidance_transforms: Sequence[guidance.Transform]
  apply_denoise_at_end: bool = True
  return_full_paths: bool = False

  def generate(
      self,
      num_samples: int,
      rng: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Generate a batch of samples by solving an SDE."""
    if self.tspan.ndim != 1:
      raise ValueError("`tspan` must be a 1-d array.")

    init_rng, solver_rng = jax.random.split(rng)
    x_shape = (num_samples,) + self.input_shape
    t0, t1 = self.tspan[0], self.tspan[-1]
    x1 = jax.random.normal(init_rng, x_shape)
    x1 *= self.scheme.sigma(t0) * self.scheme.scale(t0)

    if cond is not None:
      rep_fn = lambda x: jnp.tile(x[None], (num_samples,) + (1,) * x.ndim)
      cond = jax.tree_map(rep_fn, cond)

    params = dict(
        drift=dict(guidance_inputs=guidance_inputs, cond=cond), diffusion={}
    )
    # Output of integrator must have time at axis 0.
    paths = self.integrator(self.dynamics, x1, self.tspan, solver_rng, params)

    if self.apply_denoise_at_end:
      denoise_fn = self.get_guided_denoise_fn(guidance_inputs=guidance_inputs)
      final = denoise_fn(
          jnp.divide(paths[-1], self.scheme.scale(t1)),
          self.scheme.sigma(t1),
          cond,
      )
      paths = jnp.concatenate([paths, final[None]], axis=0)

    return paths if self.return_full_paths else paths[-1]

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
