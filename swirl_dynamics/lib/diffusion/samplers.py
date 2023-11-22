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

"""Diffusion samplers."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import flax
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.lib.diffusion import guidance
from swirl_dynamics.lib.solvers import ode
from swirl_dynamics.lib.solvers import sde

Array = jax.Array
PyTree = Any
Cond = Mapping[str, PyTree] | None
DenoiseFn = Callable[[Array, Array, Cond], Array]
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

  def _score(x: Array, sigma: Array, cond: Cond = None) -> Array:
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

  start = diffusion.MAX_DIFFUION_TIME
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


class Sampler(Protocol):
  """Interface for diffusion samplers."""

  def generate(
      self, num_samples: int, rng: Array, **kwargs
  ) -> tuple[Array, Any]:
    """Generate a specified number of diffusion samples.

    Args:
      num_samples: The number of samples to generate.
      rng: The base rng for the generation process.
      **kwargs: Additional keyword arguments.

    Returns:
      A tuple of (`samples`, `aux`) where `samples` are the generated samples
      and `aux` contains the auxiliary output of the generation process.
    """
    ...


def _apply_guidance_transforms(
    denoise_fn: DenoiseFn,
    transforms: Sequence[guidance.Transform],
    guidance_inputs: Mapping[str, PyTree],
) -> DenoiseFn:
  for transform in transforms:
    denoise_fn = transform(denoise_fn, guidance_inputs)
  return denoise_fn


@flax.struct.dataclass
class OdeSampler:
  """Draw samples by solving an probabilistic flow ODE.

  Attributes:
    input_shape: The tensor shape of a sample (excluding the batch dimension).
    integrator: The ODE solver to use.
    scheme: The diffusion scheme which contains the scale and noise schedules to
      follow.
    denoise_fn: The denoising function; required to work on batched states and
      noise levels.
    guidance_transforms: An optional sequence of guidance transforms that
      modifies the denoising function in a post-process fashion.
    apply_denoise_at_end: Whether to apply the denoise function for another time
      to the terminal state.
  """

  input_shape: tuple[int, ...]
  integrator: ode.OdeSolver
  scheme: diffusion.Diffusion
  denoise_fn: DenoiseFn
  guidance_transforms: Sequence[guidance.Transform] = ()
  apply_denoise_at_end: bool = True

  def generate(
      self,
      num_samples: int,
      rng: Array,
      tspan: Array,
      cond: Cond = None,
      guidance_inputs: Mapping[str, Any] | None = None,
  ) -> tuple[Array, dict[str, Array]]:
    """Generate samples by solving the sampling ODE.

    Args:
      num_samples: The number of distinct samples to generate.
      rng: The jax random seed to be used for sampling.
      tspan: The time steps for integrating the ode.
      cond: The (explicit) conditioning inputs, i.e. those to be directly passed
        through the denoiser interface. These inputs should not come with a
        batch dimension - one will be created based on the number of samples (by
        repeating every leaf of the pytree) to generate.
      guidance_inputs: The inputs to the (a posteriori) guidance transforms.
        They will *not* be passed to the denoiser directly but rather used to
        "construct" a new denoising function. These inputs should also not come
        with a batch dimension but the exact shapes are handled inside the
        guidance transforms.

    Returns:
      A tuple of generated samples and auxiliary outputs. The latter currently
      consists of the entire ode trajectory.
    """
    if tspan.ndim != 1:
      raise ValueError("`tspan` must be a 1-d array.")

    x_shape = (num_samples,) + self.input_shape
    t0, t1 = tspan[0], tspan[-1]
    x1 = jax.random.normal(rng, x_shape)
    x1 *= self.scheme.sigma(t0) * self.scheme.scale(t0)
    if cond is not None:
      rep_fn = lambda x: jnp.tile(x[None], (num_samples,) + (1,) * x.ndim)
      cond = jax.tree_map(rep_fn, cond)

    params = dict(cond=cond, guidance_inputs=guidance_inputs)
    trajectories = self.integrator(self.dynamics, x1, tspan, params)
    samples = trajectories[-1]

    if self.apply_denoise_at_end:
      denoise_fn = _apply_guidance_transforms(
          self.denoise_fn,
          self.guidance_transforms,
          guidance_inputs=guidance_inputs,
      )
      samples = denoise_fn(
          jnp.divide(samples, self.scheme.scale(t1)),
          self.scheme.sigma(t1),
          cond,
      )
    return samples, {"trajectories": trajectories}

  @property
  def dynamics(self) -> ode.OdeDynamics:
    """The RHS of the sampling ODE.

    In score function (eq. 3 in Karras et al. https://arxiv.org/abs/2206.00364):

      dx = [ṡ(t)/s(t) x - s(t)² σ̇(t)σ(t) ∇pₜ(x)] dt,

    or, in terms of denoise function (eq. 81):

      dx = [σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) dt

    where s(t), σ(t) are the scale and noise schedule of the diffusion scheme.
    """

    def _dynamics(x: Array, t: Array, params: PyTree) -> Array:
      assert not t.ndim, "`t` must be a scalar."
      denoise_fn = _apply_guidance_transforms(
          self.denoise_fn,
          self.guidance_transforms,
          guidance_inputs=params["guidance_inputs"],
      )
      s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
      x_hat = jnp.divide(x, s)
      dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
      dlog_s_dt = dlog_dt(self.scheme.scale)(t)
      target = denoise_fn(x_hat, sigma, params["cond"])
      return (dlog_sigma_dt + dlog_s_dt) * x - dlog_sigma_dt * s * target

    return _dynamics


@flax.struct.dataclass
class SdeSampler:
  """Draws samples by solving an SDE."""

  input_shape: tuple[int, ...]
  integrator: sde.SdeSolver
  scheme: diffusion.Diffusion
  denoise_fn: DenoiseFn
  guidance_transforms: Sequence[guidance.Transform] = ()
  apply_denoise_at_end: bool = True

  def generate(
      self,
      num_samples: int,
      rng: Array,
      tspan: Array,
      cond: Cond = None,
      guidance_inputs: Mapping[str, Any] | None = None,
  ) -> tuple[Array, dict[str, Array]]:
    """Generate samples by solving an SDE.

    Args:
      num_samples: The number of distinct samples to generate.
      rng: The jax random seed to be used for sampling.
      tspan: The time steps for integrating the sde.
      cond: The (explicit) conditioning inputs, i.e. those to be directly passed
        through the denoiser interface. These inputs *should not come with a
        batch dimension* - one will be created based on the number of samples
        (by repeating every leaf of the pytree).
      guidance_inputs: The inputs to the (a posteriori) guidance transforms.
        They will *not* be passed to the denoiser directly but rather used to
        "construct" a new denoising function. Like `cond`, these inputs should
        not come with a batch dimension in principle, but the shape handling
        logic may be customized inside the specific guidance transforms.

    Returns:
      A tuple of generated samples and auxiliary outputs. The latter currently
      consists of the entire sde trajectory.
    """
    if tspan.ndim != 1:
      raise ValueError("`tspan` must be a 1-d array.")

    init_rng, solver_rng = jax.random.split(rng)
    x_shape = (num_samples,) + self.input_shape
    t0, t1 = tspan[0], tspan[-1]
    x1 = jax.random.normal(init_rng, x_shape)
    x1 *= self.scheme.sigma(t0) * self.scheme.scale(t0)
    if cond is not None:
      rep_fn = lambda x: jnp.tile(x[None], (num_samples,) + (1,) * x.ndim)
      cond = jax.tree_map(rep_fn, cond)

    params = dict(
        drift=dict(guidance_inputs=guidance_inputs, cond=cond), diffusion={}
    )
    trajectories = self.integrator(self.dynamics, x1, tspan, solver_rng, params)
    samples = trajectories[-1]

    if self.apply_denoise_at_end:
      denoise_fn = _apply_guidance_transforms(
          self.denoise_fn,
          self.guidance_transforms,
          guidance_inputs=guidance_inputs,
      )
      samples = denoise_fn(
          jnp.divide(samples, self.scheme.scale(t1)),
          self.scheme.sigma(t1),
          cond,
      )
    return samples, {"trajectories": trajectories}

  @property
  def dynamics(self) -> sde.SdeDynamics:
    """Terms of the sampling SDE.

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

    def _drift(x: Array, t: Array, params: PyTree) -> Array:
      assert not t.ndim, "`t` must be a scalar."
      denoise_fn = _apply_guidance_transforms(
          self.denoise_fn,
          self.guidance_transforms,
          guidance_inputs=params["guidance_inputs"],
      )
      s, sigma = self.scheme.scale(t), self.scheme.sigma(t)
      x_hat = jnp.divide(x, s)
      dlog_sigma_dt = dlog_dt(self.scheme.sigma)(t)
      dlog_s_dt = dlog_dt(self.scheme.scale)(t)
      drift = (2 * dlog_sigma_dt + dlog_s_dt) * x
      drift -= 2 * dlog_sigma_dt * s * denoise_fn(x_hat, sigma, params["cond"])
      return drift

    def _diffusion(x: Array, t: Array, params: PyTree) -> Array:
      del x, params
      assert not t.ndim, "`t` must be a scalar."
      dsquare_sigma_dt = dsquare_dt(self.scheme.sigma)(t)
      return jnp.sqrt(dsquare_sigma_dt) * self.scheme.scale(t)

    return sde.SdeDynamics(_drift, _diffusion)
