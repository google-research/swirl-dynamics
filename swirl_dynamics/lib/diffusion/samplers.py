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

"""Diffusion samplers."""

from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Any, Protocol, TypeAlias, TypeVar

import flax
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.lib.diffusion import guidance
from swirl_dynamics.lib.solvers import ode
from swirl_dynamics.lib.solvers import sde

Array: TypeAlias = jax.Array
ArrayMapping: TypeAlias = Mapping[str, Array]
Params: TypeAlias = Mapping[str, Any]
Shape: TypeAlias = tuple[int, ...]
Logp = TypeVar("Logp", Array, None)


class NoiseDist(Protocol):

  def __call__(self, rng: Array, *, shape: Shape) -> Array:
    ...


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


def eval_fn_and_trace_jac(
    fn: Callable[[Array], Array],
    x: Array,
    num_probes: int | None = None,
    rng: Array | None = None,
) -> tuple[Array, Array]:
  """Evaluates the value and the trace of the Jacobian of `fn` at `x`.

  If `num_probes` is `None`, the trace is evaluated exactly using
  vector-Jacobian products vmapped over rows of an identity matrix. The cost
  is proportional to the number of sample dimensions, which can be expensive.

  Otherwise, the trace is approximated with Hutchinson's trace estimator
  (eq. 7, https://arxiv.org/abs/1810.01367):

    Trace(dF/dx) = E [εᵀ(dF/dx)ε],

  where E denotes expection over ε, which are typically sampled from standard
  Gaussian distributions. The cost of this estimator is proportional to the
  number of probes (i.e. `num_probes`) instead of the sample dimension.

  Args:
    fn: The function to evaluate. It must take a single argument and return a
      single output of the same shape.
    x: The input to `fn`. The batch dimension should be the leading axis.
    num_probes: The number of probes to use for Hutchinson's trace estimator. If
      `None`, the trace is computed exactly.
    rng: Random key for sampling probes. Required if `num_probes` is not `None`.

  Returns:
    The evaluated function value and the trace of its Jacobian.
  """
  if num_probes is not None and rng is None:
    raise ValueError("`rng` is required if `num_probes` is not `None`.")

  fn_flat = lambda x_flat: fn(x_flat.reshape(*x.shape)).reshape(*x_flat.shape)
  y, vjp_fn = jax.vjp(fn_flat, x.reshape(x.shape[0], -1))

  if num_probes is None:
    # Exact trace computation.
    eye = jnp.stack([jnp.eye(y.shape[1])] * x.shape[0], axis=0)
    (dfndx,) = jax.vmap(vjp_fn, in_axes=1, out_axes=1)(eye)
    trace_jac = jnp.trace(dfndx, axis1=1, axis2=2)
  else:
    # Approximate trace computation using Hutchinson's trace estimator.
    eps = jax.random.bernoulli(rng, shape=(num_probes, *y.shape))
    eps = eps.astype(x.dtype) * 2 - 1  # scale to {-1, 1}
    (eps_vjp,) = jax.vmap(vjp_fn)(eps)
    trace_jac = jnp.mean(jnp.einsum("nbd,nbd->nb", eps_vjp, eps), axis=0)

  return y.reshape(*x.shape), trace_jac


def array_to_int(data: Array) -> Array:
  """Hashes an arbitrary array to an integer."""
  bits = jax.lax.bitcast_convert_type(data.ravel(), jnp.uint32)
  primes = jnp.arange(1, bits.size + 1, dtype=jnp.uint32) * 63689
  return jnp.sum(bits * primes).astype(jnp.uint64)


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
    noise_dist: The distribution to sample noise levels from. Defaults to a
      standard normal distribution.
  """

  input_shape: tuple[int, ...]
  scheme: diffusion.Diffusion
  denoise_fn: DenoiseFn
  tspan: Array
  guidance_transforms: Sequence[guidance.Transform] = ()
  apply_denoise_at_end: bool = True
  return_full_paths: bool = False
  noise_dist: NoiseDist = jax.random.normal

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
    x1 = self.noise_dist(init_rng, shape=x_shape)
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
  """Use an probability flow ODE to generate samples or compute log likelihood.

  Attributes:
    integrator: The ODE solver for solving the sampling ODE.
    num_probes: The number of probes to use for Hutchinson's trace estimator
      when computing the log likelihood of samples. If `None`, the trace is
      computed exactly.
  """

  integrator: ode.OdeSolver = ode.HeunsMethod()
  num_probes: int | None = None

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
    """The right-hand side function of the sampling ODE.

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
      rng: Array | None = None,
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
      rng: A random key for sampling probes. Required if using the approximate
        trace estimator (i.e. `num_probes` is not `None`).

    Returns:
      The computed log likelihood.
    """
    if inputs.shape[1:] != self.input_shape:
      raise ValueError(
          "`inputs` is expected to have shape (batch_size,) +"
          f" {self.input_shape}, but has shape {inputs.shape} instead."
      )

    batch_size = inputs.shape[0]
    dim = np.prod(self.input_shape)
    params = dict(cond=cond, guidance_inputs=guidance_inputs, rng=rng)

    dynamics_fn = self.log_likelihood_augmented_dynamics

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
  def log_likelihood_augmented_dynamics(self) -> ode.OdeDynamics:
    """Exact augmented dynamics for computing the log likelihood.

    The probability flow ODE dynamics is augmented with the dynamics of the log
    density, described by the instantaneous change of variable formula (see
    Grathwohl et al. https://arxiv.org/abs/1810.01367, eq. 2):

      d(log p(x))/dt = - Trace(dF/dx),

    where F denotes right-hand side of the probability flow.
    """

    def _augmented_dynamics(x: Array, t: Array, params: Params) -> Array:
      x = x[:, :-1]
      fn = functools.partial(self.dynamics, t=t, params=params)
      rng = jax.random.fold_in(params["rng"], array_to_int(t))
      dxdt, trace_jac = eval_fn_and_trace_jac(
          fn, x.reshape(x.shape[0], *self.input_shape), self.num_probes, rng
      )
      return jnp.concatenate(
          [dxdt.reshape(*x.shape), -trace_jac[:, None]], axis=-1
      )

    return _augmented_dynamics


@flax.struct.dataclass
class ExponentialOdeSampler(Sampler):
  """Draw samples by solving the sampling ODE with 1st-order exponential solver.

  Solves (see `OdeSampler.dynamics`)

    dx = [σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) dt

  where s(t), σ(t) are the scale and noise schedule of the diffusion scheme
  respectively. D() is the denoising function.
  """

  num_probes: int | None = None

  def __post_init__(self):
    if self.return_full_paths:
      raise ValueError(
          "`ExponentialOdeSampler` does not currently support returning full"
          " paths."
      )

  def denoise(
      self,
      noisy: Array,
      rng: Array,
      tspan: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Applies iterative denoising to given noisy states."""
    params = dict(guidance_inputs=guidance_inputs, cond=cond)

    def cond_fn(loop_state: tuple[int, Array]) -> bool:
      return loop_state[0] < len(tspan) - 1

    def body_fn(loop_state: tuple[int, Array]) -> tuple[int, Array]:
      i, xi = loop_state
      x_next = self.backward_step(
          x1=xi, t1=tspan[i], t0=tspan[i + 1], params=params
      )
      return i + 1, x_next

    _, samples = jax.lax.while_loop(cond_fn, body_fn, (0, noisy))
    return samples

  def backward_step(
      self, x1: Array, t1: Array, t0: Array, params: Params
  ) -> Array:
    """Applies one step of the exponential solver to the sampling ODE.

    The step formula reads

      x₀ = (s₀σ₀) / (s₁σ₁) x₁ + s₀(1 - σ₀ / σ₁) D(x₁/s₁, σ₁)

    where x₁ is the current state; s₁, σ₁ are the scale and noise levels at the
    current time; s₀, σ₀ are for the output time. t0 < t1.

    Args:
      x1: The current state.
      t1: The current time.
      t0: The output time.
      params: The parameters for the denoise function.

    Returns:
      The stepped state.
    """
    denoise_fn = self.get_guided_denoise_fn(
        guidance_inputs=params["guidance_inputs"]
    )
    s0, sigma0 = self.scheme.scale(t0), self.scheme.sigma(t0)
    s1, sigma1 = self.scheme.scale(t1), self.scheme.sigma(t1)
    x0 = (s0 * sigma0) / (s1 * sigma1) * x1
    x0 += (
        s0
        * (1 - sigma0 / sigma1)
        * denoise_fn(jnp.divide(x1, s1), sigma1, params["cond"])
    )
    return x0

  def forward_step(
      self,
      x0: Array,
      t0: Array,
      t1: Array,
      params: Params,
      logp0: Logp = None,
      rng: Array | None = None,
  ) -> tuple[Array, Logp]:
    """Applies one forward step of the exponential solver to the sampling ODE.

    The forward step formula reads

      x₁ = (s₁σ₁) / (s₀σ₀) x₀ + s₁(1 - σ₁ / σ₀) D(x₀/s₀, σ₀)

    with the associated log likelihood evolving as

      log p₁ = log p₀ + N (log (s₁ / s₀) + log (σ₁ / σ₀))
               + (log (σ₀ / σ₁) Trace(∂D/∂x)(x=x₀/s₀, σ₀=σ₀))

    where x₀ is the current state; s₀, σ₀ are the scale and noise levels at the
    current time; s₀, σ₀ are for the output time; N is the number of dimensions
    in x. t1 > t0.

    Args:
      x0: The current state.
      t0: The current time.
      t1: The output time.
      params: The parameters for the denoise function.
      logp0: The log likelihood of the current state. If provided, the ODE will
        be augmented to include the log likelihood.
      rng: A random key for sampling probes. Required if using the approximate
        trace estimator (i.e. `num_probes` is not `None`).

    Returns:
      The stepped state.
    """
    denoise_fn = self.get_guided_denoise_fn(
        guidance_inputs=params["guidance_inputs"]
    )
    s0, sigma0 = self.scheme.scale(t0), self.scheme.sigma(t0)
    s1, sigma1 = self.scheme.scale(t1), self.scheme.sigma(t1)

    if logp0 is None:
      denoised = denoise_fn(jnp.divide(x0, s0), sigma0, params["cond"])
      x1 = (s1 * sigma1) / (s0 * sigma0) * x0
      x1 += s1 * (1 - sigma1 / sigma0) * denoised
      logp1 = None

    else:
      denoise_fn0 = lambda x: denoise_fn(x, sigma0, params["cond"])
      denoised, trace_jac = eval_fn_and_trace_jac(
          fn=denoise_fn0,
          x=jnp.divide(x0, s0),
          num_probes=self.num_probes,
          rng=rng,
      )
      x1 = (s1 * sigma1) / (s0 * sigma0) * x0
      x1 += s1 * (1 - sigma1 / sigma0) * denoised
      logp1 = (
          logp0
          + np.prod(x1.shape[1:])
          * (jnp.log(s1 / s0) + jnp.log(sigma1 / sigma0))
          + (jnp.log(sigma0 / sigma1)) * trace_jac
      )
    return x1, logp1

  def compute_log_likelihood(
      self,
      inputs: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
      rng: Array | None = None,
  ) -> Array:
    """Computes the log likelihood of given data using the probability flow.

    Solves the augmented probability flow ODE forward (from zero to max noise
    level) with an 1st-order exponential solver.

    Args:
      inputs: Data whose log likelihood is computed. Expected to have shape
        `(batch, *input_shape)`.
      cond: Conditioning inputs for the denoise function. The batch dimension
        should match that of `data`.
      guidance_inputs: Inputs for constructing the guided denoising function.
      rng: A random key for sampling probes. Required if using the approximate
        trace estimator (i.e. `self.num_probes` is not `None`).

    Returns:
      The computed log likelihood.
    """
    if inputs.shape[1:] != self.input_shape:
      raise ValueError(
          "`inputs` is expected to have shape (batch_size,) +"
          f" {self.input_shape}, but has shape {inputs.shape} instead."
      )

    batch_size = inputs.shape[0]
    dim = np.prod(self.input_shape)
    params = dict(cond=cond, guidance_inputs=guidance_inputs)

    def cond_fn(loop_state: tuple[int, Array, Array]) -> bool:
      return loop_state[0] < len(self.tspan) - 1

    def body_fn(
        loop_state: tuple[int, Array, Array],
    ) -> tuple[int, Array, Array]:
      i, xi, logpi = loop_state
      x_next, logp_next = self.forward_step(
          x0=xi,
          t1=self.tspan[::-1][i],
          t0=self.tspan[::-1][i + 1],
          params=params,
          logp0=logpi,
          rng=rng,
      )
      return i + 1, x_next, logp_next

    _, x1, dlogp01 = jax.lax.while_loop(
        cond_fn, body_fn, (0, inputs, jnp.zeros((batch_size,)))
    )

    sigma1 = self.scheme.sigma(self.tspan[0]) * self.scheme.sigma(self.tspan[0])
    x1_flat = x1.reshape(batch_size, -1)
    log_p1 = (
        -dim / 2 * jnp.log(2 * jnp.pi)
        - dim * jnp.log(sigma1)
        - 0.5 * jnp.einsum("nd,nd->n", x1_flat, x1_flat) / jnp.square(sigma1)
    )
    # The sign before `dlogp01` is minus because the integration limits are
    # reversed compared to eq. 3 in https://arxiv.org/abs/1810.01367
    return log_p1 - dlogp01


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


@flax.struct.dataclass
class ExponentialSdeSampler(Sampler):
  """Draw samples by solving the sampling SDE with 1st-order exponential solver.

  Solves (see `SdeSampler.dynamics`)

    dx = [2 σ̇(t)/σ(t) + ṡ(t)/s(t)] x - [2 s(t)σ̇(t)/σ(t)] D(x/s(t), σ(t)) dt
         + s(t) √[2σ̇(t)σ(t)] dωₜ

  where s(t), σ(t) are the scale and noise schedule of the diffusion scheme
  respectively. D() is the denoising function.
  """

  def __post_init__(self):
    if self.return_full_paths:
      raise ValueError(
          "`ExponentialSdeSampler` does not currently support returning full"
          " paths."
      )

  def denoise(
      self,
      noisy: Array,
      rng: Array,
      tspan: Array,
      cond: ArrayMapping | None = None,
      guidance_inputs: ArrayMapping | None = None,
  ) -> Array:
    """Applies iterative denoising to given noisy states."""
    params = dict(guidance_inputs=guidance_inputs, cond=cond)
    rngs = jax.random.split(rng, len(tspan))

    def cond_fn(loop_state: tuple[int, Array]) -> bool:
      return loop_state[0] < len(tspan) - 1

    def body_fn(loop_state: tuple[int, Array]) -> tuple[int, Array]:
      i, xi = loop_state
      x_next = self.backward_step(
          x1=xi, t1=tspan[i], t0=tspan[i + 1], rng=rngs[i], params=params
      )
      return i + 1, x_next

    _, samples = jax.lax.while_loop(cond_fn, body_fn, (0, noisy))
    return samples

  def backward_step(
      self, x1: Array, t1: Array, t0: Array, rng: Array, params: Params
  ) -> Array:
    """Applies one backward step of the exponential solver to the sampling SDE.

    The step formula reads

      x₀ = (s₀σ²₀) / (s₁σ²₁) x₁ + s₀(1 - σ²₀ / σ²₁) D(x₁/s₁, σ₁)
          + s₀(σ₀ / σ₁) √[σ²₁ - σ²₀] ε

    where x₁ is the current state; s₁, σ₁ are the scale and noise levels at the
    current time; s₀, σ₀ are for the output time; ε ~ N(0, 1) is a standard
    normal random sample.

    Args:
      x1: The current state.
      t1: The current time.
      t0: The output time.
      rng: The random key for the drawing noise samples.
      params: The parameters for the denoise function.

    Returns:
      The stepped state.
    """
    denoise_fn = self.get_guided_denoise_fn(
        guidance_inputs=params["guidance_inputs"]
    )
    s0, sigma0 = self.scheme.scale(t0), self.scheme.sigma(t0)
    s1, sigma1 = self.scheme.scale(t1), self.scheme.sigma(t1)
    x0 = (s0 * sigma0**2) / (s1 * sigma1**2) * x1
    x0 += (
        s0
        * (1 - sigma0**2 / sigma1**2)
        * denoise_fn(jnp.divide(x1, s1), sigma1, params["cond"])
    )
    x0 += (
        s0
        * (sigma0 / sigma1)
        * jnp.sqrt(sigma1**2 - sigma0**2)
        * jax.random.normal(rng, x1.shape, x1.dtype)
    )
    return x0


@flax.struct.dataclass
class TStudentOdeSampler(OdeSampler):
  """Draw samples by solving an ODE from a t-student initial draw.

  Solves (see `OdeSampler.dynamics`).

  Attributes:
    df: The degrees of freedom of the t-Student distribution.
  """

  df: int = 3

  def __post_init__(self):
    noise_dist = functools.partial(jax.random.t, df=self.df)
    object.__setattr__(self, "noise_dist", noise_dist)


@flax.struct.dataclass
class TStudentSdeSampler(SdeSampler):
  """Draws samples by solving an SDE from a t-student initial draw.

  Attributes:
    df: The degrees of freedom of the t-student distribution.
  """

  df: int = 3

  def __post_init__(self):
    noise_dist = functools.partial(jax.random.t, df=self.df)
    object.__setattr__(self, "noise_dist", noise_dist)
