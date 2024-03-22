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

"""Modules for diffusion models."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
ScheduleFn = Callable[[chex.Numeric], chex.Numeric]

MIN_DIFFUSION_TIME = 0.0
MAX_DIFFUSION_TIME = 1.0


@dataclasses.dataclass(frozen=True)
class InvertibleSchedule:
  """An invertible schedule.

  The schedule consists of a forward function that maps diffusion times to
  noise/scale/logsnr values, and an inverse that maps noise/scale/logsnr
  values back to the corresponding diffusion times, such that
  t = inverse(forward(t)). These functions should be monotonic wrt their input
  argument.

  Attributes:
    forward: A monotonic schedule function that maps diffusion times (scalar or
      array) to the noise/scale/logsnr values.
    inverse: The inverse schedule that maps scheduled values (scalar or array)
      back to the corresponding diffusion times.
  """

  forward: ScheduleFn
  inverse: ScheduleFn

  def __call__(self, t: chex.Numeric) -> chex.Numeric:
    return self.forward(t)


def sigma2logsnr(sigma: InvertibleSchedule) -> InvertibleSchedule:
  """Converts a sigma schedule to a logsnr schedule."""
  forward = lambda t: -2 * jnp.log(sigma(t))
  inverse = lambda logsnr: sigma.inverse(jnp.exp(-logsnr / 2))
  return InvertibleSchedule(forward, inverse)


def logsnr2sigma(logsnr: InvertibleSchedule) -> InvertibleSchedule:
  """Converts a logsnr schedule to a sigma schedule."""
  forward = lambda t: jnp.exp(-logsnr(t) / 2)
  inverse = lambda sigma: logsnr.inverse(-2 * jnp.log(sigma))
  return InvertibleSchedule(forward, inverse)


@dataclasses.dataclass(frozen=True)
class Diffusion:
  """Diffusion scheme.

  Fully parametrizes the Gaussian perturbation kernel:

    p(x_t|x_0) = N(x_t; s_t * x_0, s_t * σ_t * I)

  where x_0 and x_t are original and noised samples. s_t and σ_t are the scale
  and noise schedules. t ∈ [0, 1]. I denotes the identity matrix. This
  particular parametrization follows Karras et al.
  (https://arxiv.org/abs/2206.00364).

  Attributes:
    scale: The scale schedule (as a function of t).
    sigma: The noise schedule (as monotonically increasing function of t).
    logsnr: The log signal-to-noise (LogSNR) schedule equivalent to the sigma
      schedule.
    sigma_max: The maximum noise level of the scheme.
  """

  scale: ScheduleFn
  sigma: InvertibleSchedule

  @property
  def logsnr(self) -> InvertibleSchedule:
    return logsnr2sigma(self.sigma)

  @property
  def sigma_max(self) -> chex.Numeric:
    return self.sigma(MAX_DIFFUSION_TIME)

  @classmethod
  def create_variance_preserving(
      cls, sigma: InvertibleSchedule, data_std: float = 1.0
  ) -> Diffusion:
    """Creates a variance preserving diffusion scheme.

    Derive the scale schedule s_t from the noise schedule σ_t such that
    s_t^2 * (σ_d^2 + σ_t^2) (where σ_d denotes data standard deviation) remains
    constant (at σ_d^2) for all t. See Song et al.
    (https://arxiv.org/abs/2011.13456) for reference.

    Args:
      sigma: The sigma (noise) schedule.
      data_std: The standard deviation (scalar) of the data.

    Returns:
      A variance preserving diffusion scheme.
    """
    var = jnp.square(data_std)
    scale = lambda t: jnp.sqrt(var / (var + jnp.square(sigma(t))))
    return cls(scale=scale, sigma=sigma)

  @classmethod
  def create_variance_exploding(
      cls, sigma: InvertibleSchedule, data_std: float = 1.0
  ) -> Diffusion:
    """Creates a variance exploding diffusion scheme.

    Scale s_t is kept constant at 1. The noise schedule is scaled by the
    data standard deviation such that the amount of noise added is proportional
    to the data variation. See Song et al.
    (https://arxiv.org/abs/2011.13456) for reference.

    Args:
      sigma: The sigma (noise) schedule.
      data_std: The standard deviation (scalar) of the data.

    Returns:
      A variance exploding diffusion scheme.
    """
    scaled_forward = lambda t: sigma(t) * data_std
    scaled_inverse = lambda y: sigma.inverse(y / data_std)
    scaled_sigma = InvertibleSchedule(scaled_forward, scaled_inverse)
    return cls(scale=jnp.ones_like, sigma=scaled_sigma)


def create_variance_preserving_scheme(
    sigma: InvertibleSchedule, data_std: float = 1.0
) -> Diffusion:
  """Alias for `Diffusion.create_variance_preserving`."""
  return Diffusion.create_variance_preserving(sigma, data_std)


def create_variance_exploding_scheme(
    sigma: InvertibleSchedule, data_std: float = 1.0
) -> Diffusion:
  """Alias for `Diffusion.create_variance_exploding`."""
  return Diffusion.create_variance_exploding(sigma, data_std)


def _linear_rescale(
    in_min: float, in_max: float, out_min: float, out_max: float
) -> InvertibleSchedule:
  """Linearly rescale input between specified ranges."""
  in_range = in_max - in_min
  out_range = out_max - out_min
  fwd = lambda x: out_min + (x - in_min) / in_range * out_range
  inv = lambda y: in_min + (y - out_min) / out_range * in_range
  return InvertibleSchedule(fwd, inv)


def tangent_noise_schedule(
    clip_max: float = 100.0, start: float = 0.0, end: float = 1.5
) -> InvertibleSchedule:
  """Tangent noise schedule.

  This schedule is obtained by taking the section of the tan(t) function
  inside domain [`start`, `end`], and applying linear rescaling such that the
  input domain is [0, 1] and output range is [0, `clip_max`].

  This is really the "cosine" schedule proposed in Dhariwal and Nicol
  (https://arxiv.org/abs/2105.05233). The original schedule is
  a cosine function in γ, i.e. γ = cos(pi/2 * t) for t in [0, 1]. With
  γ = 1 / (σ^2 + 1), the corresponding σ schedule is a tangent function.

  The "shifted" cosine schedule proposed in Hoogeboom et al.
  (https://arxiv.org/abs/2301.11093) simply corresponds to adjusting
  the `clip_max` parameter. Empirical evidence suggests that one should consider
  increasing this maximum noise level when modeling higher resolution images.

  Args:
    clip_max: The maximum noise level in the schedule.
    start: The left endpoint of the tangent function domain used.
    end: The right endpoint of the tangent function domain used.

  Returns:
    A tangent noise schedule.
  """
  if not -np.pi / 2 < start < end < np.pi / 2:
    raise ValueError("Must have -pi/2 < `start` < `end` < pi/2.")

  in_rescale = _linear_rescale(
      in_min=0.0, in_max=MAX_DIFFUSION_TIME, out_min=start, out_max=end
  )
  out_rescale = _linear_rescale(
      in_min=np.tan(start), in_max=np.tan(end), out_min=0.0, out_max=clip_max
  )
  sigma = lambda t: out_rescale(jnp.tan(in_rescale(t)))
  inverse = lambda y: in_rescale.inverse(jnp.arctan(out_rescale.inverse(y)))
  return InvertibleSchedule(sigma, inverse)


def power_noise_schedule(
    clip_max: float = 100.0,
    p: float = 1.0,
    start: float = 0.0,
    end: float = 1.0,
) -> InvertibleSchedule:
  """Power noise schedule.

  This schedule is obtained by taking the section of the t^p (where p > 0)
  function inside domain [`start`, `end`], and applying linear rescaling such
  that the input domain is [0, 1] and output range is [0, `clip_max`].

  Variance exploding schedules in Karras et al.
  (https://arxiv.org/abs/2206.00364) and Song et al.
  (https://arxiv.org/abs/2011.13456) use p = 1 and p = 0.5 respectively.

  Args:
    clip_max: The maximum noise level in the schedule.
    p: The degree of power schedule.
    start: The left endpoint of the power function domain used.
    end: The right endpoint of the power function domain used.

  Returns:
    A power noise schedule.
  """
  if not (0 <= start < end and p > 0):
    raise ValueError("Must have `p` > 0 and 0 <= `start` < `end`.")

  in_rescale = _linear_rescale(
      in_min=MIN_DIFFUSION_TIME,
      in_max=MAX_DIFFUSION_TIME,
      out_min=start,
      out_max=end,
  )
  out_rescale = _linear_rescale(
      in_min=start**p, in_max=end**p, out_min=0.0, out_max=clip_max
  )
  sigma = lambda t: out_rescale(jnp.power(in_rescale(t), p))
  inverse = lambda y: in_rescale.inverse(  # pylint:disable=g-long-lambda
      jnp.power(out_rescale.inverse(y), 1 / p)
  )
  return InvertibleSchedule(sigma, inverse)


def exponential_noise_schedule(
    clip_max: float = 100.0,
    base: float = np.e**0.5,
    start: float = 0.0,
    end: float = 5.0,
) -> InvertibleSchedule:
  """Exponential noise schedule.

  This schedule is obtained by taking the section of the base^t (where base > 1)
  function inside domain [`start`, `end`], and applying linear rescaling such
  that the input domain is [0, 1] and output range is [0, `clip_max`]. This
  schedule is always a convex function.

  Args:
    clip_max: The maximum noise level in the schedule.
    base: The base of the exponential. Defaults to sqrt(e) so that σ^2 follows
      schedule exp(t).
    start: The left endpoint of the exponential function domain used.
    end: The right endpoint of the exponential function domain used.

  Returns:
    An exponential noise schedule.
  """
  if not (start < end and base > 1.0):
    raise ValueError("Must have `base` > 1 and `start` < `end`.")

  in_rescale = _linear_rescale(
      in_min=MIN_DIFFUSION_TIME,
      in_max=MAX_DIFFUSION_TIME,
      out_min=start,
      out_max=end,
  )
  out_rescale = _linear_rescale(
      in_min=base**start, in_max=base**end, out_min=0.0, out_max=clip_max
  )
  sigma = lambda t: out_rescale(jnp.power(base, in_rescale(t)))
  inverse = lambda y: in_rescale.inverse(  # pylint:disable=g-long-lambda
      jnp.log(out_rescale.inverse(y)) / jnp.log(base)
  )
  return InvertibleSchedule(sigma, inverse)


# ********************
# Noise sampling
# ********************


class NoiseLevelSampling(Protocol):

  def __call__(self, rng: jax.Array, shape: tuple[int, ...]) -> Array:
    """Samples noise levels for training."""
    ...


def _uniform_samples(
    rng: jax.Array,
    shape: tuple[int, ...],
    uniform_grid: bool,
) -> Array:
  """Generates samples from uniform distribution on [0, 1]."""
  if uniform_grid:
    s0 = jax.random.uniform(rng, dtype=jnp.float32)
    grid = jnp.linspace(0, 1, np.prod(shape), endpoint=False, dtype=jnp.float32)
    samples = jnp.reshape(jnp.remainder(grid + s0, 1), shape)
  else:
    samples = jax.random.uniform(rng, shape, dtype=jnp.float32)
  return samples


def log_uniform_sampling(
    scheme: Diffusion, clip_min: float = 1e-4, uniform_grid: bool = False
) -> NoiseLevelSampling:
  """Samples noise whose natural log follows a uniform distribution."""

  def _noise_sampling(rng: jax.Array, shape: tuple[int, ...]) -> Array:
    samples = _uniform_samples(rng, shape, uniform_grid)
    log_min, log_max = jnp.log(clip_min), jnp.log(scheme.sigma_max)
    samples = (log_max - log_min) * samples + log_min
    return jnp.exp(samples)

  return _noise_sampling


def time_uniform_sampling(
    scheme: Diffusion, clip_min: float = 1e-4, uniform_grid: bool = False
) -> NoiseLevelSampling:
  """Samples noise from a uniform distribution in t."""

  def _noise_sampling(rng: jax.Array, shape: tuple[int, ...]) -> Array:
    samples = _uniform_samples(rng, shape, uniform_grid)
    min_t = scheme.sigma.inverse(clip_min)
    samples = (MAX_DIFFUSION_TIME - min_t) * samples + min_t
    return jnp.asarray(scheme.sigma(samples))

  return _noise_sampling


def normal_sampling(
    scheme: Diffusion,
    clip_min: float = 1e-4,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> NoiseLevelSampling:
  """Samples noise from a normal distribution.

  This noise sampling is first used in Karras et al.
  (https://arxiv.org/abs/2206.00364). The default mean and standard deviation
  settings are designed for diffusion scheme with sigma_max = 80.

  Args:
    scheme: The diffusion scheme.
    clip_min: The minimum noise cutoff.
    p_mean: The mean of the sampling normal distribution.
    p_std: The standard deviation of the sampling normal distribution.

  Returns:
    A normal sampling function.
  """

  def _noise_sampler(rng: jax.Array, shape: tuple[int, ...]) -> Array:
    log_sigma = jax.random.normal(rng, shape, dtype=jnp.float32)
    log_sigma = p_mean + p_std * log_sigma
    return jnp.clip(jnp.exp(log_sigma), clip_min, scheme.sigma_max)

  return _noise_sampler


# ********************
# Noise weighting
# ********************


class NoiseLossWeighting(Protocol):

  def __call__(self, sigma: chex.Numeric) -> Array:
    """Returns weights of the input noise levels in the loss function."""
    ...


def inverse_squared_weighting(sigma: Array) -> Array:
  return 1 / jnp.square(sigma)


def edm_weighting(data_std: float = 1.0) -> NoiseLossWeighting:
  """Weighting proposed in Karras et al. (https://arxiv.org/abs/2206.00364).

  This weighting ensures the effective weights are uniform across noise levels
  (see appendix B.6, eqns 139 to 144).

  Args:
    data_std: the standard deviation of the data.

  Returns:
    The weighting function.
  """

  def _weight_fn(sigma: Array) -> Array:
    return (jnp.square(data_std) + jnp.square(sigma)) / jnp.square(
        data_std * sigma
    )

  return _weight_fn
