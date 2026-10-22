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

"""Interpolants for the stochastic interpolants framework (Albergo et al.)."""

from collections.abc import Callable
import dataclasses
from typing import Protocol, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array

vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))


class Interpolant(Protocol):
  """Protocol for defining a generic interpolant."""

  def __call__(self, t: Array, x_0: Array, x_1: Array, z: Array) -> Array:
    """Interpolates between x_0 and x_1 at time t."""
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolant(Interpolant):
  """Training a stochastic interpolants model.

  Attributes:
    alpha: Time-dependent interpolation weight of the initial distribution.
    alpha_dot: Time derivative of `alpha`.
    beta: Time-dependent interpolation weight of the target distribution.
    beta_dot: Time derivative of `beta`.
    gamma: Time-dependent weight of the stochastic part of the interpolant.
    gamma_dot: Time derivative of `gamma`.
  """

  alpha: Callable[[Array], Array]
  alpha_dot: Callable[[Array], Array]
  beta: Callable[[Array], Array]
  beta_dot: Callable[[Array], Array]
  gamma: Callable[[Array], Array] = lambda t: 0.0 * t
  gamma_dot: Callable[[Array], Array] = lambda t: 0.0 * t

  def __call__(self, t: Array, x_0: Array, x_1: Array, z: Array) -> Array:
    return self.calculate_interpolant(t, x_0, x_1, z)

  def calculate_interpolant(
      self, t: Array, x_0: Array, x_1: Array, z: Array
  ) -> Array:
    """Calculates the interpolant at time t."""
    return (
        vmap_mult(self.alpha(t), x_0)
        + vmap_mult(self.beta(t), x_1)
        + vmap_mult(self.gamma(t), z)
    )

  def calculate_time_derivative_interpolant(
      self, t: Array, x_0: Array, x_1: Array, z: Array
  ) -> Array:
    """Calculates the time derivative of the interpolant at time t."""
    return (
        vmap_mult(self.alpha_dot(t), x_0)
        + vmap_mult(self.beta_dot(t), x_1)
        + vmap_mult(self.gamma_dot(t), z)
    )

  def calculate_target_score(
      self, t: Array, x_0: Array, x_1: Array, z: Array
  ) -> Array:
    """Calculates the target score at time t."""
    del x_0, x_1
    return -vmap_mult(1 / self.gamma(t), z)

  def __hash__(self):
    return hash((self.alpha, self.beta, self.gamma))

  def __eq__(self, other):
    return (self.alpha, self.beta, self.gamma) == (
        other.alpha,
        other.beta,
        other.gamma,
    )


class LinearInterpolant(StochasticInterpolant):
  """Linear interpolant with a zero mean noise."""

  def __init__(self, sigma: float = 1.0):
    """Initializes the instance."""
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: sigma * jnp.sqrt(2 * t * (1 - t)),
        gamma_dot=lambda t: sigma * (1 - 2 * t) / jnp.sqrt(2 * t * (1 - t)),
    )


class LinearInterpolantSinusoidalNoise(StochasticInterpolant):
  """Linear interpolant with a sinusoidal noise schedule."""

  def __init__(self, sigma: float = 1.0):
    """Initializes the instance."""
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: sigma * jnp.square(jnp.sin(jnp.pi * t)),
        gamma_dot=lambda t: sigma * jnp.pi * jnp.sin(2 * jnp.pi * t),
    )


class TrigonometricInterpolant(StochasticInterpolant):
  """Trigonometric interpolant with no noise."""

  def __init__(self):
    """Initializes the instance."""
    super().__init__(
        alpha=lambda t: jnp.cos(0.5 * jnp.pi * t),
        alpha_dot=lambda t: -0.5 * jnp.pi * jnp.sin(0.5 * jnp.pi * t),
        beta=lambda t: jnp.sin(0.5 * jnp.pi * t),
        beta_dot=lambda t: 0.5 * jnp.pi * jnp.cos(0.5 * jnp.pi * t),
        gamma=lambda t: 0 * t,
        gamma_dot=lambda t: 0 * t,
    )


class RectifiedFlow(StochasticInterpolant):
  """Interpolant version of Rectified flow."""

  def __init__(self):
    """Initializes the instance."""
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: 0 * t,
        gamma_dot=lambda t: 0 * t,
    )
