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

"""Interpolants for the stochastic interpolants framework.

References:
  [1] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.
      Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.
"""

from collections.abc import Callable
import dataclasses
from typing import Protocol

import jax
import jax.numpy as jnp

Array = jax.Array

vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))


class Interpolant(Protocol):
  """Protocol for defining a generic interpolant."""

  def __call__(self, t: Array, x_0: Array, x_1: Array, z: Array) -> Array:
    r"""Interpolates between `x_0` and `x_1` at time `t`.

    Assumes a zero-mean noise process with interpolant
    $x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z$.

    Args:
      t: Interpolation time with leading batch dimension.
      x_0: Sample from the initial distribution.
      x_1: Sample from the target distribution.
      z: Zero-mean noise sample with the same shape as `x_0` and `x_1`.

    Returns:
      Interpolated sample at time `t`.
    """
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolant(Interpolant):
  """Parameterized stochastic interpolant between two distributions.

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
    """Evaluates the interpolant at time `t`."""
    return self.calculate_interpolant(t, x_0, x_1, z)

  def calculate_interpolant(
      self, t: Array, x_0: Array, x_1: Array, z: Array
  ) -> Array:
    r"""Calculates $x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z$.

    Args:
      t: Interpolation time with leading batch dimension.
      x_0: Sample from the initial distribution.
      x_1: Sample from the target distribution.
      z: Noise sample with the same shape as `x_0` and `x_1`.

    Returns:
      The interpolated state at time `t`.
    """
    return (
        vmap_mult(self.alpha(t), x_0)
        + vmap_mult(self.beta(t), x_1)
        + vmap_mult(self.gamma(t), z)
    )

  def calculate_time_derivative_interpolant(
      self, t: Array, x_0: Array, x_1: Array, z: Array
  ) -> Array:
    r"""Calculates the time derivative $\dot{x}_t$ following Eq. (2.10) in [1].

    Args:
      t: Interpolation time with leading batch dimension.
      x_0: Sample from the initial distribution.
      x_1: Sample from the target distribution.
      z: Zero-mean noise sample with the same shape as `x_0` and `x_1`.

    Returns:
      The time derivative $\dot{\alpha}(t) x_0 + \dot{\beta}(t) x_1 +
      \dot{\gamma}(t) z$ at time `t`.
    """
    return (
        vmap_mult(self.alpha_dot(t), x_0)
        + vmap_mult(self.beta_dot(t), x_1)
        + vmap_mult(self.gamma_dot(t), z)
    )

  def calculate_target_score(
      self, t: Array, x_0: Array, x_1: Array, z: Array
  ) -> Array:
    r"""Calculates the target score $-\gamma(t)^{-1} z$ per Eq. (2.13) in [1].

    Args:
      t: Interpolation time with leading batch dimension.
      x_0: Sample from the initial distribution (unused).
      x_1: Sample from the target distribution (unused).
      z: Noise sample modulated by `gamma(t)`.

    Returns:
      The target score at time `t`.
    """
    del x_0, x_1
    return -vmap_mult(1 / self.gamma(t), z)


class LinearInterpolant(StochasticInterpolant):
  r"""Linear interpolant with Brownian-bridge noise schedule.

  Uses $\alpha(t) = 1 - t$, $\beta(t) = t$, and
  $\gamma(t) = \sigma \sqrt{2 t (1 - t)}$.
  """

  def __init__(self, sigma: float = 1.0):
    """Initializes the linear interpolant.

    Args:
      sigma: Scale factor for the stochastic noise schedule.
    """
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: sigma * jnp.sqrt(2 * t * (1 - t)),
        gamma_dot=lambda t: sigma * (1 - 2 * t) / jnp.sqrt(2 * t * (1 - t)),
    )


class LinearInterpolantSinusoidalNoise(StochasticInterpolant):
  r"""Linear interpolant with smooth squared-sine noise schedule.

  Uses $\alpha(t) = 1 - t$, $\beta(t) = t$, and
  $\gamma(t) = \sigma \sin^2(\pi t)$, whose derivative
  $\dot{\gamma}(t) = \sigma \pi \sin(2 \pi t)$ is smooth at $t = 0$ and $t = 1$.
  """

  def __init__(self, sigma: float = 1.0):
    """Initializes the sinusoidal-noise linear interpolant.

    Args:
      sigma: Scale factor for the stochastic noise schedule.
    """
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: sigma * jnp.square(jnp.sin(jnp.pi * t)),
        gamma_dot=lambda t: sigma * jnp.pi * jnp.sin(2 * jnp.pi * t),
    )


class TrigonometricInterpolant(StochasticInterpolant):
  r"""Deterministic variance-preserving trigonometric interpolant.

  Uses $\alpha(t) = \cos(\pi t / 2)$, $\beta(t) = \sin(\pi t / 2)$, and
  $\gamma(t) = 0$.
  """

  def __init__(self):
    """Initializes the trigonometric interpolant."""
    super().__init__(
        alpha=lambda t: jnp.cos(0.5 * jnp.pi * t),
        alpha_dot=lambda t: -0.5 * jnp.pi * jnp.sin(0.5 * jnp.pi * t),
        beta=lambda t: jnp.sin(0.5 * jnp.pi * t),
        beta_dot=lambda t: 0.5 * jnp.pi * jnp.cos(0.5 * jnp.pi * t),
        gamma=lambda t: 0 * t,
        gamma_dot=lambda t: 0 * t,
    )


class RectifiedFlow(StochasticInterpolant):
  r"""Deterministic linear interpolant ($\alpha(t) = 1 - t$, $\beta(t) = t$)."""

  def __init__(self):
    """Initializes the rectified flow interpolant."""
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: 0 * t,
        gamma_dot=lambda t: 0 * t,
    )
