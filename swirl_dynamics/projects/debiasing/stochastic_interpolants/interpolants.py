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

We implement the interpolant described in [1].

References:

[1]: Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.
"""

import dataclasses
from typing import Callable, Protocol, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array

# Helper function to avoid broadcasting.
vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))


class Interpolant(Protocol):
  """Protocol for defining a generic interpolant."""

  def __call__(self, t: Array, x_0: Array, x_1: Array, z: Array) -> Array:
    r"""Interpolates between x_0 and x_1 at time t.

    We assume that the noising process has a zero mean, and the interpolant is
    defined as
    $ x_t = \alpha(t) * x_0 + \beta(t) * x_1 + \gamma(t) z.$

    Args:
      t: The time at which the interpolation is performed.
      x_0: A sample from the initial distribution.
      x_1: A sample from the target distribution.
      z: A noise vector with the same shape as x_0 and x_1.
    """
    ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class StochasticInterpolant(Interpolant):
  """Training a stochastic interpolants model.

  Attributes:
    alpha: A time dependent function that controls the interpolation coeficients
      of the initial distribution.
    alpha_dot: The time derivative of the alpha function.
    beta: A time dependent function that controls the interpolation coeficients
      of the target distribution.
    beta_dot: the time derivative of the beta function.
    gamma: A time dependent function that controls the weights of the stochastic
      part of the interpolant. By default we set it to 0.
    gamma_dot: The time derivative of the gamma function. By default we set it
      to 0.
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
    r"""Calculates the interpolant at time t.

    Args:
      t: The time at which the interpolation is performed.
      x_0: A sample from the initial distribution.
      x_1: A sample from the target distribution.
      z: A noise vector with the same shape as x_0 and x_1.

    Returns:
      The interpolant at time t follwing
      $ x_t = \alpha(t) * x_0 + \beta(t) * x_1 + \gamma(t) z$
    """
    return (
        vmap_mult(self.alpha(t), x_0)
        + vmap_mult(self.beta(t), x_1)
        + vmap_mult(self.gamma(t), z)
    )

  def calculate_time_derivative_interpolant(
      self, t: Array, x_0: Array, x_1: Array, z: Array
  ) -> Array:
    r"""Calculates the time derivative of the interpolant at time t.

    Args:
      t: The time at which the interpolation is performed.
      x_0: A sample from the initial distribution.
      x_1: A sample from the target distribution.
      z: A noise vector with the same shape as x_0 and x_1.

    Returns:
      The time derivative of the interpolant at time t following Eq. (2.10) in
      [1]. Where we assume that the noising process has a zero mean.
      $ v_t = \alpha_dot(t) * x_0 + beta_dot(t) * x_1 + gamma_dot(t) z$.
    """
    return (
        vmap_mult(self.alpha_dot(t), x_0)
        + vmap_mult(self.beta_dot(t), x_1)
        + vmap_mult(self.gamma_dot(t), z)
    )

  def calculate_target_score(
      self, t: Array, x_0: Array, x_1: Array, z: Array
  ) -> Array:
    r"""Calculates the target score at time t.

    Here we use the fact that
    $$ \nabla \log \rho(t, x) = - \gamma^{-1}(t) \mathbb{E}(z | x_t = x),$$
    which is equation 2.13 in [1].

    Args:
      t: The time at which the interpolation is performed.
      x_0: A sample from the initial distribution.
      x_1: A sample from the target distribution.
      z: The noise added to the gamma function.

    Returns:
      The target score at time t.
    """
    del x_0, x_1
    return - vmap_mult(1 / self.gamma(t), z)

  def __hash__(self):
    return hash((self.alpha, self.beta, self.gamma))

  def __eq__(self, other):
    return (
        self.alpha == other.alpha
        and self.beta == other.beta
        and self.gamma == other.gamma
    )


class LinearInterpolant(StochasticInterpolant):
  r"""Linear interpolant with a zero mean noise.

  This interpolant is defined as
  $$ x_t = \alpha(t) * x_0 + \beta(t) * x_1 + \gamma(t) z,$$
  where $\alpha(t) = 1 - t, \beta(t) = t$, $\gamma(t) = \sqrt{2t(1-t)}$,
  and $z \sim N(0, \sigma^2)$.
  """

  def __init__(self, sigma: float = 1.0):
    """Initializes the linear interpolant.

    Args:
      sigma: The standard deviation of the noise.

    Returns:
      The linear interpolant.
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
  r"""Linear interpolant with a sinusoidal noise schedule.

  This is similar to the linear interpolant, but with a sinusoidal noise
  schedule, following $\gamma(t) = \sigma \sin(2\pi t)$. The main advantage of
  this interpolant is that the derivative of the noise schedule does not have a
  discontinuity at $t=0$ or $t=1$.
  """

  def __init__(self, sigma: float = 1.0):
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: sigma * jnp.square(jnp.sin(jnp.pi * t)),
        gamma_dot=lambda t: sigma * jnp.pi * jnp.sin(2 * jnp.pi * t),
    )


class TrigonometricInterpolant(StochasticInterpolant):
  r"""Trigonometric interpolant with no noise.

  This interpolant is defined as
  $$ x_t = \alpha(t) * x_0 + \beta(t) * x_1 + \gamma(t) z,$$
  where $\alpha(t) = \cos(0.5 \pi t), \beta(t) = \sin(0.5 \pi t)$,
  $\gamma(t) = 0$,
  """

  def __init__(self):
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
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: 0 * t,
        gamma_dot=lambda t: 0 * t,
    )
