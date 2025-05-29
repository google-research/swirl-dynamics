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

  def calculate_target_score(self, t: Array, x_0: Array, x_1: Array) -> Array:
    """Calculates the target score at time t.

    Args:
      t: The time at which the interpolation is performed.
      x_0: A sample from the initial distribution.
      x_1: A sample from the target distribution.

    Returns:
      The target score at time t.
    """
    del x_1
    # Making the assumption that x_0 is Gaussian and that we have an OU process.
    return vmap_mult(1 / self.alpha(t), -x_0)

  def __hash__(self):
    return hash((self.alpha, self.beta, self.gamma))

  def __eq__(self, other):
    return (
        self.alpha == other.alpha
        and self.beta == other.beta
        and self.gamma == other.gamma
    )


class LinearInterpolant(StochasticInterpolant):
  """Linear interpolant with a zero mean noise."""

  def __init__(self):
    super().__init__(
        alpha=lambda t: 1 - t,
        alpha_dot=lambda t: 0 * t - 1.0,
        beta=lambda t: t,
        beta_dot=lambda t: 0 * t + 1.0,
        gamma=lambda t: jnp.sqrt(2 * t * (1 - t)),
        gamma_dot=lambda t: (1 - 2 * t) / jnp.sqrt(2 * t * (1 - t)),
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
