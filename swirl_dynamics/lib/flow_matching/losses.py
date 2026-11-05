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

"""Loss functions for stochastic interpolants and flow matching.

References:
  [1] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.
      Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from swirl_dynamics.lib.flow_matching import interpolants

Array = jax.Array
StochasticInterpolantLossFn = Callable[
    [Array, Array, Array, Array, Array, interpolants.StochasticInterpolant],
    Array,
]


def score_loss(
    s_t: Array,
    t: Array,
    x_0: Array,
    x_1: Array,
    noise: Array,
    interpolant: interpolants.StochasticInterpolant,
) -> Array:
  """Computes the score matching loss on a batch of samples (Eq. 2.15 in [1]).

  Args:
    s_t: Predicted score at time `t`, with shape `(batch_size, *dims)`.
    t: Interpolation times with shape `(batch_size,)`.
    x_0: Initial distribution samples with shape `(batch_size, *dims)`.
    x_1: Target distribution samples with shape `(batch_size, *dims)`.
    noise: Noise sample modulated by `gamma(t)`, with shape
      `(batch_size, *dims)`.
    interpolant: Stochastic interpolant defining the target score.

  Returns:
    Mean squared error between `s_t` and the target score.
  """
  return jnp.mean(
      jnp.square(s_t - interpolant.calculate_target_score(t, x_0, x_1, noise))
  )


def denoising_loss(
    eta_t: Array,
    t: Array,
    x_0: Array,
    x_1: Array,
    z: Array,
    interpolant: interpolants.StochasticInterpolant,
) -> Array:
  """Computes the denoising loss on a batch of samples (Eq. 2.19 in [1]).

  Args:
    eta_t: Predicted noise at time `t`, with shape `(batch_size, *dims)`.
    t: Interpolation times with shape `(batch_size,)` (unused).
    x_0: Initial distribution samples with shape `(batch_size, *dims)` (unused).
    x_1: Target distribution samples with shape `(batch_size, *dims)` (unused).
    z: Ground-truth noise sample with shape `(batch_size, *dims)`.
    interpolant: Stochastic interpolant instance (unused).

  Returns:
    Mean squared error between `eta_t` and `z`.
  """
  del t, x_0, x_1, interpolant
  return jnp.mean(jnp.square(eta_t - z))


def velocity_loss(
    v_t: Array,
    t: Array,
    x_0: Array,
    x_1: Array,
    noise: Array,
    interpolant: interpolants.StochasticInterpolant,
) -> Array:
  """Computes the velocity matching loss on a batch (Eq. 2.12 in [1]).

  Args:
    v_t: Predicted velocity at time `t`, with shape `(batch_size, *dims)`.
    t: Interpolation times with shape `(batch_size,)`.
    x_0: Initial distribution samples with shape `(batch_size, *dims)`.
    x_1: Target distribution samples with shape `(batch_size, *dims)`.
    noise: Noise sample used in the interpolant time derivative, with shape
      `(batch_size, *dims)`.
    interpolant: Stochastic interpolant defining the target velocity.

  Returns:
    Mean squared error between `v_t` and the target time derivative.
  """
  target = interpolant.calculate_time_derivative_interpolant(t, x_0, x_1, noise)
  return jnp.mean(jnp.square(v_t - target))
