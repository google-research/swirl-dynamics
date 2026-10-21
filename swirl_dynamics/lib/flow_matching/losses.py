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

"""Loss functions for stochastic interpolants and flow matching."""

from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.numpy as jnp
from swirl_dynamics.lib.flow_matching import interpolants

vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))

Array: TypeAlias = jax.Array
StochasticInterpolantLossFn: TypeAlias = Callable[
    [Array, Array, Array, Array, Array, interpolants.Interpolant], Array
]


def score_loss(
    s_t: Array,
    t: Array,
    x_0: Array,
    x_1: Array,
    noise: Array,
    interpolant: interpolants.StochasticInterpolant,
) -> Array:
  """Computes the loss for the score on a batch of samples."""
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
  """Computes the denoising loss for the score on a batch of samples."""
  del t, x_0, x_1, interpolant
  return jnp.mean((eta_t - z) ** 2)


def velocity_loss(
    v_t: Array,
    t: Array,
    x_0: Array,
    x_1: Array,
    noise: Array,
    interpolant: interpolants.StochasticInterpolant,
) -> Array:
  """Computes the loss for the velocity on a batch of samples."""
  target = interpolant.calculate_time_derivative_interpolant(t, x_0, x_1, noise)
  return jnp.mean(jnp.square(v_t - target))
