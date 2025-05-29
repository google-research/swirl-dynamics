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

"""Implements the losses for the stochastic interpolants model.

References:
[1]: Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.
"""

from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp
from swirl_dynamics.projects.debiasing.stochastic_interpolants import interpolants

vmap_mult = jax.vmap(jnp.multiply, in_axes=(0, 0))

Array: TypeAlias = jax.Array
StochasticInterpolantLossFn: TypeAlias = Callable[
    [Array, Array, Array, Array, Array, interpolants.Interpolant], Array
]


# TODO: create a loss class encapsulating the different losses.
def score_loss(
    net_eval: Array,
    t: Array,
    x_0: Array,
    x_1: Array,
    noise: Array,
    interpolant: interpolants.StochasticInterpolant,
) -> Array:
  """Compute the loss for the score on an batch of samples.

  Args:
    net_eval: The network output at time t given by the flow model. The shape
      should be the same as x_0 and x_1, i.e. (batch_size, *dimension),
      where *dimension is the dimension of the state.
    t: The time at which x_t and the velocity are computed. The shape should be
      the same as the leading dimension of x_0 and x_1, i.e. (batch_size,).
    x_0: A sample from the initial distribution. This array has the same shape
      as x_1 and net_eval, i.e. (batch_size, *dimension).
    x_1: A sample from the target distribution. This array has the same shape
      as x_0 and net_eval, i.e. (batch_size, *dimension).
    noise: The noise added to the gamma function.
    interpolant: The stochastic interpolant used to compute the loss.

  Returns:
    The mean squared error between the score given by the flow model and the
    target score.
  """
  return jnp.sum(net_eval**2) - 2 * jnp.sum(
      net_eval
      * (
          interpolant.calculate_target_score(t, x_0, x_1)
          + 1 / interpolant.gamma(t) * noise
      )
  )


def velocity_loss(
    v_t: Array,
    t: Array,
    x_0: Array,
    x_1: Array,
    noise: Array,
    interpolant: interpolants.StochasticInterpolant,
) -> Array:
  """Compute the loss for the velocity on a batch of samples.

  This computes the loss for the velocity at time t given by the flow model
  following Eq. 2.13 in [1].

  Args:
    v_t: The velocity at time t given by the flow model. The shape should be the
      same as x_0 and x_1, i.e. (batch_size, *dimension), where *dimension is
      the dimension of the state.
    t: The time at which x_t and the velocity are computed. The shape should be
      the same as the leading dimension of x_0 and x_1, i.e. (batch_size,).
    x_0: A sample from the initial distribution. This array has the same shape
      as v_t, i.e. (batch_size, *dimension).
    x_1: A sample from the target distribution. This array has the same shape
      as v_t, i.e. (batch_size, *dimension).
    noise: The noise added to the gamma function.
    interpolant: The stochastic interpolant used to compute the loss.

  Returns:
    The mean squared error between the velocity and the target velocity.
  """

  return jnp.mean(v_t**2) - 2 * jnp.mean(
      v_t
      * interpolant.calculate_time_derivative_interpolant(t, x_0, x_1, noise)
  )
