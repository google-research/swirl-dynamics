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

"""Different strategies for downweighting loss from rolled out steps.

For each method, the input `num_time_steps` (int) corresponds to the number of
steps to be included in a batch, where the first time step corresponds to the
initial condition. Thus `num_time_steps = k + 1`, where `k` is number of rollout
steps.
"""

import jax
from jax import numpy as jnp

Array = jax.Array


def geometric(
    num_time_steps: int, r: float = 0.1, clip: float = 10e-4
) -> Array:
  """Decay loss contribution as `loss * r^(k-1)`, where `r < 1`.

  Args:
    num_time_steps: steps to be included in a batch, i.e., rollout steps + 1
    r: geometric weight.
    clip: minimum weight.

  Returns:
    Rollout weights array.
  """
  assert r < 1, f"Geometric decay factor `r` ({r}) should be less than 1."
  assert clip > 0, f"Minimum weight `clip` ({clip}) should be greater than 0."
  return jnp.clip(r ** jnp.arange(0, num_time_steps - 1), a_min=clip)


def inverse_sqrt(num_time_steps: int, clip: float = 10e-4) -> Array:
  """Decay loss contribution as `loss * 1 / sqrt(k)`.

  Args:
    num_time_steps: steps to be included in a batch, i.e., rollout steps + 1
    clip: minimum weight.

  Returns:
    Rollout weights array.
  """
  assert clip > 0, f"Minimum weight `clip` ({clip}) should be greater than 0."
  return jnp.clip(jnp.arange(1, num_time_steps) ** -0.5, a_min=clip)


def inverse_squared(num_time_steps: int, clip: float = 10e-4) -> Array:
  """Decay loss contribution as `loss * 1 / (k^2)`.

  Args:
    num_time_steps: steps to be included in a batch, i.e., rollout steps + 1
    clip: minimum weight.

  Returns:
    Rollout weights array.
  """
  assert clip > 0, f"Minimum weight `clip` ({clip}) should be greater than 0."
  return jnp.clip(jnp.arange(1, num_time_steps) ** -2.0, a_min=clip)


def linear(num_time_steps: int, m: float = 1.0, clip: float = 10e-4) -> Array:
  """Decay loss contribution as `loss * 1 / m*k`.

  Args:
    num_time_steps: steps to be included in a batch, i.e., rollout steps + 1
    m: slope of decay.
    clip: minimum weight.

  Returns:
    Rollout weights array.
  """
  assert m > 0, f"Linear decay factor `m` ({m}) should be greater than 0."
  assert clip > 0, f"Minimum weight `clip` ({clip}) should be greater than 0."
  return jnp.clip((m * jnp.arange(1, num_time_steps)) ** -1.0, a_min=clip)


def no_weight(num_time_steps: int) -> Array:
  """No decay. All steps contribute equally."""
  return jnp.ones(num_time_steps - 1)
