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

"""Preconditioning for diffusion denoising models.

For details, see Appendix B of Karras et al (2022).
https://arxiv.org/abs/2206.00364

Two options are provided:
1. A flax module wrapping a non-preconditioned network, which would be applied
   at the time of module instantiation.

2. A decorator to apply to the `__call__` method of a denoising network at the
   time of class definition. This method is simpler if control over the class
   definition is available, and the non-preconditioned network is not required.
"""

from collections.abc import Mapping
import functools
from typing import TypeAlias

from flax import linen as nn
import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


def compute_denoising_preconditioners(
    sigma: Array, sigma_data: float, x_shape: tuple[int, ...]
) -> tuple[Array, Array, Array, Array]:
  """Computes preconditioning coefficients for EDM-based denoising."""
  if sigma.ndim < 1:
    sigma = jnp.broadcast_to(sigma, (x_shape[0],))

  if sigma.ndim != 1 or x_shape[0] != sigma.shape[0]:
    raise ValueError(
        "sigma must be 1D and have the same leading (batch) dimension as x"
        f" ({x_shape[0]})!"
    )

  total_var = sigma_data**2 + sigma**2
  c_skip = sigma_data**2 / total_var
  c_out = sigma * sigma_data / jnp.sqrt(total_var)
  c_in = 1 / jnp.sqrt(total_var)
  c_noise = 0.25 * jnp.log(sigma)

  def expand_dims(c: Array) -> Array:
    """Expand dimensions of 1D array `c` to be broadcastable with `x_shape`."""
    return jnp.expand_dims(c, axis=tuple(range(1, len(x_shape))))

  c_in = expand_dims(c_in)
  c_out = expand_dims(c_out)
  c_skip = expand_dims(c_skip)

  return c_in, c_out, c_skip, c_noise


class Preconditioned(nn.Module):
  """A wrapper module to apply preconditioning to a denoising network.

  Attributes:
    network: The network module to be wrapped, whose `__call__` method is
      expected to follow the same signature as the `__call__` method of this
      module.
    sigma_data: The standard deviation of the data population.
  """

  network: nn.Module
  sigma_data: float

  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: Mapping[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Applies preconditioning and wraps the network call.

    Args:
      x: The model input (e.g., noised sample).
      sigma: The noise level.
      cond: Conditional inputs for the network.
      is_training: Training mode flag.

    Returns:
      Output from the preconditioned network.
    """
    c_in, c_out, c_skip, c_noise = compute_denoising_preconditioners(
        sigma, self.sigma_data, x.shape
    )

    # Call the wrapped network instance.
    f_x = self.network(c_in * x, c_noise, cond, is_training=is_training)

    return c_skip * x + c_out * f_x


def precondition(call_function):
  """Decorator to apply EDM preconditioning to a network's `__call__` method.

  This decorator assumes that the attribute `self.sigma_data` exists and is a
  `float`, which represents the standard deviation of the data.

  Args:
    call_function: The function to be decorated.
  Returns:
    The decorated function.
  """

  @functools.wraps(call_function)
  def wrapped_call(
      self,
      x: Array,
      sigma: Array,
      cond: Mapping[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    if not hasattr(self, "sigma_data"):
      raise AttributeError(
          "Attribute `sigma_data` is required on `self` instance of type"
          f" {type(self)}."
      )

    c_in, c_out, c_skip, c_noise = compute_denoising_preconditioners(
        sigma, self.sigma_data, x.shape
    )

    # Call the wrapped function.
    f_x = call_function(self, c_in * x, c_noise, cond, is_training=is_training)

    return c_skip * x + c_out * f_x
  return wrapped_call
