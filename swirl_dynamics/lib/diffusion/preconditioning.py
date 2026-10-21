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
import math
from typing import TypeAlias

from flax import linen as nn
import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


def compute_denoising_preconditioners(
    sigma: Array,
    sigma_data: float,
    x_shape: tuple[int, ...],
    df: float | None = None,
) -> tuple[Array, Array, Array, Array]:
  """Computes preconditioning coefficients for EDM and t-EDM denoising.

  Args:
    sigma: Noise level array (scalar or 1D batch array).
    sigma_data: Standard deviation of the target data distribution.
    x_shape: Shape of the model input tensor `x`.
    df: Optional degrees of freedom for Student-t noise (t-EDM). Must be > 2
      when provided.

  Returns:
    A 4-tuple `(c_in, c_out, c_skip, c_noise)` preserving `sigma.dtype`.
  """
  if sigma.ndim < 1:
    sigma = jnp.broadcast_to(sigma, (x_shape[0],))

  if sigma.ndim != 1 or x_shape[0] != sigma.shape[0]:
    raise ValueError(
        "sigma must be 1D and have the same leading (batch) dimension as x"
        f" ({x_shape[0]})!"
    )

  if df is not None:
    if df <= 2:
      raise ValueError(f"`df` must be greater than 2, got {df}.")
    df_scale = math.sqrt((df - 2) / df)
    scaled_sigma_data = sigma_data * df_scale
  else:
    df_scale = 1.0
    scaled_sigma_data = sigma_data

  total_std = jnp.hypot(scaled_sigma_data, sigma)
  c_skip = (scaled_sigma_data / total_std) ** 2
  c_out = sigma_data * (sigma / total_std)
  c_in = df_scale / total_std
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
    orig_dtype = x.dtype
    net_dtype = getattr(self.network, "dtype", orig_dtype)
    calc_dtype = jnp.result_type(x, sigma, jnp.float32)
    c_in, c_out, c_skip, c_noise = compute_denoising_preconditioners(
        jnp.asarray(sigma, dtype=calc_dtype), self.sigma_data, x.shape
    )
    x_in = jnp.multiply(c_in, x.astype(calc_dtype)).astype(net_dtype)
    f_x = self.network(x_in, c_noise, cond, is_training=is_training)
    return (
        jnp.multiply(c_skip, x.astype(calc_dtype))
        + jnp.multiply(c_out, f_x.astype(calc_dtype))
    ).astype(orig_dtype)


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

    orig_dtype = x.dtype
    net_dtype = getattr(self, "dtype", orig_dtype)
    calc_dtype = jnp.result_type(x, sigma, jnp.float32)
    c_in, c_out, c_skip, c_noise = compute_denoising_preconditioners(
        jnp.asarray(sigma, dtype=calc_dtype), self.sigma_data, x.shape
    )
    x_in = jnp.multiply(c_in, x.astype(calc_dtype)).astype(net_dtype)
    f_x = call_function(self, x_in, c_noise, cond, is_training=is_training)
    return (
        jnp.multiply(c_skip, x.astype(calc_dtype))
        + jnp.multiply(c_out, f_x.astype(calc_dtype))
    ).astype(orig_dtype)

  return wrapped_call
