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

"""This is a jax port of the Nvidia EDM2 [1] to swirl_dynamics.

References:
[1] Karras, Tero, et al. "Analyzing and improving the training dynamics of
diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 2024.
"""

import math

import jax
import jax.numpy as jnp

Array = jax.Array
DTypeLike = jax.typing.DTypeLike


def compute_norm(
    x: Array,
    axis: tuple[int, ...] | int = (1, 2, 3),
    dtype: DTypeLike = jnp.float32,
) -> Array:
  """Computes the normalization of the input array along the specified axes.

  Args:
    x: The input array.
    axis: The axes to normalize along.
    dtype: The data type to cast the input to.

  Returns:
    The norm of the input array along the specified axes, with a zero
    subgradient for zero-norm groups.
  """
  calc_dtype = jnp.result_type(x, dtype)
  x = x.astype(calc_dtype)
  squared_norm = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  is_zero = squared_norm == 0
  # Avoid differentiating sqrt at zero, even in the inactive where branch.
  safe_squared_norm = jnp.where(is_zero, 1.0, squared_norm)
  return jnp.where(is_zero, 0.0, jnp.sqrt(safe_squared_norm))


def normalize(
    x: Array,
    axis: tuple[int, ...] | int | None = None,
    eps: float = 1e-6,
    dtype: DTypeLike = jnp.float32,
) -> Array:
  """Computes the normalization of the input array along the specified axes."""
  if axis is None:
    axis = tuple(range(1, x.ndim))

  calc_dtype = jnp.result_type(x, dtype)
  norm = compute_norm(x, axis=axis, dtype=calc_dtype)
  if x.size == 0:
    return x.astype(calc_dtype)

  scale = math.sqrt(math.prod(norm.shape) / math.prod(x.shape))
  rescaled_norm = eps + scale * norm
  return x.astype(calc_dtype) / rescaled_norm


def mp_silu(x: Array) -> Array:
  """Computes the MP-SILU activation function.

  Args:
    x: The input array.

  Returns:
    The MP-SILU activation function with a normalization.
  """
  return jax.nn.silu(x) / 0.596


def mp_convex(x: Array, y: Array, t: float) -> Array:
  """Computes the MP-convex-SILU activation function.

  Args:
    x: The first input array.
    y: The second input array.
    t: The mixing parameter.

  Returns:
    A magnitude-preserving convex sum of two vectors.
  """
  convex_sum = x * (1 - t) + y * t
  return convex_sum / jnp.sqrt(jnp.square(1 - t) + jnp.square(t))


def mp_cat(x: Array, y: Array, dim: int = 1, t: float = 0.5) -> Array:
  """Magnitude-preserving concatenation.

  Args:
    x: The first input array.
    y: The second input array.
    dim: The dimension to concatenate along.
    t: The mixing parameter, must be in [0, 1]. If t=0, the output is x, if t=1,
      the output is y.

  Returns:
    A magnitude-preserving concatenation of two vectors along the specified
    dimension.
  """
  if t < 0.0 or t > 1.0:
    raise ValueError(f't must be in [0, 1], got {t}')

  # Compute the normalization factors depending on the number of elements in the
  # concatenation dimension.
  num_els_x = x.shape[dim]
  num_els_y = y.shape[dim]
  c = jnp.sqrt((num_els_x + num_els_y) / ((1 - t) ** 2 + t**2))
  weight_x = c / jnp.sqrt(num_els_x) * (1 - t)
  weight_y = c / jnp.sqrt(num_els_y) * t
  return jnp.concatenate([weight_x * x, weight_y * y], axis=dim)
