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

# coding=utf-8
"""WENO auxiliary functions following [1].

This library assumes that the double precision is turned on. This can be done
by setting the value of the flag jax_enable_x64 to true by adding

>>> from jax import config
>>> config.update("jax_enable_x64", True)

to the beginning of the code.

Refs:

[1] D. Bezgin, S.J. Schmidt and N.A. Klaus, "WENO3-NN: A maximum-order
three-point data-driven weighted essentially non-oscillatory scheme",
Journal of Computational Physics, Volume 452, Issue C, Mar 2022.

[2] G.S. Jiang, C.W. Shu, "Efficient implementation of weighted ENO schemes",
Journal of Computational Physics, 126 (1) (1996) 202-228.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp

Array = jax.Array


def upwind_weights(order: int = 3) -> tuple[Array, Array]:
  """Computation of the upwind weights.

  Args:
    order: Order of the interpolation of the ENO polynomials.

  Returns:
    The optimal weights for the ENO polynomials for third order reconstruction
    for smooth functions.
  """

  if order not in [3, 5]:
    raise ValueError("Only 3rd and 5th order polynomials are supported, not ",
                     f"order {order}.")

  d_minus, d_plus = jnp.array([0.0]), jnp.array([0.0])

  if order == 3:
    d_minus = jnp.array([2.0, 1.0], dtype=jnp.float64) / 3.0
    d_plus = jnp.array([1.0, 2.0], dtype=jnp.float64) / 3.0

  elif order == 5:
    d_minus = 0.1 * jnp.array([3.0, 6.0, 1.0], dtype=jnp.float64)
    d_plus = 0.1 * jnp.array([1.0, 6.0, 3.0], dtype=jnp.float64)

  return (d_minus, d_plus)  # pytype: disable=bad-return-type  # jnp-type


def beta(u_bar: Array, order: int = 3) -> Array:
  """Computation of the smoothness indicators in (10) of [1].

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    order: Order of the interpolation of the ENO polynomials.

  Returns:
    The indicators of the local smoothness, β, of the function.
  """
  if order not in [3, 5]:
    raise ValueError("Only 3rd and 5th order polynomials are supported, not ",
                     f"order {order}.")

  beta_array = jnp.array([0., 0.])

  if order == 3:
    beta_0 = jnp.square(u_bar[1] - u_bar[0])
    beta_1 = jnp.square(u_bar[2] - u_bar[1])

    beta_array = jnp.array([beta_0, beta_1])

  elif order == 5:
    # from Eqs. (3.1), (3.2), and (3.3) in [2].
    beta_0 = (13. / 12. * (jnp.square(u_bar[0] - 2.0 * u_bar[1]+ u_bar[2])) +
              0.25 *  (jnp.square(u_bar[0] - 4.0 * u_bar[1]+ 3 * u_bar[2])))
    beta_1 = (13. / 12. * (jnp.square(u_bar[1] - 2.0 * u_bar[2]+ u_bar[3])) +
              0.25 * (jnp.square(u_bar[1] - u_bar[3])))
    beta_2 = (13. / 12. * (jnp.square(u_bar[2] - 2.0 * u_bar[3]+ u_bar[4])) +
              0.25 *  (jnp.square(3 * u_bar[2] - 4.0 * u_bar[3] + u_bar[4])))

    beta_array = jnp.array([beta_0, beta_1, beta_2])

  return beta_array


def omega_plus(u_bar: Array,
               order: int = 3,
               p: int = 2,
               eps: jnp.float64 = 1e-15,
               ) -> Array:
  """Computes the WENO weights in the interpolation.

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    order: Order of the interpolation of the ENO polynomials.
    p: Polynomial degree of the smoothness indicator.
    eps: Regularizer to avoid division by zero.

  Returns:
    The interpolation weights for the ENO polynomials.
  """
  # Unpacking the computation of β as described in Eq. (10) in [1].
  beta_w = beta(u_bar, order)

  # Extracting the upwind weights d₀ and d₁ as described in Eq. (8) in [1].
  _, d_plus = upwind_weights(order)

  # Computing α as described in Eq. (8) in [1].
  alpha = d_plus / jnp.power(beta_w + eps, p)

  # Computing ω as described in Eq. (8) in [1].
  alpha_sum = jnp.sum(alpha)
  omega = alpha / alpha_sum

  return omega


def interpolants_plus(u_bar: Array, order: int = 3) -> Array:
  """Computes the polynomial interpolants.

  We follow [2] to compute the interpolants, using the notation in [1].

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    order: Order of the underlying interpolation for smooth functions.

  Returns:
    The ENO polynomials evaluated at x_{n+1/2}.
  """
  if len(u_bar) != order:
    raise ValueError(f"Input size ({len(u_bar)}) and polynomial order ",
                     f"({order}) do not match.")

  eno_polynomials = jnp.array([0., 0.])

  if order == 3:
    # u^0_{i+1/2} = 0.5( -u_{i-1} + 3 u_{i}).
    u_plus_0 = 0.5 * (-u_bar[0] + 3 * u_bar[1])
    # u^1_{i+1/2} = 0.5(u_{i} + u_{i+1}).
    u_plus_1 = 0.5 * (u_bar[1] + u_bar[2])

    eno_polynomials = jnp.array([u_plus_0, u_plus_1])

  elif order == 5:

    # u^0_{i+1/2} = ( -2 u_{i-2} - 7 u_{i-1} + 11 * u_{0}) / 6.
    u_plus_0 = (2 * u_bar[0] - 7 * u_bar[1] + 11 * u_bar[2]) / 6.
    # u^1_{i+1/2} = ( - u_{i-1} + 5 u_{i} + 2 * u_{i+1}) / 6.
    u_plus_1 = (-u_bar[1] + 5 * u_bar[2] + 2 * u_bar[3]) / 6.
    # u^1_{i+1/2} = ( 2 u_{i} + 5 u_{i+1} -1 u_{i+2}) / 6.
    u_plus_2 = (2 * u_bar[2] + 5 * u_bar[3] - u_bar[4]) / 6.

    eno_polynomials = jnp.array([u_plus_0, u_plus_1, u_plus_2])

  return eno_polynomials


def interpolants_minus(u_bar: Array, order: int = 3) -> Array:
  """Computes the third order interpolants in Eq. (7) in [1].

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    order: Order of the underlying interpolation for smooth functions.

  Returns:
    The ENO polynomials evaluated at x_{n-1/2}.
  """

  if len(u_bar) != order:
    raise ValueError(f"Input size ({len(u_bar)}) and polynomial order ",
                     f"({order}) do not match.")

  eno_polynomials = jnp.array([0., 0.])

  if order == 3:
    # u^1_{i-1/2} = 0.5(u_{i-1} + u_{i}).
    u_minus_0 = 0.5 * (u_bar[0] + u_bar[1])
    # u^0_{i-1/2} = 0.5( -u_{i+1} + 3 u_{i}).
    u_minus_1 = 0.5 * (3 * u_bar[1] - u_bar[2])

    eno_polynomials = jnp.array([u_minus_1, u_minus_0])

  elif order == 5:
    # u^1_{i-1/2} = ( 2 u_{i} + 5 u_{i+1} -1 u_{i+2}) / 6.
    u_minus_0 = (- u_bar[0] + 5 * u_bar[1] + 2 * u_bar[2]) / 6.
    # u^1_{i-1/2} = ( - u_{i-1} + 5 u_{i} + 2 * u_{i+1}) / 6.
    u_minus_1 = (2 * u_bar[1] + 5 * u_bar[2] - u_bar[3]) / 6.
    # u^0_{i-1/2} = ( -2 u_{i-2} - 7 u_{i-1} + 11 * u_{0}) / 6.
    u_minus_2 = (11 * u_bar[2] - 7 * u_bar[3] + 2 * u_bar[4]) / 6.

    eno_polynomials = jnp.array([u_minus_2, u_minus_1, u_minus_0])

  return eno_polynomials


def weno_interpolation_plus(
    u_bar: Array,
    omega_fun: Callable[[Array, Optional[int]], Array],
    order: int = 3,
) -> jnp.float64:
  """Interpolation to u_{i+1/2}.

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    omega_fun: Function that computes the weights from the average of u, u_bar.
    order: Order of the method.

  Returns:
    The value of u interpolated to x_{i+1/2}.
  """
  assert len(u_bar) == order, ("Input size and order do not match. They should",
                               "be equal. Instead they are :",
                               f"input shape {len(u_bar)} and order {order}.")

  # Computing ω.
  omega = omega_fun(u_bar, order)

  # Computing the interpolants (3rd order in  Eq. (7) in [1]).
  u_interp = interpolants_plus(u_bar, order)

  # Computing the interpolant following Eq. (6) in [1] using dot product.
  u_plus = jnp.dot(omega, u_interp)

  return u_plus


def weno_interpolation(
    u_bar: Array,
    omega_fun: Callable[[Array, Optional[int]], Array],
    order: int = 3,
) -> Array:
  """Interpolation to u_{i-1/2} and u_{i+1/2}.

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    omega_fun: Function that computes the weights from the average of u, u_bar.
    order: Order of the method.

  Returns:
    The value of u interpolated to x_{i-1/2} and x_{i+1/2}.
  """
  if len(u_bar) != order:
    raise ValueError(f"Input size ({len(u_bar)}) and polynomial order ",
                     f"({order}) do not match.")

  # Computing ω.
  omega_p = omega_fun(u_bar, order)
  omega_m = omega_fun(u_bar[::-1], order)

  # Computing the interpolants (3rd order in  Eq. (7) in [1]).
  u_inter_p = interpolants_plus(u_bar, order)
  u_inter_m = interpolants_minus(u_bar, order)

  # Computing the interpolant following Eq. (6) in [1] using dot product.
  u_plus = jnp.dot(omega_p, u_inter_p)
  u_minus = jnp.dot(omega_m, u_inter_m)

  return jnp.array([u_minus, u_plus])
