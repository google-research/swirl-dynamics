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

"""Sinkhorn algorithm for optimal transport.

References:

[1] Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal
transport." Advances in neural information processing systems 26 (2013).

[2] Pooladian, Aram-Alexandre, and Niles-Weed, Jonathan. "Entropic estimation of
optimal transport maps." arXiv preprint arXiv:2109.12004 (2021).
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
State = tuple[Array, Array, Array, Array]


class SinkhornOutput(NamedTuple):
  """Holds the output of a Sinkhorn solver applied to a problem.

  Attributes:
    potentials: The potentials for u and v, in log space.
    cost_matrix: The cost matrix of the Sinkhorn iteration, which is the
      distance between each pair of points.
    epsilon: The regularization parameter for the entropic regularization.
    reg_ot_cost: The regularized optimal transport cost.
    threshold: The convergence threshold.
    converged: Whether the iteration converged.
    num_iterations: The number of iterations performed.
  """

  potentials: tuple[Array, Array]
  cost_matrix: Array
  epsilon: float
  reg_ot_cost: Array | None = None
  threshold: float | None = None
  converged: bool | Array | None = None
  num_iterations: int | Array | None = None

  @property
  def fu(self) -> Array:
    """The first dual potential."""
    return self.potentials[0]

  @property
  def gv(self) -> Array:
    """The second dual potential."""
    return self.potentials[1]

  @property
  def cost(self) -> Array:
    """The cost matrix."""
    return self.cost_matrix

  @property
  def transport_plan(self) -> Array:
    """The transport plan."""
    if self.epsilon <= 0:
      raise ValueError("Epsilon should be positive.")
    kernel = -self.cost + self.fu[:, None] + self.gv[None, :]
    kernel /= self.epsilon
    return jnp.exp(kernel)


class SinkhornSolver:
  """Optimal Transport solver with entropic regularisation using log-sum-exp."""

  def __init__(
      self,
      epsilon: float,
      sharding: jax.sharding.Sharding | None = None,
      num_iterations: int = 100,
      threshold: float = 1e-3,
      metric: Callable[[Array], Array] = lambda x: jnp.pow(x, 2),
      eps_marginal: float = 0.0,
  ):
    self.epsilon = epsilon
    self.num_iterations = num_iterations
    self.metric = metric
    self.sharding = sharding
    self.threshold = threshold
    self.eps_marginal = eps_marginal
    self._solver = jax.jit(self._forward_solve)

  def __call__(
      self,
      x: Array,
      y: Array,
      a: Array | None = None,
      b: Array | None = None,
  ) -> SinkhornOutput:
    return self._solver(x, y, a, b)

  def _forward_solve(
      self,
      x: Array,
      y: Array,
      a: Array | None = None,
      b: Array | None = None,
  ) -> SinkhornOutput:
    """Computation of the cost and the coupling matrix."""
    if x.ndim != 2 or y.ndim != 2:
      raise ValueError(
          "x and y should be 2d arrays, instead their dimensions are"
          f" {x.ndim}, {y.ndim} respectively."
      )

    num_x = x.shape[0]
    num_y = y.shape[0]

    if a is None:
      a = jnp.ones((num_x,), dtype=x.dtype) / num_x
    else:
      a = jnp.asarray(a, dtype=x.dtype)
      a = a / jnp.sum(a)
    if b is None:
      b = jnp.ones((num_y,), dtype=y.dtype) / num_y
    else:
      b = jnp.asarray(b, dtype=y.dtype)
      b = b / jnp.sum(b)

    u, v = jnp.zeros_like(a), jnp.zeros_like(b)
    cost_matrix = self._compute_cost(x, y)
    if self.sharding is not None:
      cost_matrix = jax.lax.with_sharding_constraint(cost_matrix, self.sharding)

    def body_fun(val: tuple[int, State]) -> tuple[int, State]:
      i, (_, _, u, v) = val
      u0, v0 = u, v

      kernel_matrix = self._log_gibbs_kernel(u, v, cost_matrix)
      u_ = jnp.log(a + self.eps_marginal) - jax.scipy.special.logsumexp(
          kernel_matrix, axis=1
      )
      u = self.epsilon * u_ + u

      kernel_matrix_t = self._log_gibbs_kernel(u, v, cost_matrix).T
      v_ = jnp.log(b + self.eps_marginal) - jax.scipy.special.logsumexp(
          kernel_matrix_t, axis=1
      )
      v = self.epsilon * v_ + v

      return (i + 1, (u0, v0, u, v))

    def cond_fun(val: tuple[int, State]) -> Array:
      i, (u0, v0, u, v) = val
      return jnp.logical_and(
          (jnp.linalg.norm(u0 - u) + jnp.linalg.norm(v0 - v)) > self.threshold,
          i < self.num_iterations,
      )

    num_its, (_, _, u, v) = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=(0, (jnp.ones_like(u), jnp.ones_like(v), u, v)),
    )

    kernel_matrix = self._log_gibbs_kernel(u, v, cost_matrix)
    pi = jnp.exp(kernel_matrix)
    reg_ot_cost = jnp.sum(pi * cost_matrix, axis=(-2, -1))
    converged = num_its < self.num_iterations

    return SinkhornOutput(
        potentials=(u, v),
        cost_matrix=cost_matrix,
        epsilon=self.epsilon,
        num_iterations=num_its,
        reg_ot_cost=reg_ot_cost,
        converged=converged,
        threshold=self.threshold,
    )

  def _compute_cost(self, x: Array, y: Array) -> Array:
    """Computes the cost matrix of the Sinkhorn iteration."""
    x_ = x[:, None, :]
    y_ = y[None, :, :]
    return jnp.sum(self.metric(x_ - y_), axis=-1)

  def _log_gibbs_kernel(self, u: Array, v: Array, cost_matrix: Array) -> Array:
    """Computes the kernel K = diag(u) exp(-C/eps) diag(v) in log space."""
    kernel = -cost_matrix + u[:, None] + v[None, :]
    return kernel / self.epsilon

  def transport_fn(
      self, potential: Array, y: Array, weights: Array | None = None
  ) -> Callable[[Array], Array]:
    r"""Transport functions using the formulation in Proposition 2 of [2]."""
    f_eps = lambda pt: self._potential_fn(pt, potential, y, weights)
    grad_map = jax.vmap(lambda pt: pt - 0.5 * jax.grad(f_eps)(pt))
    return lambda x: (
        grad_map(jnp.atleast_2d(x))[0] if x.ndim == 1 else grad_map(x)
    )

  def _potential_fn(
      self,
      x: Array,
      potential: Array,
      y: Array,
      weights: Array | None = None,
  ) -> Array:
    """Callback function to compute the potential."""
    x_2d = jnp.atleast_2d(x)
    if x_2d.shape[-1] != y.shape[-1]:
      raise ValueError(
          "x and y should have the same feature dimension, got"
          f" {x_2d.shape[-1]} and {y.shape[-1]}."
      )
    log_w = (potential[None, :] - self._compute_cost(x_2d, y)) / self.epsilon
    if weights is not None:
      log_w = log_w + jnp.log(weights)[None, :]
    lse = -self.epsilon * jax.scipy.special.logsumexp(log_w, axis=-1)
    return lse[0] if x.ndim == 1 else lse

  def transport_fn_direct(
      self, potential: Array, y: Array, weights: Array | None = None
  ) -> Callable[[Array], Array]:
    """Transport directly using the barycentric projection formula in [1]."""
    if potential.ndim != 1 or potential.shape[0] != y.shape[0]:
      raise ValueError(
          f"Expected 1D potential matching `y.shape[0]`, got {potential.shape}"
          f" vs {y.shape}."
      )

    def _transport_direct(x: Array) -> Array:
      x_2d = jnp.atleast_2d(x)
      log_w = (potential[None, :] - self._compute_cost(x_2d, y)) / self.epsilon
      if weights is not None:
        log_w = log_w + jnp.log(weights)[None, :]
      out = jax.nn.softmax(log_w, axis=-1) @ y
      return out[0] if x.ndim == 1 else out

    return _transport_direct
