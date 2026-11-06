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

"""Sinkhorn divergence and debiased entropic optimal transport maps.

References:

[1] Feydy, Jean, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari,
Alain Trouvé, and Gabriel Peyré. "Interpolating between Optimal Transport and
MMD using Sinkhorn Divergences." AISTATS (PMLR 89), 2019 (arXiv:1810.08278).

[2] Pooladian, Aram-Alexandre, Marco Cuturi, and Jonathan Niles-Weed. "Debiaser
Beware: Pitfalls of Centering Regularized Transport Maps." ICML (PMLR 162), 2022
(arXiv:2202.08919).
"""

from collections.abc import Callable
import functools
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from swirl_dynamics.lib.ot import sinkhorn

Array = jax.Array
_SymState = tuple[Array, Array, Array]

__all__ = [
    "SinkhornDivergenceOutput",
    "debiased_transport_fn",
    "jacobian_min_eigenvalue",
    "sinkhorn_divergence",
    "sinkhorn_divergence_with_potentials",
]


class SinkhornDivergenceOutput(NamedTuple):
  """Holds the output of a Sinkhorn divergence computation.

  Attributes:
    divergence: Scalar Sinkhorn divergence `S_eps(a, b)`.
    ot_xy: Output of `SinkhornSolver` on the cross-problem `(x, y, a, b)`.
    self_potential_x: Symmetric scaling-gauge dual potential `u_xx` of shape
      `(n,)` solving the self-transport problem on `(x, a)`.
    self_potential_y: Symmetric scaling-gauge dual potential `v_yy` of shape
      `(m,)` solving the self-transport problem on `(y, b)`.
    converged: Boolean flag indicating whether all three solves (cross and both
      symmetric self-solves) converged within threshold.
  """

  divergence: Array
  ot_xy: sinkhorn.SinkhornOutput
  self_potential_x: Array
  self_potential_y: Array
  converged: Array | bool


@functools.lru_cache(maxsize=32)
def _default_solver(
    epsilon: float,
    num_iterations: int,
    threshold: float,
) -> sinkhorn.SinkhornSolver:
  """Returns a cached default `SinkhornSolver` when `solver is None`."""
  return sinkhorn.SinkhornSolver(
      epsilon=epsilon,
      num_iterations=num_iterations,
      threshold=threshold,
  )


@functools.partial(
    jax.jit,
    static_argnames=("inner_iterations", "criterion", "metric", "sharding"),
)
def _solve_symmetric_self_jit(
    x: Array,
    a: Array,
    *,
    epsilon: float | Array,
    num_iterations: int | Array,
    threshold: float | Array,
    eps_marginal: float | Array,
    inner_iterations: int,
    criterion: Literal["marginal", "potentials"],
    metric: Callable[[Array], Array],
    sharding: jax.sharding.Sharding | None,
) -> tuple[Array, Array, Array]:
  """JIT kernel for the symmetric self-transport problem on `(x, a)`."""
  cost_xx = jnp.sum(metric(x[:, None, :] - x[None, :, :]), axis=-1)
  if sharding is not None:
    cost_xx = jax.lax.with_sharding_constraint(cost_xx, sharding)

  num_iters = jnp.asarray(num_iterations, dtype=jnp.int32)
  inner_iters = jnp.asarray(inner_iterations, dtype=jnp.int32)
  a = jnp.asarray(a, dtype=cost_xx.dtype)
  a_reg = a + eps_marginal
  mask_a = a_reg > 0
  safe_a = jnp.where(mask_a, a_reg, 1.0)
  safe_log_a = jnp.log(safe_a)
  u = jnp.where(mask_a, jnp.zeros_like(a), -jnp.inf)

  def _sweep(u_in: Array) -> Array:
    mask_u = jnp.isfinite(u_in)[None, :]
    safe_u_row = jnp.where(mask_u, u_in[None, :], 0.0)
    logits = jnp.where(mask_u, (safe_u_row - cost_xx) / epsilon, -jnp.inf)
    target_u = epsilon * (
        safe_log_a - jax.scipy.special.logsumexp(logits, axis=1)
    )
    safe_u_prev = jnp.where(mask_a, u_in, 0.0)
    return jnp.where(mask_a, 0.5 * (safe_u_prev + target_u), -jnp.inf)

  def body_fun(val: _SymState) -> _SymState:
    i, u_curr, _ = val
    steps = jnp.minimum(inner_iters, num_iters - i)
    if inner_iterations == 1:
      u_prev = u_curr
      u_next = _sweep(u_curr)
    else:

      def _inner_body(
          idx: int, carry: tuple[Array, Array]
      ) -> tuple[Array, Array]:
        del idx
        _, u_c = carry
        return (u_c, _sweep(u_c))

      u_prev, u_next = jax.lax.fori_loop(
          0, steps, _inner_body, (u_curr, u_curr)
      )

    if criterion == "marginal":
      kernel = (-cost_xx + u_next[:, None] + u_next[None, :]) / epsilon
      row_marginals = jnp.exp(jax.scipy.special.logsumexp(kernel, axis=1))
      err = 2.0 * jnp.sum(jnp.abs(row_marginals - a))
    else:
      safe_u_prev = jnp.where(mask_a, u_prev, 0.0)
      err = 2.0 * jnp.linalg.norm(jnp.where(mask_a, u_next, 0.0) - safe_u_prev)
    return (i + steps, u_next, err)

  def cond_fun(val: _SymState) -> Array:
    i, _, err = val
    return jnp.logical_and(err > threshold, i < num_iters)

  init_err = jnp.asarray(jnp.inf, dtype=cost_xx.dtype)
  init_i = jnp.asarray(0, dtype=jnp.int32)
  _, u, err = jax.lax.while_loop(
      cond_fun=cond_fun,
      body_fun=body_fun,
      init_val=(init_i, u, init_err),
  )
  converged = jnp.logical_and(err <= threshold, jnp.isfinite(err))

  mask_pos = a > 0
  safe_a_orig = jnp.where(mask_pos, a, 1.0)
  safe_u_fin = jnp.where(mask_pos, u, 0.0)
  kernel_xx = (-cost_xx + u[:, None] + u[None, :]) / epsilon
  ot_xx_cost = 2.0 * jnp.sum(
      jnp.where(
          mask_pos,
          safe_a_orig * (safe_u_fin - epsilon * jnp.log(safe_a_orig)),
          0.0,
      )
  ) + epsilon * (jnp.sum(a) ** 2 - jnp.sum(jnp.exp(kernel_xx)))
  return u, ot_xx_cost, converged


def _solve_symmetric_self(
    solver: sinkhorn.SinkhornSolver, x: Array, a: Array
) -> tuple[Array, Array, Array]:
  """Solves the symmetric self-transport problem on `(x, a)` [1, Eq. 25].

  Args:
    solver: Configured `SinkhornSolver` providing `epsilon`, `threshold`,
      `num_iterations`, `inner_iterations`, `criterion`, `eps_marginal`,
      `metric`, and `sharding`.
    x: Point cloud of shape `(n, d)`.
    a: Non-negative marginal weights of shape `(n,)` summing to 1.

  Returns:
    A tuple `(u_xx, ot_xx_cost, converged)` containing the symmetric
    scaling-gauge dual potential `u_xx` of shape `(n,)`, the Fenchel-
    Rockafellar dual self-cost `OT_eps(a, a)`, and a boolean convergence flag.
  """
  return _solve_symmetric_self_jit(
      x,
      a,
      epsilon=solver.epsilon,
      num_iterations=solver.num_iterations,
      threshold=solver.threshold,
      eps_marginal=solver.eps_marginal,
      inner_iterations=solver.inner_iterations,
      criterion=solver.criterion,
      metric=solver.metric,
      sharding=solver.sharding,
  )


def sinkhorn_divergence_with_potentials(
    x: Array,
    y: Array,
    a: Array | None = None,
    b: Array | None = None,
    *,
    solver: sinkhorn.SinkhornSolver | None = None,
    epsilon: float = 0.1,
    num_iterations: int = 1000,
    threshold: float = 1e-6,
) -> SinkhornDivergenceOutput:
  r"""Computes the Sinkhorn divergence and potentials between two point clouds.

  Evaluates the debiased Sinkhorn divergence [1]:

  .. math::

    S_\varepsilon(a, b) = \text{OT}_\varepsilon(a, b)
      - \frac{1}{2} \text{OT}_\varepsilon(a, a)
      - \frac{1}{2} \text{OT}_\varepsilon(b, b)

  using `entropic_ot_cost` on the cross-problem and symmetric fixed-point
  updates [1, Eq. 25] on the self-problems.

  Args:
    x: Source point cloud of shape `(n, d)`.
    y: Target point cloud of shape `(m, d)`.
    a: Optional non-negative source weights of shape `(n,)` summing to 1. If
      `None`, uniform empirical weights `1 / n` are used.
    b: Optional non-negative target weights of shape `(m,)` summing to 1. If
      `None`, uniform empirical weights `1 / m` are used.
    solver: Optional pre-configured `SinkhornSolver`. When provided, its
      `epsilon`, `num_iterations`, `threshold`, `metric`, `criterion`,
      `inner_iterations`, `eps_marginal`, and `sharding` are used; if `None`, a
      default `SinkhornSolver` configured with `epsilon`, `num_iterations`, and
      `threshold` is used. When `sharding` is set and `n != m`, prefer 1-D row
      sharding `PartitionSpec("x", None)` because a 2-D `PartitionSpec("x",
      "y")` tuned for rectangular `(n, m)` can silently replicate on the square
      `(n, n)` and `(m, m)` self-cost matrices without a runtime warning.
    epsilon: Entropic regularization parameter used when `solver is None`.
    num_iterations: Maximum Sinkhorn iterations used when `solver is None`.
    threshold: Convergence threshold used when `solver is None`.

  Returns:
    A `SinkhornDivergenceOutput` holding `divergence`, `ot_xy`,
    `self_potential_x`, `self_potential_y`, and `converged`.

  Raises:
    ValueError: If point cloud shapes, solver hyperparameters, or concrete
      marginal probability weights are invalid.
  """
  if solver is not None:
    active_solver = solver
  else:
    active_solver = _default_solver(
        float(epsilon), int(num_iterations), float(threshold)
    )
  ot_xy = active_solver(x, y, a=a, b=b)
  a_vec, b_vec = ot_xy.resolved_weights()

  u_xx, ot_xx, conv_x = _solve_symmetric_self(active_solver, x, a_vec)
  v_yy, ot_yy, conv_y = _solve_symmetric_self(active_solver, y, b_vec)

  div = ot_xy.entropic_ot_cost - 0.5 * ot_xx - 0.5 * ot_yy
  converged = jnp.logical_and(
      jnp.asarray(ot_xy.converged), jnp.logical_and(conv_x, conv_y)
  )
  return SinkhornDivergenceOutput(
      divergence=div,
      ot_xy=ot_xy,
      self_potential_x=u_xx,
      self_potential_y=v_yy,
      converged=converged,
  )


def sinkhorn_divergence(
    x: Array,
    y: Array,
    a: Array | None = None,
    b: Array | None = None,
    *,
    solver: sinkhorn.SinkhornSolver | None = None,
    epsilon: float = 0.1,
    num_iterations: int = 1000,
    threshold: float = 1e-6,
) -> Array:
  """Computes the scalar Sinkhorn divergence `S_eps(a, b)` [1].

  Args:
    x: Source point cloud of shape `(n, d)`.
    y: Target point cloud of shape `(m, d)`.
    a: Optional non-negative source weights of shape `(n,)` summing to 1. If
      `None`, uniform empirical weights `1 / n` are used.
    b: Optional non-negative target weights of shape `(m,)` summing to 1. If
      `None`, uniform empirical weights `1 / m` are used.
    solver: Optional pre-configured `SinkhornSolver` (prefer 1-D row sharding
      `PartitionSpec("x", None)` when `sharding` is set and `n != m`). If
      `None`, a default `SinkhornSolver` configured with `epsilon`,
      `num_iterations`, and `threshold` is used.
    epsilon: Entropic regularization parameter used when `solver is None`.
    num_iterations: Maximum Sinkhorn iterations used when `solver is None`.
    threshold: Convergence threshold used when `solver is None`.

  Returns:
    Scalar `Array` giving the Sinkhorn divergence `S_eps(a, b)`.

  Raises:
    ValueError: If point cloud shapes, solver hyperparameters, or concrete
      marginal probability weights are invalid.
  """
  return sinkhorn_divergence_with_potentials(
      x,
      y,
      a=a,
      b=b,
      solver=solver,
      epsilon=epsilon,
      num_iterations=num_iterations,
      threshold=threshold,
  ).divergence


def _validate_precomputed_ot_xy(
    x: Array,
    y: Array,
    a: Array | None,
    b: Array | None,
    ot_xy: sinkhorn.SinkhornOutput,
) -> Array:
  """Validates precomputed `ot_xy` shapes/weights and returns `a_vec`."""
  a_arr = jnp.asarray(a) if a is not None else None
  b_arr = jnp.asarray(b) if b is not None else None
  sinkhorn.validate_solve_shapes(x, y, a_arr, b_arr)
  if ot_xy.fu.shape != (x.shape[0],) or ot_xy.gv.shape != (y.shape[0],):
    raise ValueError(
        "ot_xy potentials must match (x.shape[0],) and (y.shape[0],), got"
        f" ot_xy.fu.shape={ot_xy.fu.shape}, ot_xy.gv.shape={ot_xy.gv.shape},"
        f" x.shape={x.shape}, and y.shape={y.shape}."
    )
  if a_arr is not None:
    sinkhorn.validate_weights(a_arr, "a")
    if (
        ot_xy.weights is not None
        and not isinstance(a_arr, jax.core.Tracer)
        and not isinstance(ot_xy.weights[0], jax.core.Tracer)
        and not bool(jnp.allclose(a_arr, ot_xy.weights[0], atol=1e-6))
    ):
      raise ValueError("a does not match precomputed ot_xy.weights[0].")
  if b_arr is not None:
    sinkhorn.validate_weights(b_arr, "b")
    if (
        ot_xy.weights is not None
        and not isinstance(b_arr, jax.core.Tracer)
        and not isinstance(ot_xy.weights[1], jax.core.Tracer)
        and not bool(jnp.allclose(b_arr, ot_xy.weights[1], atol=1e-6))
    ):
      raise ValueError("b does not match precomputed ot_xy.weights[1].")
  if ot_xy.weights is not None:
    return ot_xy.weights[0]
  if a_arr is not None:
    return jnp.asarray(a_arr, dtype=x.dtype)
  return ot_xy.resolved_weights()[0]


def debiased_transport_fn(
    solver: sinkhorn.SinkhornSolver,
    x: Array,
    y: Array,
    a: Array | None = None,
    b: Array | None = None,
    *,
    batch_size: int | None = None,
    ot_xy: sinkhorn.SinkhornOutput | None = None,
) -> Callable[[Array], Array]:
  r"""Returns the centered (debiased) entropic optimal transport map [2].

  Constructs the centered map derived from the first variation of the Sinkhorn
  divergence:

  .. math::

    T^{\text{deb}}_\varepsilon(q) = q + T_{\varepsilon, x \to y}(q)
      - T_{\varepsilon, x \to x}(q)

  Caveat (Pooladian, Cuturi, and Niles-Weed, 2022 [2]):
    While centering removes entropic shrinkage bias and guarantees that
    `T^{\text{deb}}_\varepsilon(q) = q` identically when `(x, a) == (y, b)`,
    subtracting two convex entropic potentials gives a difference of convex
    functions. Consequently, `T^{\text{deb}}_\varepsilon` is NOT guaranteed to
    be the gradient of a convex function (its symmetrized Jacobian can become
    indefinite / non-monotone) at small `epsilon` or when source and target
    supports are poorly aligned. Callers requiring monotonicity (cyclical
    monotonicity / positive-semidefinite Jacobian) should inspect
    `jacobian_min_eigenvalue` on their evaluation points.

  Args:
    solver: Configured `SinkhornSolver` instance.
    x: Source point cloud of shape `(n, d)`.
    y: Target point cloud of shape `(m, d)`.
    a: Optional non-negative source weights of shape `(n,)` summing to 1. If
      `None`, uniform empirical weights `1 / n` are used.
    b: Optional non-negative target weights of shape `(m,)` summing to 1. If
      `None`, uniform empirical weights `1 / m` are used.
    batch_size: Optional positive integer chunk size forwarded to
      `solver.transport_fn` to bound peak memory during query evaluation. If
      `None`, all query points are evaluated in parallel via `jax.vmap`.
    ot_xy: Optional precomputed `SinkhornOutput` from `solver(x, y, a=a, b=b)`
      to avoid re-solving the `O(nm)` cross-transport problem. If `None`, the
      cross-transport problem is solved internally via `solver(x, y, a=a, b=b)`.

  Returns:
    A callable mapping query points `q` of shape `(d,)` or `(B, d)` to
    transported points of the same shape.

  Raises:
    ValueError: If point cloud shapes, marginal weights, `batch_size`, or
      precomputed `ot_xy` potentials/weights are invalid.
  """
  if ot_xy is None:
    ot_xy = solver(x, y, a=a, b=b)
    a_vec, _ = ot_xy.resolved_weights()
  else:
    a_vec = _validate_precomputed_ot_xy(x, y, a, b, ot_xy)
  self_potential_x, _, _ = _solve_symmetric_self(solver, x, a_vec)
  t_xy = solver.transport_fn(ot_xy.gv, y, batch_size=batch_size)
  t_xx = solver.transport_fn(self_potential_x, x, batch_size=batch_size)

  def _map(query: Array) -> Array:
    return query + t_xy(query) - t_xx(query)

  return _map


def jacobian_min_eigenvalue(
    map_fn: Callable[[Array], Array], points: Array
) -> Array:
  """Computes the minimum eigenvalue of the symmetrized Jacobian of `map_fn`.

  Provides a quantitative monotonicity diagnostic [2]: a map `T` is locally
  monotone at `pt` iff `0.5 * (DT(pt) + DT(pt)^T)` is positive semidefinite
  (minimum eigenvalue `>= 0`). Negative values indicate local non-monotonicity
  (folding / orientation reversal).

  Args:
    map_fn: Differentiable vector field mapping `(d,) -> (d,)`.
    points: Evaluation point of shape `(d,)` or batch of points of shape `(B,
      d)`.

  Returns:
    Scalar `Array` when `points.ndim == 1`, or 1-D `Array` of shape `(B,)` when
    `points.ndim == 2`.

  Raises:
    ValueError: If `points.ndim` is not 1 or 2.
  """
  if points.ndim not in (1, 2):
    raise ValueError(
        "points must be a 1D or 2D array, got"
        f" points.ndim={points.ndim} with points.shape={points.shape}."
    )

  def _min_eig_one(pt: Array) -> Array:
    jac = jax.jacrev(map_fn)(pt)
    jac_sym = 0.5 * (jac + jac.T)
    return jnp.linalg.eigvalsh(jac_sym)[0]

  if points.ndim == 1:
    return _min_eig_one(points)
  return jax.vmap(_min_eig_one)(points)
