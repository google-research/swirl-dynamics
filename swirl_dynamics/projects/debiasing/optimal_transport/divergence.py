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
from typing import NamedTuple

import jax
import jax.numpy as jnp
from swirl_dynamics.projects.debiasing.optimal_transport import sinkhorn

Array = jax.Array
_SymState = tuple[Array, Array, Array]


class SinkhornDivergenceOutput(NamedTuple):
  """Holds the output of a Sinkhorn divergence computation.

  Attributes:
    divergence: Scalar Sinkhorn divergence ``S_eps(a, b)``.
    ot_xy: Output of ``SinkhornSolver`` on the cross-problem ``(x, y, a, b)``.
    self_potential_x: Symmetric scaling-gauge dual potential ``u_xx`` of shape
      ``(n,)`` solving the self-transport problem on ``(x, a)``.
    self_potential_y: Symmetric scaling-gauge dual potential ``v_yy`` of shape
      ``(m,)`` solving the self-transport problem on ``(y, b)``.
    converged: Boolean flag indicating whether all three solves (cross and both
      symmetric self-solves) converged within threshold.
  """

  divergence: Array
  ot_xy: sinkhorn.SinkhornOutput
  self_potential_x: Array
  self_potential_y: Array
  converged: Array | bool


@functools.partial(jax.jit, static_argnums=(0,))
def _solve_symmetric_self(
    solver: sinkhorn.SinkhornSolver, x: Array, a: Array
) -> tuple[Array, Array, Array]:
  """Solves the symmetric self-transport problem on ``(x, a)`` [1, Eq. 25].

  Args:
    solver: Configured ``SinkhornSolver`` providing ``epsilon``, ``threshold``,
      ``num_iterations``, ``inner_iterations``, ``criterion``, ``eps_marginal``,
      ``sharding``, and ``_compute_cost``.
    x: Point cloud of shape ``(n, d)``.
    a: Non-negative marginal weights of shape ``(n,)`` summing to 1.

  Returns:
    A tuple ``(u_xx, ot_xx_cost, converged)`` containing the symmetric
    scaling-gauge dual potential ``u_xx`` of shape ``(n,)``, the Fenchel-
    Rockafellar dual self-cost ``OT_eps(a, a)``, and a boolean convergence flag.
  """
  cost_xx = solver._compute_cost(x, x)  # pylint: disable=protected-access
  if solver.sharding is not None:
    cost_xx = jax.lax.with_sharding_constraint(cost_xx, solver.sharding)

  a = jnp.asarray(a, dtype=cost_xx.dtype)
  a_reg = a + solver.eps_marginal
  mask_a = a_reg > 0
  safe_a = jnp.where(mask_a, a_reg, 1.0)
  safe_log_a = jnp.log(safe_a)
  u = jnp.where(mask_a, jnp.zeros_like(a), -jnp.inf)

  def _sweep(u_in: Array) -> Array:
    mask_u = jnp.isfinite(u_in)[None, :]
    safe_u_row = jnp.where(mask_u, u_in[None, :], 0.0)
    logits = jnp.where(
        mask_u, (safe_u_row - cost_xx) / solver.epsilon, -jnp.inf
    )
    target_u = solver.epsilon * (
        safe_log_a - jax.scipy.special.logsumexp(logits, axis=1)
    )
    safe_u_prev = jnp.where(mask_a, u_in, 0.0)
    return jnp.where(mask_a, 0.5 * (safe_u_prev + target_u), -jnp.inf)

  def body_fun(val: _SymState) -> _SymState:
    i, u_curr, _ = val
    steps = jnp.minimum(solver.inner_iterations, solver.num_iterations - i)
    if solver.inner_iterations == 1:
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

    if solver.criterion == "marginal":
      kernel = (-cost_xx + u_next[:, None] + u_next[None, :]) / solver.epsilon
      row_marginals = jnp.exp(jax.scipy.special.logsumexp(kernel, axis=1))
      err = 2.0 * jnp.sum(jnp.abs(row_marginals - a))
    else:
      safe_u_prev = jnp.where(mask_a, u_prev, 0.0)
      err = 2.0 * jnp.linalg.norm(jnp.where(mask_a, u_next, 0.0) - safe_u_prev)
    return (i + steps, u_next, err)

  def cond_fun(val: _SymState) -> Array:
    i, _, err = val
    return jnp.logical_and(err > solver.threshold, i < solver.num_iterations)

  init_err = jnp.asarray(jnp.inf, dtype=cost_xx.dtype)
  init_i = jnp.asarray(0, dtype=jnp.int32)
  _, u, err = jax.lax.while_loop(
      cond_fun=cond_fun,
      body_fun=body_fun,
      init_val=(init_i, u, init_err),
  )
  converged = jnp.logical_and(err <= solver.threshold, jnp.isfinite(err))

  mask_pos = a > 0
  safe_a_orig = jnp.where(mask_pos, a, 1.0)
  safe_u_fin = jnp.where(mask_pos, u, 0.0)
  kernel_xx = (-cost_xx + u[:, None] + u[None, :]) / solver.epsilon
  ot_xx_cost = 2.0 * jnp.sum(
      jnp.where(
          mask_pos,
          safe_a_orig * (safe_u_fin - solver.epsilon * jnp.log(safe_a_orig)),
          0.0,
      )
  ) + solver.epsilon * (jnp.sum(a) ** 2 - jnp.sum(jnp.exp(kernel_xx)))
  return u, ot_xx_cost, converged


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

  using ``entropic_ot_cost`` on the cross-problem and symmetric fixed-point
  updates [1, Eq. 25] on the self-problems.

  Args:
    x: Source point cloud of shape ``(n, d)``.
    y: Target point cloud of shape ``(m, d)``.
    a: Optional non-negative source weights of shape ``(n,)`` summing to 1.
    b: Optional non-negative target weights of shape ``(m,)`` summing to 1.
    solver: Optional pre-configured ``SinkhornSolver``. When provided, its
      ``epsilon``, ``num_iterations``, ``threshold``, ``metric``, ``criterion``,
      ``inner_iterations``, ``eps_marginal``, and ``sharding`` are used.
    epsilon: Entropic regularization parameter used when ``solver is None``.
    num_iterations: Maximum Sinkhorn iterations used when ``solver is None``.
    threshold: Convergence threshold used when ``solver is None``.

  Returns:
    A ``SinkhornDivergenceOutput`` holding ``divergence``, ``ot_xy``,
    ``self_potential_x``, ``self_potential_y``, and ``converged``.
  """
  if solver is not None:
    active_solver = solver
  else:
    active_solver = sinkhorn.SinkhornSolver(
        epsilon=epsilon,
        num_iterations=num_iterations,
        threshold=threshold,
    )
  ot_xy = active_solver(x, y, a=a, b=b)
  assert ot_xy.weights is not None
  a_vec, b_vec = ot_xy.weights

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
  """Computes the scalar Sinkhorn divergence ``S_eps(a, b)`` [1].

  Args:
    x: Source point cloud of shape ``(n, d)``.
    y: Target point cloud of shape ``(m, d)``.
    a: Optional non-negative source weights of shape ``(n,)`` summing to 1.
    b: Optional non-negative target weights of shape ``(m,)`` summing to 1.
    solver: Optional pre-configured ``SinkhornSolver``.
    epsilon: Entropic regularization parameter used when ``solver is None``.
    num_iterations: Maximum Sinkhorn iterations used when ``solver is None``.
    threshold: Convergence threshold used when ``solver is None``.

  Returns:
    Scalar ``Array`` giving the Sinkhorn divergence ``S_eps(a, b)``.
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


def debiased_transport_fn(
    solver: sinkhorn.SinkhornSolver,
    x: Array,
    y: Array,
    a: Array | None = None,
    b: Array | None = None,
    *,
    batch_size: int | None = None,
) -> Callable[[Array], Array]:
  r"""Returns the centered (debiased) entropic optimal transport map [2].

  Constructs the centered map derived from the first variation of the Sinkhorn
  divergence:

  .. math::

    T^{\text{deb}}_\varepsilon(q) = q + T_{\varepsilon, x \to y}(q)
      - T_{\varepsilon, x \to x}(q)

  Caveat (Pooladian, Cuturi, and Niles-Weed, 2022 [2]):
    While centering removes entropic shrinkage bias and guarantees that
    ``T^{\text{deb}}_\varepsilon(q) = q`` identically when ``(x, a) == (y, b)``,
    subtracting two convex entropic potentials gives a difference of convex
    functions. Consequently, ``T^{\text{deb}}_\varepsilon`` is NOT guaranteed to
    be the gradient of a convex function (its symmetrized Jacobian can become
    indefinite / non-monotone) at small ``epsilon`` or when source and target
    supports are poorly aligned. Callers requiring monotonicity (cyclical
    monotonicity / positive-semidefinite Jacobian) should inspect
    ``jacobian_min_eigenvalue`` on their evaluation points.

  Args:
    solver: Configured ``SinkhornSolver`` instance.
    x: Source point cloud of shape ``(n, d)``.
    y: Target point cloud of shape ``(m, d)``.
    a: Optional non-negative source weights of shape ``(n,)`` summing to 1.
    b: Optional non-negative target weights of shape ``(m,)`` summing to 1.
    batch_size: Optional positive integer chunk size forwarded to
      ``solver.transport_fn`` to bound peak memory during query evaluation.

  Returns:
    A callable mapping query points ``q`` of shape ``(d,)`` or ``(B, d)`` to
    transported points of the same shape.
  """
  res = sinkhorn_divergence_with_potentials(x, y, a=a, b=b, solver=solver)
  t_xy = solver.transport_fn(res.ot_xy.gv, y, batch_size=batch_size)
  t_xx = solver.transport_fn(res.self_potential_x, x, batch_size=batch_size)

  def _map(query: Array) -> Array:
    return query + t_xy(query) - t_xx(query)

  return _map


def jacobian_min_eigenvalue(
    map_fn: Callable[[Array], Array], points: Array
) -> Array:
  """Computes the minimum eigenvalue of the symmetrized Jacobian of ``map_fn``.

  Provides a quantitative monotonicity diagnostic [2]: a map ``T`` is locally
  monotone at ``pt`` iff ``0.5 * (DT(pt) + DT(pt)^T)`` is positive semidefinite
  (minimum eigenvalue ``>= 0``). Negative values indicate local non-monotonicity
  (folding / orientation reversal).

  Args:
    map_fn: Differentiable vector field mapping ``(d,) -> (d,)``.
    points: Evaluation point of shape ``(d,)`` or batch of points of shape ``(B,
      d)``.

  Returns:
    Scalar ``Array`` when ``points.ndim == 1``, or 1-D ``Array`` of shape
    ``(B,)`` when ``points.ndim == 2``.

  Raises:
    ValueError: If ``points.ndim`` is not 1 or 2.
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
