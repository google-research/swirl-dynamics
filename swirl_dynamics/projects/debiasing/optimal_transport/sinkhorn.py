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
from typing import Literal, NamedTuple

from absl import logging
import jax
import jax.numpy as jnp


Array = jax.Array
State = tuple[Array, Array, Array, Array]


# TODO: Enable dataclasses for SinkhornOutput.
# @jax.tree_util.register_dataclass
# @dataclasses.dataclass(kw_only=True)
class SinkhornOutput(NamedTuple):
  """Holds the output of a Sinkhorn solver applied to a problem.

  Attributes:
    potentials: The potentials for u and v, in log space.
    cost_matrix: The cost matrix of the Sinkhorn iteration, which is the
      distance between each pair of points.
    epsilon: The regularization parameter for the entropic regularization.
    reg_ot_cost: Bare primal transport cost ``<P, C>`` (matching OTT-JAX
      ``primal_cost``, without the entropic KL term; see ``entropic_ot_cost``
      for the KL-regularized objective).
    threshold: The convergence threshold.
    converged: Whether the iteration converged.
    num_iterations: The number of iterations performed.
    marginal_error: Total L1 marginal violation ``||P 1 - a||_1 + ||P^T 1 -
      b||_1`` of the transport plan.
    weights: Marginal probability weights ``(a, b)`` for sets A and B.
  """

  potentials: tuple[Array, Array]
  cost_matrix: Array
  epsilon: float
  reg_ot_cost: Array | None = None
  threshold: float | None = None
  converged: Array | bool | None = None
  num_iterations: Array | int | None = None
  marginal_error: Array | None = None
  weights: tuple[Array, Array] | None = None
  # TODO: Compute deviation from optimality in the solutions.
  # errors: jnp.ndarray | None = None

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
    if not isinstance(self.epsilon, jax.core.Tracer) and self.epsilon <= 0:
      raise ValueError("Epsilon should be positive.")
    kernel = -self.cost + self.fu[:, None] + self.gv[None, :]
    kernel /= self.epsilon

    return jnp.exp(kernel)

  @property
  def kantorovich_potentials(self) -> tuple[Array, Array]:
    r"""Returns finite Kantorovich-gauge potentials ``(f, g)``.

    The Kantorovich potentials satisfy
    ``P_{ij} = exp((f_i + g_j - C_{ij}) / epsilon) * a_i * b_j``.
    On positive-support entries (``a_i > 0`` / ``b_j > 0``),
    ``f_i = u_i - epsilon * log(a_i)`` and ``g_j = v_j - epsilon * log(b_j)``.
    On zero-weight entries where ``u_i = -inf`` or ``v_j = -inf``, the finite
    entropic c-transform is evaluated using double-``jnp.where`` masking so
    that ``f`` and ``g`` are finite at every index.
    """
    if not isinstance(self.epsilon, jax.core.Tracer) and self.epsilon <= 0:
      raise ValueError("Epsilon should be positive.")
    if self.weights is not None:
      a, b = self.weights
    else:
      num_x = self.fu.shape[0]
      num_y = self.gv.shape[0]
      a = jnp.ones((num_x,), dtype=self.fu.dtype) / num_x
      b = jnp.ones((num_y,), dtype=self.gv.dtype) / num_y

    mask_a = jnp.logical_and(a > 0, jnp.isfinite(self.fu))
    safe_a = jnp.where(mask_a, a, 1.0)
    safe_u = jnp.where(mask_a, self.fu, 0.0)
    f_pos = safe_u - self.epsilon * jnp.log(safe_a)

    mask_v = jnp.isfinite(self.gv)[None, :]
    safe_v_2d = jnp.where(mask_v, self.gv[None, :], 0.0)
    logits_f = jnp.where(
        mask_v, (safe_v_2d - self.cost_matrix) / self.epsilon, -jnp.inf
    )
    f_ctrans = -self.epsilon * jax.scipy.special.logsumexp(logits_f, axis=1)
    f = jnp.where(mask_a, f_pos, f_ctrans)

    mask_b = jnp.logical_and(b > 0, jnp.isfinite(self.gv))
    safe_b = jnp.where(mask_b, b, 1.0)
    safe_v = jnp.where(mask_b, self.gv, 0.0)
    g_pos = safe_v - self.epsilon * jnp.log(safe_b)

    mask_u = jnp.isfinite(self.fu)[:, None]
    safe_u_2d = jnp.where(mask_u, self.fu[:, None], 0.0)
    logits_g = jnp.where(
        mask_u, (safe_u_2d - self.cost_matrix) / self.epsilon, -jnp.inf
    )
    g_ctrans = -self.epsilon * jax.scipy.special.logsumexp(logits_g, axis=0)
    g = jnp.where(mask_b, g_pos, g_ctrans)
    return f, g

  @property
  def entropic_ot_cost(self) -> Array:
    """The entropic optimal transport cost (Fenchel-Rockafellar dual objective).

    At marginal convergence, this equals the KL-regularized primal objective
    ``<P, C> + epsilon * KL(P || a x b) = <f, a> + <g, b>``, matching OTT-JAX's
    ``reg_ot_cost`` (``kl_reg_cost``), and differs from OTT-JAX's
    ``ent_reg_cost`` by ``+epsilon * (H(a) + H(b))``.
    """
    if self.weights is not None:
      a, b = self.weights
    else:
      num_x = self.fu.shape[0]
      num_y = self.gv.shape[0]
      a = jnp.ones((num_x,), dtype=self.fu.dtype) / num_x
      b = jnp.ones((num_y,), dtype=self.gv.dtype) / num_y
    mask_a = a > 0
    mask_b = b > 0
    safe_a = jnp.where(mask_a, a, 1.0)
    safe_b = jnp.where(mask_b, b, 1.0)
    safe_u = jnp.where(mask_a, self.fu, 0.0)
    safe_v = jnp.where(mask_b, self.gv, 0.0)
    term_a = jnp.sum(
        jnp.where(
            mask_a, safe_a * (safe_u - self.epsilon * jnp.log(safe_a)), 0.0
        )
    )
    term_b = jnp.sum(
        jnp.where(
            mask_b, safe_b * (safe_v - self.epsilon * jnp.log(safe_b)), 0.0
        )
    )
    legendre_gap = self.epsilon * (
        jnp.sum(a) * jnp.sum(b) - jnp.sum(self.transport_plan)
    )
    return term_a + term_b + legendre_gap


def _validate_potential_and_y(potential: Array, y: Array) -> None:
  """Validates shapes of dual potential vector and target point cloud."""
  if y.ndim != 2:
    raise ValueError(
        f"y must be a 2D array of shape (m, d), got y.ndim={y.ndim} with"
        f" y.shape={y.shape}."
    )
  if potential.ndim != 1 or potential.shape[0] != y.shape[0]:
    raise ValueError(
        "potential must be a 1D array with potential.shape[0] == y.shape[0],"
        f" got potential.ndim={potential.ndim},"
        f" potential.shape={potential.shape}, and y.shape={y.shape}."
    )


def _validate_query_points(x: Array, y: Array) -> None:
  """Validates query array rank (1-D or 2-D) and feature dimension."""
  if x.ndim not in (1, 2):
    raise ValueError(
        f"x must be a 1D or 2D array, got x.ndim={x.ndim} with"
        f" x.shape={x.shape}."
    )
  if x.shape[-1] != y.shape[-1]:
    raise ValueError(
        "x and y must have the same feature dimension, got"
        f" x.shape[-1]={x.shape[-1]} and y.shape[-1]={y.shape[-1]}."
    )


def _validate_solve_shapes(
    x: Array, y: Array, a: Array | None, b: Array | None
) -> None:
  """Validates static shapes of point clouds and optional marginal weights."""
  if x.ndim != 2 or y.ndim != 2:
    raise ValueError(
        "x and y should be 2d arrays, instead their dimensions are"
        f" {x.ndim}, {y.ndim} respectively."
    )
  if x.shape[0] <= 0 or y.shape[0] <= 0:
    raise ValueError(
        "x and y must be non-empty along axis 0, got"
        f" x.shape={x.shape} and y.shape={y.shape}."
    )
  if x.shape[1] != y.shape[1]:
    raise ValueError(
        "x and y must have the same feature dimension, got"
        f" x.shape[1]={x.shape[1]} and y.shape[1]={y.shape[1]}."
    )
  if a is not None and (a.ndim != 1 or a.shape[0] != x.shape[0]):
    raise ValueError(
        f"a must be a 1D array of length x.shape[0]={x.shape[0]}, got"
        f" a.ndim={a.ndim} and a.shape={a.shape}."
    )
  if b is not None and (b.ndim != 1 or b.shape[0] != y.shape[0]):
    raise ValueError(
        f"b must be a 1D array of length y.shape[0]={y.shape[0]}, got"
        f" b.ndim={b.ndim} and b.shape={b.shape}."
    )


def _validate_weights(weights: Array, name: str) -> None:
  """Validates concrete marginal probability weights outside of JIT."""
  if isinstance(weights, jax.core.Tracer):
    return
  if not bool(jnp.all(jnp.isfinite(weights))):
    raise ValueError(f"{name} must contain only finite values.")
  if not bool(jnp.all(weights >= 0)):
    raise ValueError(f"{name} must contain non-negative weights.")
  total = float(jnp.sum(weights))
  if abs(total - 1.0) > 1e-6:
    raise ValueError(
        f"{name} must sum to 1 within 1e-6, got sum({name})={total}."
    )


class SinkhornSolver:
  """Optimal Transport solver with entropic regularisation using log-sum-exp.

  This class implements the Sinkhorn algorithm [1] for optimal transport with
  entropic regularization. The algorithm is based on the log-sum-exp approach,
  which provides a fast and numerically stable implementation.

  Attributes:
    epsilon: The regularization parameter for the Sinkhorn iteration.
    num_iterations: Maximum number of iteration for the iteration to converge.
    metric: Function for the distance between two samples.
    sharding: Positional sharding for the cost matrix. By default we assume that
      there is only one accelerator.
    threshold: Convergence threshold to stop the iteration.
    eps_marginal: The regularization parameter for the marginal densities.
    criterion: Stopping criterion ("marginal" or "potentials").
    inner_iterations: Number of Sinkhorn sweeps per convergence check.
    _solver: The jitted solver function.
  """

  def __init__(
      self,
      epsilon: float,
      sharding: jax.sharding.Sharding | None = None,
      num_iterations: int = 100,
      threshold: float = 1e-3,
      metric: Callable[[Array], Array] = lambda x: jnp.pow(x, 2),
      eps_marginal: float = 0.0,
      criterion: Literal["marginal", "potentials"] = "marginal",
      inner_iterations: int = 1,
  ):
    """Initializes the Sinkhorn solver.

    Args:
      epsilon: Entropic regularization parameter (must be positive).
      sharding: Optional sharding constraint for the cost matrix.
      num_iterations: Maximum number of Sinkhorn iterations (must be
        non-negative).
      threshold: Convergence threshold on marginal or potential error (must be
        non-negative).
      metric: Elementwise distance function applied before summing across
        features.
      eps_marginal: Non-negative additive stabilizer for marginal densities.
      criterion: Convergence criterion, either ``"marginal"`` (total L1 marginal
        violation) or ``"potentials"`` (L2 change in dual potentials).
      inner_iterations: Number of Sinkhorn sweeps executed between convergence
        checks (must be at least 1).

    Raises:
      ValueError: If `epsilon <= 0`, `num_iterations < 0`, `threshold < 0`,
        `eps_marginal < 0`, `criterion` is invalid, or `inner_iterations < 1`.
    """
    if epsilon <= 0:
      raise ValueError(f"epsilon must be positive, got {epsilon}.")
    if num_iterations < 0:
      raise ValueError(
          f"num_iterations must be non-negative, got {num_iterations}."
      )
    if threshold < 0:
      raise ValueError(f"threshold must be non-negative, got {threshold}.")
    if eps_marginal < 0:
      raise ValueError(
          f"eps_marginal must be non-negative, got {eps_marginal}."
      )
    if criterion not in ("marginal", "potentials"):
      raise ValueError(
          f"criterion must be 'marginal' or 'potentials', got {criterion!r}."
      )
    if inner_iterations < 1:
      raise ValueError(
          f"inner_iterations must be at least 1, got {inner_iterations}."
      )
    if not jax.config.jax_enable_x64:
      logging.warning(
          "jax_enable_x64 is False; Sinkhorn potentials may lose accuracy or"
          " underflow in float32 when epsilon is small relative to median(C)."
      )
    self.epsilon = epsilon
    self.num_iterations = num_iterations
    self.metric = metric
    self.sharding = sharding
    self.threshold = threshold
    self.eps_marginal = eps_marginal
    self.criterion = criterion
    self.inner_iterations = inner_iterations
    self._solver = jax.jit(self._forward_solve)

  def __call__(
      self,
      x: Array,
      y: Array,
      a: Array | None = None,
      b: Array | None = None,
  ) -> SinkhornOutput:
    """Solves the entropic optimal transport problem between point clouds.

    Args:
      x: Source point cloud of shape ``(n, d)``.
      y: Target point cloud of shape ``(m, d)``.
      a: Optional non-negative source probability weights of shape ``(n,)``
        summing to 1. If ``None``, uniform empirical weights ``1 / n`` are used.
      b: Optional non-negative target probability weights of shape ``(m,)``
        summing to 1. If ``None``, uniform empirical weights ``1 / m`` are used.

    Returns:
      A ``SinkhornOutput`` instance holding potentials, costs, and diagnostics.

    Raises:
      ValueError: If shapes or concrete probability weight invariants are
        invalid.
    """
    if a is not None:
      a = jnp.asarray(a)
    if b is not None:
      b = jnp.asarray(b)
    _validate_solve_shapes(x, y, a, b)
    if a is not None:
      _validate_weights(a, "a")
    if b is not None:
      _validate_weights(b, "b")
    return self._solver(x, y, a, b)

  def _marginal_error(
      self, u: Array, v: Array, cost_matrix: Array, a: Array, b: Array
  ) -> Array:
    """Computes L1 marginal violation ``||P 1 - a||_1 + ||P^T 1 - b||_1``."""
    kernel = self._log_gibbs_kernel(u, v, cost_matrix)
    row_marginals = jnp.exp(jax.scipy.special.logsumexp(kernel, axis=1))
    col_marginals = jnp.exp(jax.scipy.special.logsumexp(kernel, axis=0))
    return jnp.sum(jnp.abs(row_marginals - a)) + jnp.sum(
        jnp.abs(col_marginals - b)
    )

  def _forward_solve(
      self,
      x: Array,
      y: Array,
      a: Array | None = None,
      b: Array | None = None,
  ) -> SinkhornOutput:
    """Computation of the cost and the coupling matrix.

    Args:
      x: Collection of points of set A of shape ``(n, d)``.
      y: Collection of points of set B of shape ``(m, d)``.
      a: Optional marginal probability weights of set A of shape ``(n,)``.
      b: Optional marginal probability weights of set B of shape ``(m,)``.

    Returns:
      An instance of SinkhornOutput, which is a named tuple containing the cost
      of the minimizer, the cost matrix, the potentials in log space, and the
      number of iterations required for convergence.
    """
    if a is not None:
      a = jnp.asarray(a)
    if b is not None:
      b = jnp.asarray(b)
    _validate_solve_shapes(x, y, a, b)

    num_x = x.shape[0]
    num_y = y.shape[0]

    # Computes cost matrix.
    cost_matrix = self._compute_cost(x, y)
    # Adds sharding constraints, if sharding is provided.
    if self.sharding is not None:
      cost_matrix = jax.lax.with_sharding_constraint(cost_matrix, self.sharding)

    # Defines the marginal densities as empirical measures when not provided.
    if a is None:
      a = jnp.ones((num_x,), dtype=cost_matrix.dtype) / num_x
    else:
      a = jnp.asarray(a, dtype=cost_matrix.dtype)
    if b is None:
      b = jnp.ones((num_y,), dtype=cost_matrix.dtype) / num_y
    else:
      b = jnp.asarray(b, dtype=cost_matrix.dtype)

    a_reg = a + self.eps_marginal
    b_reg = b + self.eps_marginal
    mask_a = a_reg > 0
    mask_b = b_reg > 0
    safe_a = jnp.where(mask_a, a_reg, 1.0)
    safe_b = jnp.where(mask_b, b_reg, 1.0)
    safe_log_a = jnp.log(safe_a)
    safe_log_b = jnp.log(safe_b)

    # Initialises approximation vectors in log domain (-inf on zero support).
    u = jnp.where(mask_a, jnp.zeros_like(a), -jnp.inf)
    v = jnp.where(mask_b, jnp.zeros_like(b), -jnp.inf)

    def _sweep(v_in: Array) -> tuple[Array, Array]:
      mask_v = jnp.isfinite(v_in)[None, :]
      safe_v = jnp.where(mask_v, v_in[None, :], 0.0)
      logits_u = jnp.where(
          mask_v, (safe_v - cost_matrix) / self.epsilon, -jnp.inf
      )
      u_out = jnp.where(
          mask_a,
          self.epsilon
          * (safe_log_a - jax.scipy.special.logsumexp(logits_u, axis=1)),
          -jnp.inf,
      )

      mask_u = jnp.isfinite(u_out)[:, None]
      safe_u = jnp.where(mask_u, u_out[:, None], 0.0)
      logits_v = jnp.where(
          mask_u, (safe_u - cost_matrix) / self.epsilon, -jnp.inf
      )
      v_out = jnp.where(
          mask_b,
          self.epsilon
          * (safe_log_b - jax.scipy.special.logsumexp(logits_v, axis=0)),
          -jnp.inf,
      )
      return u_out, v_out

    def body_fun(val: State) -> State:
      i, u, v, _ = val
      steps = jnp.minimum(self.inner_iterations, self.num_iterations - i)
      if self.inner_iterations == 1:
        u_prev, v_prev = u, v
        u, v = _sweep(v)
      else:

        def _inner_body(
            idx: int, carry: tuple[Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array]:
          del idx
          _, _, u_curr, v_curr = carry
          u_next, v_next = _sweep(v_curr)
          return (u_curr, v_curr, u_next, v_next)

        u_prev, v_prev, u, v = jax.lax.fori_loop(
            0, steps, _inner_body, (u, v, u, v)
        )

      if self.criterion == "marginal":
        err = self._marginal_error(u, v, cost_matrix, a, b)
      else:
        du = jnp.where(mask_a, u, 0.0) - jnp.where(mask_a, u_prev, 0.0)
        dv = jnp.where(mask_b, v, 0.0) - jnp.where(mask_b, v_prev, 0.0)
        err = jnp.linalg.norm(du) + jnp.linalg.norm(dv)

      return (i + steps, u, v, err)

    def cond_fun(val: State) -> Array:
      i, _, _, err = val
      return jnp.logical_and(err > self.threshold, i < self.num_iterations)

    init_err = jnp.asarray(jnp.inf, dtype=cost_matrix.dtype)
    init_i = jnp.asarray(0, dtype=jnp.int32)
    num_its, u, v, err = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=(init_i, u, v, init_err),
    )

    # Computes transport plan pi = diag(a)*K*diag(b).
    kernel_matrix = self._log_gibbs_kernel(u, v, cost_matrix)
    pi = jnp.exp(kernel_matrix)

    # Sinkhorn distance.
    reg_ot_cost = jnp.sum(pi * cost_matrix, axis=(-2, -1))

    # Checks if the iteration converged.
    converged = jnp.logical_and(err <= self.threshold, jnp.isfinite(err))
    marginal_error = self._marginal_error(u, v, cost_matrix, a, b)

    return SinkhornOutput(
        potentials=(u, v),
        cost_matrix=cost_matrix,
        epsilon=self.epsilon,
        num_iterations=num_its,
        reg_ot_cost=reg_ot_cost,
        converged=converged,
        threshold=self.threshold,
        marginal_error=marginal_error,
        weights=(a, b),
    )

  def _compute_cost(self, x: Array, y: Array) -> Array:
    r"""Computes the cost matrix of the Sinkhorn iteration.

    Args:
      x: Collection of points of set A.
      y: Collection of point of set B.

    Returns:
      A matrix C of size (num_x, num_y) corresponding to the distance between
      each pair of points, namely:

      C_{i,j} = distance(x_i, y_j).

      Here the metric is defined as the sum over the feature dimension of a
      user-prescribed metric. Basically,

      distance(x_i, y_j) = sum_{k=1}^d self.metric(x_{i,k} - y_{j,k}),

      where k is the index of the feature dimension.
    """
    x_ = x[:, None, :]
    y_ = y[None, :, :]
    cost_matrix = jnp.sum(self.metric(x_ - y_), axis=-1)
    return cost_matrix

  def _log_gibbs_kernel(self, u: Array, v: Array, cost_matrix: Array) -> Array:
    """Computes the kernel K = diag(u) exp(-C/eps) diag(v) in log space.

    Args:
      u: The left potential in log space.
      v: The right potential in log space.
      cost_matrix: The cost matrix of the Sinkhorn iteration.

    Returns:
      The rescaled gibbs kernel matrix diag(u) K diag(v) in log space, where K
      is exp(-C/epsilon). This is performed in log space to avoid numerical
      issues.
    """
    kernel = -cost_matrix + u[:, None] + v[None, :]
    kernel /= self.epsilon
    return kernel

  def transport_fn(
      self, potential: Array, y: Array, weights: Array | None = None
  ) -> Callable[[Array], Array]:
    r"""Returns the entropic barycentric projection transport map [2].

    Evaluates the out-of-sample entropic barycentric projection in log domain:

    .. math::

      T_\varepsilon(x) = \sum_{j=1}^m \frac{
        \exp((v_j - c(x, y_j)) / \varepsilon)
      }{
        \sum_{k=1}^m \exp((v_k - c(x, y_k)) / \varepsilon)
      } y_j

    where the scaling-gauge dual potential ``v`` returned by ``SinkhornSolver``
    already absorbs the target log-marginal ``\varepsilon \log b_j``. This form
    is valid for any cost ``self.metric`` and matches Proposition 2 of [2] when
    ``self.metric`` is the squared Euclidean distance.

    Args:
      potential: Scaling-gauge dual potential ``v`` (or ``u``) of shape
        ``(m,)``.
      y: Target points of shape ``(m, d)`` associated with ``potential``.
      weights: Must be ``None``. Passing explicit ``weights`` is disallowed
        because ``potential`` already absorbs ``epsilon * log(b)``.

    Returns:
      A callable mapping ``x`` of shape ``(d,)`` or ``(B, d)`` to transported
      points of the same shape.

    Raises:
      ValueError: If ``weights is not None``, if ``y.ndim != 2``, or if
        ``potential`` is not a 1-D vector of length ``y.shape[0]``.
    """
    if weights is not None:
      raise ValueError(
          "weights must be None in transport_fn because Sinkhorn scaling-gauge"
          " potentials already absorb epsilon * log(b)."
      )
    _validate_potential_and_y(potential, y)
    mask = jnp.isfinite(potential)
    safe_potential = jnp.where(mask, potential, 0.0)

    def _map_one(x_single: Array) -> Array:
      cost = self._compute_cost(x_single[None, :], y)[0]
      logits = jnp.where(mask, (safe_potential - cost) / self.epsilon, -jnp.inf)
      return jax.nn.softmax(logits, axis=0) @ y

    def _map(x: Array) -> Array:
      _validate_query_points(x, y)
      x_2d = x[None, :] if x.ndim == 1 else x
      out = jax.vmap(_map_one)(x_2d)
      return out[0] if x.ndim == 1 else out

    return _map

  def _potential_fn(
      self,
      x: Array,
      potential: Array,
      y: Array,
      weights: Array | None = None,
  ) -> Array:
    r"""Evaluates the entropic dual potential :math:`f_\varepsilon(x)`.

    Implements the dual optimality relation (Eq. 9 / Proposition 2 of [2]):

    .. math::

      f_\varepsilon(x) = -\varepsilon \log \sum_{j=1}^m
        \exp\left(\frac{v_j - c(x, y_j)}{\varepsilon}\right)

    where ``potential`` is in the scaling gauge so ``\varepsilon \log b_j`` is
    already absorbed into ``v_j``.

    Args:
      x: Query point(s) of shape ``(d,)`` or ``(B, d)``.
      potential: Scaling-gauge dual potential of set B with shape ``(m,)``.
      y: Target points of set B with shape ``(m, d)``.
      weights: Must be ``None``. Passing explicit ``weights`` is disallowed
        because ``potential`` already absorbs ``epsilon * log(b)``.

    Returns:
      Scalar potential when ``x.ndim == 1``, or 1-D array of shape ``(B,)`` when
      ``x.ndim == 2``.

    Raises:
      ValueError: If ``weights is not None``, if ``y.ndim != 2``, if
        ``potential`` is not a 1-D vector of length ``y.shape[0]``, if ``x`` is
        not 1-D or 2-D, or if ``x`` and ``y`` have different feature dimensions.
    """
    if weights is not None:
      raise ValueError(
          "weights must be None in _potential_fn because Sinkhorn"
          " scaling-gauge potentials already absorb epsilon * log(b)."
      )
    _validate_potential_and_y(potential, y)
    _validate_query_points(x, y)

    x_2d = x[None, :] if x.ndim == 1 else x
    cost = self._compute_cost(x_2d, y)
    mask = jnp.isfinite(potential)[None, :]
    safe_potential = jnp.where(mask, potential[None, :], 0.0)
    z = jnp.where(mask, (safe_potential - cost) / self.epsilon, -jnp.inf)
    lse = -self.epsilon * jax.scipy.special.logsumexp(z, axis=-1)
    return lse[0] if x.ndim == 1 else lse

  def _gradient_transport_fn(
      self, potential: Array, y: Array
  ) -> Callable[[Array], Array]:
    r"""Transport map via Proposition 2 of [2] for squared-Euclidean cost.

    Computes ``T(x) = x - 0.5 * \nabla f_{\epsilon}(x)``, which equals the
    entropic barycentric projection when ``self.metric(z) = z**2``.

    Args:
      potential: Scaling-gauge dual potential of shape ``(m,)``.
      y: Target points of shape ``(m, d)``.

    Returns:
      A callable mapping ``x`` of shape ``(d,)`` or ``(B, d)`` to transported
      points of the same shape.

    Raises:
      ValueError: If ``y.ndim != 2`` or ``potential`` does not have shape
        ``(y.shape[0],)``.
    """
    _validate_potential_and_y(potential, y)

    f_eps = lambda pt: self._potential_fn(pt, potential, y, weights=None)
    grad_one = lambda pt: pt - 0.5 * jax.grad(f_eps)(pt)

    def _map(x: Array) -> Array:
      _validate_query_points(x, y)
      x_2d = x[None, :] if x.ndim == 1 else x
      out = jax.vmap(grad_one)(x_2d)
      return out[0] if x.ndim == 1 else out

    return _map

  def transport_fn_direct(
      self, potential: Array, y: Array, weights: Array | None = None
  ) -> Callable[[Array], Array]:
    """Deprecated alias for ``transport_fn``."""
    logging.warning(
        "SinkhornSolver.transport_fn_direct is deprecated; use"
        " SinkhornSolver.transport_fn instead."
    )
    return self.transport_fn(potential, y, weights=weights)
