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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import ot


class SinkhornTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    prev_x64 = jax.config.jax_enable_x64
    self.addCleanup(jax.config.update, "jax_enable_x64", prev_x64)
    jax.config.update("jax_enable_x64", True)

  def test_public_exports_importable(self):
    self.assertNotEmpty(ot.__all__)
    self.assertEqual(ot.__all__, sorted(ot.__all__))
    self.assertLen(ot.__all__, len(set(ot.__all__)))
    for name in ot.__all__:
      with self.subTest(symbol=name):
        self.assertIsNotNone(getattr(ot, name, None))

  @parameterized.named_parameters(
      dict(testcase_name="uniform", mode="uniform"),
      dict(testcase_name="dirichlet", mode="dirichlet"),
      dict(testcase_name="zero_weight", mode="zero_weight"),
  )
  def test_sinkhorn_solver_marginals_potentials_and_costs(self, mode: str):
    kx, ky, ka, kb = jax.random.split(jax.random.PRNGKey(0), 4)
    n, m = 16, 20
    x = jax.random.normal(kx, (n, 3))
    y = jax.random.normal(ky, (m, 3)) + 0.4
    a_dir = jax.random.dirichlet(ka, jnp.ones(n))
    b_dir = jax.random.dirichlet(kb, jnp.ones(m))
    a_zero = a_dir.at[0].set(0.0)
    a_zero = a_zero / jnp.sum(a_zero)
    b_zero = b_dir.at[-1].set(0.0)
    b_zero = b_zero / jnp.sum(b_zero)

    weights_by_mode = {
        "uniform": (None, None),
        "dirichlet": (a_dir, b_dir),
        "zero_weight": (a_zero, b_zero),
    }
    a, b = weights_by_mode[mode]

    solver = ot.SinkhornSolver(
        epsilon=0.25, num_iterations=2000, threshold=1e-12
    )
    out = solver(x, y, a=a, b=b)
    self.assertTrue(bool(out.converged))
    a_vec, b_vec = out.resolved_weights()
    plan = out.transport_plan
    np.testing.assert_allclose(jnp.sum(plan, axis=1), a_vec, atol=1e-11)
    np.testing.assert_allclose(jnp.sum(plan, axis=0), b_vec, atol=1e-11)

    f, g = out.kantorovich_potentials
    self.assertTrue(bool(jnp.all(jnp.isfinite(f))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(g))))
    reconstructed = (
        jnp.exp((f[:, None] + g[None, :] - out.cost_matrix) / out.epsilon)
        * a_vec[:, None]
        * b_vec[None, :]
    )
    np.testing.assert_allclose(reconstructed, plan, atol=1e-12)
    dual_cost = jnp.dot(f, a_vec) + jnp.dot(g, b_vec)
    np.testing.assert_allclose(out.entropic_ot_cost, dual_cost, atol=1e-11)

  def test_sinkhorn_output_default_weights_and_epsilon_validation(self):
    x = jnp.array([[0.0], [1.0]])
    y = jnp.array([[0.0], [1.0], [2.0]])
    solver = ot.SinkhornSolver(epsilon=0.5, threshold=1e-12, num_iterations=500)
    solved = solver(x, y)
    unweighted = ot.SinkhornOutput(
        potentials=solved.potentials,
        cost_matrix=solved.cost_matrix,
        epsilon=solved.epsilon,
        weights=None,
    )
    a_res, b_res = unweighted.resolved_weights()
    np.testing.assert_allclose(a_res, jnp.ones((2,)) / 2.0)
    np.testing.assert_allclose(b_res, jnp.ones((3,)) / 3.0)
    np.testing.assert_allclose(
        unweighted.kantorovich_potentials[0],
        solved.kantorovich_potentials[0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        unweighted.entropic_ot_cost, solved.entropic_ot_cost, atol=1e-12
    )

    bad_out = ot.SinkhornOutput(
        potentials=solved.potentials,
        cost_matrix=solved.cost_matrix,
        epsilon=0.0,
    )
    with self.assertRaisesRegex(ValueError, "Epsilon should be positive"):
      _ = bad_out.transport_plan
    with self.assertRaisesRegex(ValueError, "Epsilon should be positive"):
      _ = bad_out.kantorovich_potentials

  @parameterized.named_parameters(
      dict(
          testcase_name="marginal_inner_5",
          criterion="marginal",
          inner_iterations=5,
      ),
      dict(
          testcase_name="potentials_inner_1",
          criterion="potentials",
          inner_iterations=1,
      ),
      dict(
          testcase_name="potentials_inner_5",
          criterion="potentials",
          inner_iterations=5,
      ),
  )
  def test_stopping_criteria_and_inner_iterations_agree(
      self, criterion: Any, inner_iterations: int
  ):
    kx, ky, ka, kb = jax.random.split(jax.random.PRNGKey(1), 4)
    x = jax.random.normal(kx, (16, 3))
    y = jax.random.normal(ky, (20, 3)) + 0.4
    a = jax.random.dirichlet(ka, jnp.ones(16))
    b = jax.random.dirichlet(kb, jnp.ones(20))

    baseline = ot.SinkhornSolver(
        epsilon=0.25, num_iterations=2000, threshold=1e-12
    )(x, y, a=a, b=b)
    candidate = ot.SinkhornSolver(
        epsilon=0.25,
        num_iterations=2000,
        threshold=1e-12,
        criterion=criterion,
        inner_iterations=inner_iterations,
    )(x, y, a=a, b=b)
    self.assertTrue(bool(candidate.converged))
    np.testing.assert_allclose(
        candidate.transport_plan, baseline.transport_plan, atol=1e-10
    )

  @parameterized.named_parameters(
      dict(testcase_name="c1_eps05", cost_val=1.0, eps=0.5),
      dict(testcase_name="c2_eps02", cost_val=2.0, eps=0.2),
  )
  def test_two_point_closed_form_oracle(self, cost_val: float, eps: float):
    x = jnp.array([[0.0], [jnp.sqrt(cost_val)]])
    y = jnp.array([[0.0], [jnp.sqrt(cost_val)]])
    solver = ot.SinkhornSolver(epsilon=eps, threshold=1e-13, num_iterations=500)
    out = solver(x, y)
    self.assertTrue(bool(out.converged))
    self.assertIsNotNone(out.reg_ot_cost)

    s = 1.0 / (1.0 + np.exp(-cost_val / eps))
    expected_plan = 0.5 * np.array([[s, 1.0 - s], [1.0 - s, s]])
    expected_cost = cost_val * (1.0 - s)
    expected_transport = np.array(
        [[np.sqrt(cost_val) * (1.0 - s)], [np.sqrt(cost_val) * s]]
    )
    np.testing.assert_allclose(out.transport_plan, expected_plan, atol=1e-10)
    np.testing.assert_allclose(out.reg_ot_cost, expected_cost, atol=1e-10)
    np.testing.assert_allclose(
        solver.transport_fn(out.gv, y)(x), expected_transport, atol=1e-10
    )

  def test_transport_fn_1d_2d_chunked_and_barycentric_identity(self):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
    x = jax.random.normal(k1, (15, 2))
    y = jax.random.normal(k2, (20, 2)) + 0.3
    q = jax.random.normal(k3, (9, 2))
    solver = ot.SinkhornSolver(
        epsilon=0.5, threshold=1e-12, num_iterations=1000
    )
    out = solver(x, y)
    self.assertTrue(bool(out.converged))

    t_bary = solver.transport_fn(out.gv, y)
    t_chunk = solver.transport_fn(out.gv, y, batch_size=4)
    t_grad = solver._gradient_transport_fn(out.gv, y)
    with self.assertLogs(level="WARNING"):
      t_direct = solver.transport_fn_direct(out.gv, y)

    mapped_2d = t_bary(q)
    self.assertEqual(mapped_2d.shape, (9, 2))
    np.testing.assert_allclose(t_chunk(q), mapped_2d, atol=1e-12)
    np.testing.assert_allclose(t_grad(q), mapped_2d, atol=1e-12)
    np.testing.assert_allclose(t_direct(q), mapped_2d, atol=1e-12)

    mapped_1d = t_bary(q[0])
    self.assertEqual(mapped_1d.shape, (2,))
    np.testing.assert_allclose(mapped_1d, mapped_2d[0], atol=1e-12)
    np.testing.assert_allclose(t_grad(q[0]), mapped_2d[0], atol=1e-12)

    expected_in_sample = (out.transport_plan @ y) / jnp.sum(
        out.transport_plan, axis=1, keepdims=True
    )
    np.testing.assert_allclose(t_bary(x), expected_in_sample, atol=1e-10)

    x_int = jnp.array([[0], [1]], dtype=jnp.int32)
    y_int = jnp.array([[1], [2]], dtype=jnp.int32)
    out_int = solver(x_int, y_int)
    self.assertTrue(jnp.issubdtype(out_int.cost_matrix.dtype, jnp.floating))
    np.testing.assert_allclose(jnp.sum(out_int.transport_plan), 1.0, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name="zero_epsilon",
          kwargs={"epsilon": 0.0},
          error_regex="epsilon",
      ),
      dict(
          testcase_name="negative_num_iterations",
          kwargs={"epsilon": 0.5, "num_iterations": -1},
          error_regex="num_iterations",
      ),
      dict(
          testcase_name="negative_threshold",
          kwargs={"epsilon": 0.5, "threshold": -1e-4},
          error_regex="threshold",
      ),
      dict(
          testcase_name="negative_eps_marginal",
          kwargs={"epsilon": 0.5, "eps_marginal": -1e-6},
          error_regex="eps_marginal",
      ),
      dict(
          testcase_name="invalid_criterion",
          kwargs={"epsilon": 0.5, "criterion": "invalid"},
          error_regex="criterion",
      ),
      dict(
          testcase_name="zero_inner_iterations",
          kwargs={"epsilon": 0.5, "inner_iterations": 0},
          error_regex="inner_iterations",
      ),
  )
  def test_init_validation_errors(
      self, kwargs: dict[str, Any], error_regex: str
  ):
    with self.assertRaisesRegex(ValueError, error_regex):
      ot.SinkhornSolver(**kwargs)

  @parameterized.named_parameters(
      dict(
          testcase_name="non_2d_x",
          x_shape=(4,),
          y_shape=(5, 2),
          a=None,
          b=None,
          error_regex="2d arrays",
      ),
      dict(
          testcase_name="empty_x",
          x_shape=(0, 2),
          y_shape=(5, 2),
          a=None,
          b=None,
          error_regex="non-empty",
      ),
      dict(
          testcase_name="feature_dim_mismatch",
          x_shape=(4, 2),
          y_shape=(5, 3),
          a=None,
          b=None,
          error_regex="feature dimension",
      ),
      dict(
          testcase_name="wrong_a_length",
          x_shape=(4, 2),
          y_shape=(5, 2),
          a=(0.5, 0.5),
          b=None,
          error_regex="1D array of length x.shape",
      ),
      dict(
          testcase_name="wrong_b_length",
          x_shape=(4, 2),
          y_shape=(5, 2),
          a=None,
          b=(0.5, 0.5),
          error_regex="1D array of length y.shape",
      ),
      dict(
          testcase_name="nonfinite_a",
          x_shape=(2, 2),
          y_shape=(2, 2),
          a=(float("nan"), 1.0),
          b=None,
          error_regex="finite",
      ),
      dict(
          testcase_name="negative_a",
          x_shape=(2, 2),
          y_shape=(2, 2),
          a=(-0.2, 1.2),
          b=None,
          error_regex="non-negative",
      ),
      dict(
          testcase_name="unnormalized_a",
          x_shape=(4, 2),
          y_shape=(5, 2),
          a=(1.0, 1.0, 1.0, 1.0),
          b=None,
          error_regex="sum to 1",
      ),
  )
  def test_solve_validation_errors(
      self,
      x_shape: tuple[int, ...],
      y_shape: tuple[int, ...],
      a: tuple[float, ...] | None,
      b: tuple[float, ...] | None,
      error_regex: str,
  ):
    solver = ot.SinkhornSolver(epsilon=0.5)
    x = jnp.ones(x_shape)
    y = jnp.ones(y_shape)
    a_arr = jnp.asarray(a) if a is not None else None
    b_arr = jnp.asarray(b) if b is not None else None
    with self.assertRaisesRegex(ValueError, error_regex):
      solver(x, y, a=a_arr, b=b_arr)

  def test_transport_fn_validation_errors(self):
    solver = ot.SinkhornSolver(epsilon=0.5)
    x = jnp.ones((4, 2))
    y = jnp.ones((5, 2))
    out = solver(x, y)

    with self.assertRaisesRegex(ValueError, "weights"):
      solver.transport_fn(out.gv, y, weights=jnp.ones((5,)) / 5)
    with self.assertRaisesRegex(ValueError, "batch_size"):
      solver.transport_fn(out.gv, y, batch_size=0)
    with self.assertRaisesRegex(ValueError, "batch_size"):
      solver.transport_fn(out.gv, y, batch_size=True)
    with self.assertRaisesRegex(ValueError, "y must be a 2D array"):
      solver.transport_fn(out.gv, jnp.ones((5,)))
    with self.assertRaisesRegex(ValueError, "potential must be a 1D array"):
      solver.transport_fn(out.gv[:2], y)

    t_fn = solver.transport_fn(out.gv, y)
    with self.assertRaisesRegex(ValueError, "1D or 2D array"):
      t_fn(jnp.ones((2, 2, 2)))
    with self.assertRaisesRegex(ValueError, "feature dimension"):
      t_fn(jnp.ones((4, 3)))


if __name__ == "__main__":
  absltest.main()
