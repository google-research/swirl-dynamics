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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import ot


class DivergenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    prev_x64 = jax.config.jax_enable_x64
    self.addCleanup(jax.config.update, "jax_enable_x64", prev_x64)
    jax.config.update("jax_enable_x64", True)

  @parameterized.named_parameters(
      dict(testcase_name="uniform", mode="uniform"),
      dict(testcase_name="dirichlet", mode="dirichlet"),
      dict(testcase_name="zero_weight", mode="zero_weight"),
  )
  def test_self_divergence_zero(self, mode: str):
    kx, ka = jax.random.split(jax.random.PRNGKey(0))
    n = 20
    x = 0.5 * jax.random.normal(kx, (n, 2))
    a_dir = jax.random.dirichlet(ka, jnp.ones(n))
    a_zero = a_dir.at[0].set(0.0)
    a_zero = a_zero / jnp.sum(a_zero)

    weights_by_mode = {
        "uniform": None,
        "dirichlet": a_dir,
        "zero_weight": a_zero,
    }
    a = weights_by_mode[mode]

    solver = ot.SinkhornSolver(
        epsilon=0.3, num_iterations=2000, threshold=1e-12
    )
    res = ot.sinkhorn_divergence_with_potentials(x, x, a=a, b=a, solver=solver)
    self.assertTrue(bool(res.converged))
    self.assertEqual(res.self_potential_x.shape, (n,))
    self.assertEqual(res.self_potential_y.shape, (n,))
    self.assertLess(abs(float(res.divergence)), 1e-12)

    div_scalar = ot.sinkhorn_divergence(x, x, a=a, b=a, solver=solver)
    self.assertLess(abs(float(div_scalar)), 1e-12)

  def test_positivity_and_symmetry(self):
    kx, ky, ka, kb = jax.random.split(jax.random.PRNGKey(1), 4)
    x = 0.5 * jax.random.normal(kx, (20, 2))
    y = 0.5 * jax.random.normal(ky, (24, 2)) + 0.6
    a = jax.random.dirichlet(ka, jnp.ones(20))
    b = jax.random.dirichlet(kb, jnp.ones(24))

    solver = ot.SinkhornSolver(
        epsilon=0.4, num_iterations=2000, threshold=1e-12
    )
    div_xy = ot.sinkhorn_divergence(x, y, a=a, b=b, solver=solver)
    div_yx = ot.sinkhorn_divergence(y, x, a=b, b=a, solver=solver)
    self.assertGreater(float(div_xy), 0.0)
    np.testing.assert_allclose(div_xy, div_yx, atol=1e-10)

  @parameterized.named_parameters(
      dict(
          testcase_name="uniform_unchunked",
          nonuniform=False,
          batch_size=None,
          precompute_ot_xy=False,
      ),
      dict(
          testcase_name="dirichlet_chunked",
          nonuniform=True,
          batch_size=3,
          precompute_ot_xy=False,
      ),
      dict(
          testcase_name="dirichlet_precomputed_ot_xy",
          nonuniform=True,
          batch_size=None,
          precompute_ot_xy=True,
      ),
  )
  def test_debiased_transport_fn_identity_on_identical_sets(
      self,
      nonuniform: bool,
      batch_size: int | None,
      precompute_ot_xy: bool,
  ):
    kx, keval, ka = jax.random.split(jax.random.PRNGKey(2), 3)
    x = 0.5 * jax.random.normal(kx, (20, 2))
    q = 0.5 * jax.random.normal(keval, (8, 2))
    a = jax.random.dirichlet(ka, jnp.ones(20)) if nonuniform else None

    solver = ot.SinkhornSolver(
        epsilon=0.4, num_iterations=2000, threshold=1e-12
    )
    ot_xy = solver(x, x, a=a, b=a) if precompute_ot_xy else None
    deb_fn = ot.debiased_transport_fn(
        solver, x, x, a=a, b=a, batch_size=batch_size, ot_xy=ot_xy
    )
    np.testing.assert_allclose(deb_fn(q), q, atol=1e-11)
    np.testing.assert_allclose(deb_fn(q[0]), q[0], atol=1e-11)

  def test_jacobian_min_eigenvalue_monotone_and_non_monotone(self):
    kx, ky, keval = jax.random.split(jax.random.PRNGKey(3), 3)
    x = 0.4 * jax.random.normal(kx, (24, 2))
    y = 0.6 * jax.random.normal(ky, (24, 2)) + jnp.array([0.4, -0.2])
    q = 0.4 * jax.random.normal(keval, (8, 2))

    solver = ot.SinkhornSolver(
        epsilon=0.5, num_iterations=2000, threshold=1e-12
    )
    deb_gauss = ot.debiased_transport_fn(solver, x, y)
    eigs_batch = ot.jacobian_min_eigenvalue(deb_gauss, q)
    self.assertEqual(eigs_batch.shape, (8,))
    self.assertTrue(bool(jnp.all(eigs_batch > 0.0)))

    eig_single = ot.jacobian_min_eigenvalue(deb_gauss, q[0])
    self.assertEqual(eig_single.shape, ())
    np.testing.assert_allclose(eig_single, eigs_batch[0], atol=1e-12)

    # Pooladian, Cuturi & Niles-Weed (2022) closed-form non-monotone example:
    # X = {-2, +2}, Y = {0}, eps = 2.0 => T'_deb(0) = 1 - 4 = -3.0.
    x_counter = jnp.array([[-2.0], [2.0]])
    y_counter = jnp.array([[0.0]])
    solver_counter = ot.SinkhornSolver(
        epsilon=2.0, num_iterations=2000, threshold=1e-13
    )
    deb_counter = ot.debiased_transport_fn(solver_counter, x_counter, y_counter)
    eig_zero = ot.jacobian_min_eigenvalue(deb_counter, jnp.array([0.0]))
    np.testing.assert_allclose(eig_zero, -3.0, atol=1e-10)

  def test_validation_errors(self):
    solver = ot.SinkhornSolver(epsilon=0.5)
    x = jnp.ones((4, 2))
    y = jnp.ones((5, 2))
    a_nonunif = jnp.array([0.1, 0.2, 0.3, 0.4])
    b_nonunif = jnp.array([0.1, 0.1, 0.2, 0.3, 0.3])
    out = solver(x, y, a=a_nonunif, b=b_nonunif)

    unweighted_out = ot.SinkhornOutput(
        potentials=out.potentials,
        cost_matrix=out.cost_matrix,
        epsilon=out.epsilon,
        weights=None,
    )
    deb_unweighted = ot.debiased_transport_fn(
        solver, x, y, a=a_nonunif, b=b_nonunif, ot_xy=unweighted_out
    )
    deb_weighted = ot.debiased_transport_fn(
        solver, x, y, a=a_nonunif, b=b_nonunif, ot_xy=out
    )
    np.testing.assert_allclose(deb_unweighted(x), deb_weighted(x), atol=1e-12)

    x_int = jnp.array([[0], [1]], dtype=jnp.int32)
    y_int = jnp.array([[1], [2]], dtype=jnp.int32)
    div_int = ot.sinkhorn_divergence_with_potentials(
        x_int, y_int, solver=solver
    )
    self.assertTrue(bool(jnp.isfinite(div_int.divergence)))
    deb_int = ot.debiased_transport_fn(
        solver,
        x_int,
        y_int,
        a=jnp.array([0.4, 0.6]),
        ot_xy=ot.SinkhornOutput(
            potentials=div_int.ot_xy.potentials,
            cost_matrix=div_int.ot_xy.cost_matrix,
            epsilon=div_int.ot_xy.epsilon,
            weights=None,
        ),
    )
    self.assertEqual(deb_int(x_int).shape, (2, 1))

    with self.assertRaisesRegex(ValueError, "ot_xy potentials"):
      ot.debiased_transport_fn(solver, x[:2], y, ot_xy=out)
    with self.assertRaisesRegex(ValueError, "ot_xy.weights"):
      ot.debiased_transport_fn(
          solver, x, y, a=jnp.ones((4,)) / 4, b=b_nonunif, ot_xy=out
      )
    with self.assertRaisesRegex(ValueError, "ot_xy.weights"):
      ot.debiased_transport_fn(
          solver, x, y, a=a_nonunif, b=jnp.ones((5,)) / 5, ot_xy=out
      )
    with self.assertRaisesRegex(ValueError, "sum to 1"):
      ot.debiased_transport_fn(
          solver, x, y, a=jnp.ones((4,)), b=b_nonunif, ot_xy=unweighted_out
      )
    with self.assertRaisesRegex(ValueError, "points"):
      ot.jacobian_min_eigenvalue(
          solver.transport_fn(out.gv, y), jnp.zeros((2, 2, 2))
      )


if __name__ == "__main__":
  absltest.main()
