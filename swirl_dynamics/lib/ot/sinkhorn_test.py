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
from swirl_dynamics.lib import ot

jax.config.update("jax_enable_x64", True)


class SinkhornTest(parameterized.TestCase):

  def test_sinkhorn_barycentric_projection_and_1d_query(self):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(k1, (15, 2), dtype=jnp.float64)
    y = jax.random.normal(k2, (20, 2), dtype=jnp.float64)
    solver = ot.SinkhornSolver(epsilon=0.5, threshold=1e-12, num_iterations=500)
    out = solver(x, y)
    self.assertTrue(bool(out.converged))
    t_grad = solver.transport_fn(out.gv, y)(x)
    t_direct = solver.transport_fn_direct(out.gv, y)(x)
    self.assertLess(float(jnp.max(jnp.abs(t_grad - t_direct))), 1e-10)
    self.assertEqual(solver.transport_fn(out.gv, y)(x[0]).shape, (2,))


if __name__ == "__main__":
  absltest.main()
