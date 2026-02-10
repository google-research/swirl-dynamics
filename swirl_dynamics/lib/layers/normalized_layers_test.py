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
from swirl_dynamics.lib.layers import normalized_layers


class NormalizedLayersTest(parameterized.TestCase):

  def test_compute_norm(self):
    x = jnp.array([[[[1.0, 2.0, 3.0, 4.0]]]])
    norm = normalized_layers.compute_norm(x, axis=(1, 2, 3))
    np.testing.assert_allclose(norm, jnp.array([[[[5.477226]]]]), rtol=1e-5)

  def test_normalize(self):
    x = jax.random.normal(jax.random.PRNGKey(42), (4, 8, 8, 3))
    x_normalized = normalized_layers.normalize(x, axis=(1, 2, 3))
    # Test that the norm of the output is close to 1
    self.assertAlmostEqual(
        jnp.linalg.norm(x) / jnp.linalg.norm(x_normalized), 1.0, delta=1e-1
    )

  def test_mp_silu(self):
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    y = normalized_layers.mp_silu(x)
    expected = jax.nn.silu(x) / 0.596
    np.testing.assert_allclose(y, expected, rtol=1e-5)

  def test_mp_convex(self):
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 1.0])
    t = 0.5
    z = normalized_layers.mp_convex(x, y, t)
    expected = (x * (1 - t) + y * t) / jnp.sqrt((1 - t) ** 2 + t**2)
    np.testing.assert_allclose(z, expected, rtol=1e-5)

  @parameterized.parameters(
      ((8, 8, 4), (8, 8, 4), 2, 0.5, (8, 8, 8)),
      ((8, 4, 8), (8, 4, 8), 1, 0.3, (8, 8, 8)),
  )
  def test_mp_cat_output_shape(self, x_shape, y_shape, dim, t, expected_shape):
    x = jnp.ones(x_shape)
    y = jnp.ones(y_shape)
    out = normalized_layers.mp_cat(x, y, dim=dim, t=t)
    self.assertEqual(out.shape, expected_shape)

  def test_mp_cat_raises_error_for_invalid_t(self):
    x = jnp.ones((8, 8, 4))
    y = jnp.ones((8, 8, 4))
    with self.assertRaisesRegex(ValueError, "t must be in"):
      normalized_layers.mp_cat(x, y, dim=2, t=1.1)


if __name__ == "__main__":
  absltest.main()
