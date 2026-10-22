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

"""Tests for distribution distances."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.projects.ergodic import measure_distances


def _reference_mmd(x, y, bandwidth):
  """Independent NumPy pairwise-distance reference."""
  x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
  y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)

  def kernel(a, b):
    squared_distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return sum(scale**2 / (scale**2 + squared_distances) for scale in bandwidth)

  return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()


class MmdTest(parameterized.TestCase):

  @parameterized.product(
      x_size=(1, 3, 5),
      y_size=(2, 4),
      jit=(False, True),
  )
  def test_unequal_sample_counts(self, x_size, y_size, jit):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(x_size, 2, 3)).astype(np.float32)
    y = rng.normal(size=(y_size, 2, 3)).astype(np.float32)
    score = measure_distances.mmd
    actual = (jax.jit(score) if jit else score)(jnp.asarray(x), jnp.asarray(y))
    expected = _reference_mmd(x, y, (0.2, 0.5, 0.9, 1.3))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

  @parameterized.parameters((1, 4), (4, 1), (2, 5), (3, 3))
  def test_custom_bandwidth_and_symmetry(self, x_size, y_size):
    x = jnp.arange(x_size * 2, dtype=jnp.float32).reshape(x_size, 2)
    y = jnp.arange(y_size * 2, dtype=jnp.float32).reshape(y_size, 2) / 3
    bandwidth = (0.4, 1.2)
    expected = _reference_mmd(x, y, bandwidth)
    actual = measure_distances.mmd(x, y, bandwidth=bandwidth)
    reversed_actual = measure_distances.mmd(y, x, bandwidth=bandwidth)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(reversed_actual, actual, rtol=2e-5, atol=2e-5)

  def test_identical_distributions_with_different_sample_counts(self):
    x = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    y = jnp.repeat(x, repeats=3, axis=0)
    self.assertAlmostEqual(float(measure_distances.mmd(x, y)), 0.0, places=5)


if __name__ == '__main__':
  absltest.main()
