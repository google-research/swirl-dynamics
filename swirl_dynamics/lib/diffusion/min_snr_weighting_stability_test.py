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

"""Numeric boundary tests for Min-SNR diffusion loss weighting."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import diffusion


def _reference_min_snr(sigma, data_std, gamma):
  sigma = np.asarray(sigma, dtype=np.float64)
  threshold = data_std / np.sqrt(gamma)
  # Compute the finite analytic branches independently in double precision.
  low = gamma / data_std**2 * (1 + (sigma / data_std) ** 2)
  high = 1 / data_std**2 + np.square(1 / np.where(sigma == 0, 1, sigma))
  return np.where(np.abs(sigma) <= threshold, low, high)


class MinSnrWeightingStabilityTest(parameterized.TestCase):

  @parameterized.product(
      data_std=(0.5, 1.0, 2.0), gamma=(0.5, 5.0, 20.0), compiled=(False, True)
  )
  def test_zero_small_and_large_noise_match_reference(
      self, data_std, gamma, compiled
  ):
    sigma = jnp.array(
        [0.0, 1e-4, 0.05, 0.2, 1.0, 10.0, 1e20], dtype=jnp.float32
    )
    fn = diffusion.min_snr_edm_weighting(data_std, gamma)
    actual = (jax.jit(fn) if compiled else fn)(sigma)
    expected = _reference_min_snr(sigma, data_std, gamma)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-6)
    self.assertTrue(np.all(np.isfinite(actual)))
    self.assertEqual(actual.dtype, jnp.float32)

  @parameterized.parameters(jnp.float16, jnp.bfloat16, jnp.int16)
  def test_low_precision_and_integer_inputs_promote_before_arithmetic(
      self, dtype
  ):
    sigma = jnp.array([0, 1e-4, 1, 10], dtype=dtype)
    actual = jax.jit(diffusion.min_snr_edm_weighting(1.0, gamma=5.0))(sigma)
    expected = _reference_min_snr(sigma, 1.0, 5.0)
    self.assertEqual(actual.dtype, jnp.float32)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)

  @parameterized.parameters(False, True)
  def test_zero_noise_derivative_is_finite(self, compiled):
    weight = diffusion.min_snr_edm_weighting(data_std=2.0, gamma=5.0)
    fn = jax.grad(weight)
    if compiled:
      fn = jax.jit(fn)
    for value in (0.0, 1e-4, 0.1):
      expected = 2 * 5.0 * value / 2.0**4
      actual = fn(jnp.array(value, dtype=jnp.float32))
      np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
      self.assertTrue(np.isfinite(actual))

  def test_gamma_zero_returns_zero_at_noise_boundary(self):
    sigma = jnp.array([0.0, 1e-6, 1.0, 1e20])
    result = jax.jit(diffusion.min_snr_edm_weighting(gamma=0.0))(sigma)
    np.testing.assert_array_equal(result, jnp.zeros_like(sigma))

  @parameterized.parameters((0.0, 5.0), (-1.0, 5.0), (1.0, -1.0))
  def test_invalid_weight_parameters(self, data_std, gamma):
    with self.assertRaises(ValueError):
      diffusion.min_snr_edm_weighting(data_std=data_std, gamma=gamma)

  def test_float64_precision_preserved(self):
    with jax.enable_x64():
      sigma = jnp.array([0.0, 0.01, 0.5, 10.0], dtype=jnp.float64)
      result = diffusion.min_snr_edm_weighting(1.25, 3.0)(sigma)
      self.assertEqual(result.dtype, jnp.float64)
      np.testing.assert_allclose(
          result, _reference_min_snr(sigma, 1.25, 3.0), rtol=1e-12
      )


if __name__ == '__main__':
  absltest.main()
