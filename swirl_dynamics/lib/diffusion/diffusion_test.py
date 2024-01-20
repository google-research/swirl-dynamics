# Copyright 2024 The swirl_dynamics Authors.
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
from swirl_dynamics.lib.diffusion import diffusion


class DiffusionTest(parameterized.TestCase):

  @parameterized.parameters((50.0, 0.0, 1.5), (100.0, -1.5, 1.5))
  def test_tangent_noise_schedule(self, clip_max, start, end):
    sigma = diffusion.tangent_noise_schedule(clip_max, start, end)
    self.assertAlmostEqual(sigma(1.0), clip_max, places=3)
    self.assertAlmostEqual(sigma(0.0), 0, places=8)

    test_points = np.random.default_rng(1234).uniform(0.05, 1.0, size=(10,))
    np.testing.assert_allclose(
        sigma.inverse(sigma(test_points)), test_points, rtol=1e-5
    )

  @parameterized.parameters((1.0, 0.0), (-2.0, 0.0), (0.0, 2.0))
  def test_tangent_schedule_invalid_start_end_points(self, start, end):
    with self.assertRaises(ValueError):
      diffusion.tangent_noise_schedule(100.0, start, end)

  @parameterized.parameters(
      {"clip_max": 50.0, "p": 1.0, "start": 0.0, "end": 1.0},
      {"clip_max": 100.0, "p": 2.0, "start": 0.0, "end": 1.0},
      {"clip_max": 100.0, "p": 0.5, "start": 2.0, "end": 100.0},
  )
  def test_power_noise_schedule(self, clip_max, p, start, end):
    sigma = diffusion.power_noise_schedule(clip_max, p, start, end)
    self.assertAlmostEqual(sigma(1.0), clip_max, places=3)
    self.assertEqual(sigma(0.0), 0)

    test_points = np.random.default_rng(2345).uniform(0.05, 1.0, size=(10,))
    np.testing.assert_allclose(
        sigma.inverse(sigma(test_points)), test_points, rtol=1e-5
    )

  @parameterized.parameters((0.0, 0.0, 1.0), (1.0, -2.0, 0.0), (1.0, 2.0, 0.0))
  def test_power_schedule_invalid_start_end_points(self, p, start, end):
    with self.assertRaises(ValueError):
      diffusion.power_noise_schedule(100.0, p, start, end)

  @parameterized.parameters(
      {"clip_max": 50.0, "base": 2.0, "start": 0.0, "end": 5.0},
      {"clip_max": 100.0, "base": 3.0, "start": -1.0, "end": 5.0},
  )
  def test_exponential_noise_schedule(self, clip_max, base, start, end):
    sigma = diffusion.exponential_noise_schedule(clip_max, base, start, end)
    self.assertAlmostEqual(sigma(1.0), clip_max, places=3)
    self.assertEqual(sigma(0.0), 0)

    test_points = np.random.default_rng(2345).uniform(0.05, 1.0, size=(10,))
    np.testing.assert_allclose(
        sigma.inverse(sigma(test_points)), test_points, rtol=1e-5
    )

  @parameterized.parameters((1.0, 0.0, 1.0), (2.0, 2.0, 0.0))
  def test_exponential_schedule_invalid_start_end_points(
      self, base, start, end
  ):
    with self.assertRaises(ValueError):
      diffusion.exponential_noise_schedule(100.0, base, start, end)

  @parameterized.parameters(
      diffusion.tangent_noise_schedule, diffusion.power_noise_schedule
  )
  def test_logsnr_and_sigma_transforms(self, schedule):
    sigma = schedule(clip_max=100.0)
    logsnr = diffusion.sigma2logsnr(sigma)
    self.assertTrue(jnp.isinf(logsnr(0.0)))
    self.assertAlmostEqual(logsnr(1.0), -2 * np.log(100), places=3)

    sigma2 = diffusion.logsnr2sigma(logsnr)
    test_points = np.random.default_rng(345).uniform(0.05, 1.0, size=(10,))
    np.testing.assert_allclose(
        sigma(test_points), sigma2(test_points), rtol=1e-5
    )
    np.testing.assert_allclose(
        test_points, sigma2.inverse(sigma(test_points)), rtol=1e-5
    )

  @parameterized.product(data_std=(1.0, 2.0))
  def test_create_vp(self, data_std):
    sigma = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_preserving(sigma, data_std)
    test_points = np.random.default_rng(678).uniform(0.01, 1.0, size=(10,))
    # verify that variance is indeed preserved
    variance = np.square(scheme.scale(test_points)) * (
        np.square(data_std) + np.square(scheme.sigma(test_points))
    )
    np.testing.assert_allclose(variance, np.square(data_std), rtol=1e-5)
    # verify the inverse is correct
    np.testing.assert_allclose(
        scheme.sigma.inverse(scheme.sigma(test_points)), test_points, rtol=1e-5
    )

  @parameterized.product(data_std=(1.0, 2.0))
  def test_create_ve(self, data_std):
    sigma = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma, data_std)
    test_points = np.random.default_rng(678).uniform(0.01, 1.0, size=(10,))
    np.testing.assert_allclose(scheme.scale(test_points), 1.0)
    # verify that variance is scaled by data_std
    np.testing.assert_allclose(
        scheme.sigma(test_points), sigma(test_points) * data_std, rtol=1e-5
    )
    # verify the inverse is correct
    np.testing.assert_allclose(
        scheme.sigma.inverse(scheme.sigma(test_points)), test_points, rtol=1e-5
    )


class NoiseLevelSamplingTest(parameterized.TestCase):

  @parameterized.parameters(((4,), True), ((2, 2), False))
  def test_uniform_samples(self, sample_shape, uniform_grid):
    samples = diffusion._uniform_samples(
        jax.random.PRNGKey(0), sample_shape, uniform_grid=uniform_grid
    )
    self.assertEqual(samples.shape, sample_shape)
    if uniform_grid:
      self.assertAlmostEqual(np.std(np.diff(np.sort(samples.flatten()))), 0.0)

  @parameterized.product(uniform_grid=(True, False))
  def test_log_uniform_sampling(self, uniform_grid):
    sample_shape = (25,)
    clip_min = 0.1
    scheme = absltest.mock.Mock(spec=diffusion.Diffusion)
    scheme.sigma_max = 100.0
    noise_sampling = diffusion.log_uniform_sampling(
        scheme, clip_min, uniform_grid
    )
    samples = noise_sampling(jax.random.PRNGKey(1), sample_shape)
    self.assertEqual(samples.shape, sample_shape)
    self.assertGreaterEqual(np.min(samples), clip_min)
    self.assertLessEqual(np.max(samples), scheme.sigma_max)
    if uniform_grid:
      self.assertAlmostEqual(
          np.std(np.diff(np.sort(np.log(samples)))), 0.0, places=5
      )

  @parameterized.product(uniform_grid=(True, False))
  def test_time_uniform(self, uniform_grid):
    sample_shape = (25,)
    clip_min = 0.1
    scheme = diffusion.Diffusion.create_variance_exploding(
        diffusion.tangent_noise_schedule()
    )
    noise_sampling = diffusion.time_uniform_sampling(
        scheme, clip_min, uniform_grid
    )
    samples = noise_sampling(jax.random.PRNGKey(0), sample_shape)
    self.assertEqual(samples.shape, sample_shape)
    self.assertGreaterEqual(np.min(samples), clip_min)
    self.assertLessEqual(np.max(samples), scheme.sigma_max)
    if uniform_grid:
      self.assertAlmostEqual(
          np.std(np.diff(np.sort(scheme.sigma.inverse(samples)))), 0.0, places=5
      )

  def test_edm_schedule(self):
    sample_shape = (20000,)
    p_mean, p_std = -1.2, 1.2
    scheme = absltest.mock.Mock(spec=diffusion.Diffusion)
    scheme.sigma_max = 100.0
    noise_sampling = diffusion.normal_sampling(
        scheme=scheme, p_mean=p_mean, p_std=p_std
    )
    samples = noise_sampling(jax.random.PRNGKey(1), sample_shape)
    self.assertEqual(samples.shape, sample_shape)
    self.assertAlmostEqual(np.mean(np.log(samples)), p_mean, places=2)
    self.assertAlmostEqual(np.std(np.log(samples)), p_std, places=2)


class NoiseLossWeightingTest(parameterized.TestCase):

  @parameterized.parameters(
      (4.0, 0.0625), (np.asarray([1.0, 0.5]), np.asarray([1.0, 4]))
  )
  def test_inverse_sigma_squared_schedule(self, sigma, expected_res):
    res = diffusion.inverse_squared_weighting(sigma)
    self.assertTrue(np.allclose(res, expected_res))

  @parameterized.parameters(
      (2.0, 4.0, 0.3125),
      (4.0, np.asarray([1.0, 8.0]), np.asarray([1.0625, 0.078125])),
  )
  def test_edm_weighting(self, sigma_data, sigma, expected_res):
    res = diffusion.edm_weighting(sigma_data)(sigma)
    self.assertTrue(np.allclose(res, expected_res))


if __name__ == "__main__":
  absltest.main()
