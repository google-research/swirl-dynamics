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
from swirl_dynamics.lib import diffusion as dfn_pkg
from swirl_dynamics.lib.diffusion import diffusion

jax.config.update("jax_threefry_partitionable", False)


class DiffusionTest(parameterized.TestCase):

  @parameterized.parameters((50.0, 0.0, 1.5), (100.0, -1.5, 1.5))
  def test_tangent_noise_schedule(self, clip_max, start, end):
    sigma = diffusion.tangent_noise_schedule(clip_max, start, end)
    self.assertAlmostEqual(sigma(1.0), clip_max, places=3)
    self.assertAlmostEqual(sigma(0.0), 0, places=8)  # pyrefly: ignore[no-matching-overload]

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

  @parameterized.product(
      schedule=(
          diffusion.power_noise_schedule,
          diffusion.tangent_noise_schedule,
          diffusion.exponential_noise_schedule,
      ),
      constructor=(
          diffusion.Diffusion.create_variance_preserving,
          diffusion.Diffusion.create_variance_exploding,
      ),
      data_std=(1.0, 2.0),
  )
  def test_logsnr_property_matches_sigma2logsnr(
      self, schedule, constructor, data_std
  ):
    scheme = constructor(schedule(clip_max=4.0), data_std=data_std)
    times = jnp.array([[0.025, 0.125, 0.5], [0.7, 0.9, 1.0]])
    expected_logsnr = diffusion.sigma2logsnr(scheme.sigma)
    np.testing.assert_allclose(
        jax.jit(scheme.logsnr)(times),
        expected_logsnr(times),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        jax.jit(scheme.logsnr.inverse)(scheme.logsnr(times)),
        times,
        rtol=2e-5,
        atol=2e-6,
    )

  def test_zero_noise_limit_and_compiled_derivative(self):
    scheme = diffusion.Diffusion.create_variance_exploding(
        diffusion.power_noise_schedule(clip_max=2.0)
    )
    self.assertTrue(jnp.isposinf(scheme.logsnr(0.0)))
    self.assertEqual(float(scheme.logsnr.inverse(jnp.inf)), 0.0)
    times = jnp.array([0.125, 0.5, 1.0])
    derivative = jax.jit(jax.vmap(jax.grad(scheme.logsnr.forward)))(times)
    np.testing.assert_allclose(derivative, -2 / times, rtol=1e-6)


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
    noise_sampling = diffusion.log_normal_sampling(
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

  def test_min_snr_edm_weighting_recovers_edm_at_large_gamma(self):
    """With gamma → ∞, min_snr_edm_weighting should match edm_weighting."""
    sigma = np.asarray([0.01, 0.1, 1.0, 10.0, 80.0])
    data_std = 1.0
    edm = diffusion.edm_weighting(data_std)(sigma)
    min_snr = diffusion.min_snr_edm_weighting(data_std, gamma=1e12)(sigma)
    np.testing.assert_allclose(min_snr, edm, rtol=1e-5)

  def test_min_snr_edm_weighting_downweights_high_noise(self):
    """At high sigma (low SNR), Min-SNR should down-weight vs standard EDM."""
    data_std = 1.0
    gamma = 5.0
    sigma_high = np.asarray([10.0, 80.0])  # SNR << gamma
    edm = diffusion.edm_weighting(data_std)(sigma_high)
    min_snr = diffusion.min_snr_edm_weighting(data_std, gamma)(sigma_high)
    # min_snr_factor = min(SNR, gamma) / SNR = gamma * sigma^2 / data_std^2
    # For sigma=10: factor = 5 * 100 / 1 = 500 → capped at 1
    # Actually SNR(10) = 1/100 = 0.01 << gamma=5, so min_snr_factor = 1.
    # For low SNR, min(SNR, gamma) = SNR, so factor = 1 → equals EDM.
    np.testing.assert_allclose(min_snr, edm, rtol=1e-5)

  def test_min_snr_edm_weighting_clamps_low_noise(self):
    """At low sigma (high SNR >> gamma), Min-SNR factor < 1."""
    data_std = 1.0
    gamma = 5.0
    sigma_low = np.asarray([0.01])  # SNR = 1/0.0001 = 10000 >> gamma
    edm = diffusion.edm_weighting(data_std)(sigma_low)
    min_snr = diffusion.min_snr_edm_weighting(data_std, gamma)(sigma_low)
    expected_factor = gamma / (data_std**2 / sigma_low**2)  # gamma/SNR
    np.testing.assert_allclose(min_snr, edm * expected_factor, rtol=1e-5)

  def test_min_snr_edm_weighting_zero_and_extreme_sigma_grad(self):
    weight_fn = diffusion.min_snr_edm_weighting(data_std=1.0, gamma=5.0)
    sigma = jnp.asarray([0.0, 0.01, 1.0, 65504.0], dtype=jnp.float32)
    weights = jax.jit(weight_fn)(sigma)
    grads = jax.jit(jax.grad(lambda s: jnp.sum(weight_fn(s))))(sigma)
    self.assertTrue(bool(jnp.all(jnp.isfinite(weights))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(grads))))
    self.assertAlmostEqual(float(weights[0]), 5.0, places=5)

    with self.assertRaisesRegex(ValueError, "`data_std` must be positive"):
      diffusion.min_snr_edm_weighting(data_std=0.0)
    with self.assertRaisesRegex(ValueError, "`gamma` must be positive"):
      diffusion.min_snr_edm_weighting(data_std=1.0, gamma=0.0)

  def test_t_edm_weighting_validation_and_values(self):
    weight_fn = diffusion.t_edm_weighting(df=3, data_std=2.0)
    sigma = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    weights = weight_fn(sigma)
    expected = (3.0 * np.square(sigma) + 4.0) / (3.0 * 4.0 * np.square(sigma))
    np.testing.assert_allclose(weights, expected, rtol=1e-6)

    with self.assertRaisesRegex(ValueError, "`df` must be greater than 2"):
      diffusion.t_edm_weighting(df=2)

  def test_public_exports_importable(self):
    self.assertNotEmpty(dfn_pkg.__all__)
    self.assertEqual(dfn_pkg.__all__, sorted(set(dfn_pkg.__all__)))
    self.assertIn("InterlockingFrames", dfn_pkg.__all__)
    for name in dfn_pkg.__all__:
      with self.subTest(symbol=name):
        self.assertTrue(hasattr(dfn_pkg, name))


if __name__ == "__main__":
  absltest.main()
