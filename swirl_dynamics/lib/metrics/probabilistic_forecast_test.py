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
from swirl_dynamics.lib.metrics import probabilistic_forecast


class ProbabilisticForecastMetricsTest(parameterized.TestCase):

  @parameterized.parameters(
      ((1, 4, 8, 8, 2), (1, 4, 8, 8, 2), 1),
      ((1, 4, 8, 8, 2), (1, 8, 8, 2), -1),
  )
  def test_raises_incompatible_forecasts_shapes(
      self, forecasts_shape, observations_shape, ensemble_axis
  ):
    with self.assertRaisesRegex(ValueError, "matching shapes"):
      probabilistic_forecast._process_forecasts(
          np.ones(forecasts_shape), np.ones(observations_shape), ensemble_axis  # pyrefly: ignore[bad-argument-type]
      )

  @parameterized.parameters(
      {"test_shape": (1, 8, 8, 2, 4)}, {"test_shape": (1, 8, 2, 4)}
  )
  def test_mad_with_broadcast_output_shape(self, test_shape):
    out = probabilistic_forecast._mean_abs_diff_with_broadcast(
        np.ones(test_shape)  # pyrefly: ignore[bad-argument-type]
    )
    self.assertEqual(out.shape, test_shape[:-1])

  @parameterized.parameters(
      {"test_shape": (1, 8, 8, 2, 4)}, {"test_shape": (1, 8, 2, 4)}
  )
  def test_mad_with_loop_output_shape(self, test_shape):
    out = probabilistic_forecast._mean_abs_diff_with_loop(jnp.ones(test_shape))
    self.assertEqual(out.shape, test_shape[:-1])

  @parameterized.parameters(
      {"test_shape": (1, 8, 8, 2, 4)}, {"test_shape": (1, 8, 2, 4)}
  )
  def test_mad_broadcast_loop_equivalence(self, test_shape):
    rng = np.random.default_rng(seed=404)
    forecasts = jnp.asarray(rng.normal(size=test_shape))
    out1 = probabilistic_forecast._mean_abs_diff_with_broadcast(forecasts)
    out2 = probabilistic_forecast._mean_abs_diff_with_loop(forecasts)
    np.testing.assert_allclose(out1, out2, atol=1e-5)

  @parameterized.parameters(
      {"test_shape": (1, 4, 8, 8, 2), "axis": 1},
      {"test_shape": (1, 8, 2, 4), "axis": -1},
  )
  def test_cprs_output_shape(self, test_shape, axis):
    rng = np.random.default_rng(seed=404)
    forecasts = rng.normal(size=test_shape)
    obs_shape = np.moveaxis(forecasts, axis, -1).shape[:-1]
    obs = rng.normal(size=obs_shape)
    out = probabilistic_forecast.crps(forecasts, obs, ensemble_axis=axis)  # pyrefly: ignore[bad-argument-type]
    self.assertEqual(out.shape, obs_shape)

  @parameterized.parameters(
      {"forecasts": [[1, 1, 1, 1]], "expected": 0},
      {"forecasts": [[0, 0, 0, 0]], "expected": 1},
      {"forecasts": [[0, 0, 2, 2]], "expected": 0.5},
  )
  def test_cprs_output_values(self, forecasts, expected):
    forecasts = jnp.asarray(forecasts)
    obs = jnp.asarray([1])
    out = probabilistic_forecast.crps(forecasts, obs, ensemble_axis=-1)
    self.assertEqual(out, expected)

  @parameterized.parameters(
      {"test_shape": (1, 8, 8, 2, 4), "threshold": np.array(0.5)},
      {"test_shape": (1, 8, 2, 4), "threshold": np.linspace(0, 1, 11)},
  )
  def test_tbs_output_shape(self, test_shape, threshold):
    rng = np.random.default_rng(seed=404)
    forecasts = rng.normal(size=test_shape)
    obs = rng.normal(size=test_shape[:-1])
    out = probabilistic_forecast.threshold_brier_score(
        forecasts, obs, threshold, ensemble_axis=-1  # pyrefly: ignore[bad-argument-type]
    )
    expected_shape = test_shape[:-1]
    if threshold.ndim > 0:
      expected_shape = expected_shape + threshold.shape
    self.assertEqual(out.shape, expected_shape)

  @parameterized.parameters(
      {"forecasts": [[1, 1, 1, 1]], "threshold": 0.5, "expected": 0},
      {"forecasts": [[0, 0, 0, 0]], "threshold": 0.5, "expected": 1},
      {"forecasts": [[0, 0, 2, 2]], "threshold": 0.5, "expected": 0.25},
      {
          "forecasts": [[1, 2, 3, 4]],
          "threshold": [2.5, 3.5],
          "expected": [[0.25, 0.0625]],
      },
  )
  def test_tbs_output_values(self, forecasts, threshold, expected):
    forecasts = jnp.asarray(forecasts)
    obs = jnp.asarray([1])
    out = probabilistic_forecast.threshold_brier_score(
        forecasts, obs, threshold, ensemble_axis=-1
    )
    np.testing.assert_allclose(out, np.asarray(expected), atol=1e-5)


class CrpsPrecisionTest(parameterized.TestCase):
  """CRPS arithmetic must not overflow before its floating-point reductions."""

  @parameterized.parameters(
      jnp.float16, jnp.bfloat16, jnp.int16, jnp.int32, jnp.uint16
  )
  def test_extreme_finite_members_match_double_precision_formula(self, dtype):
    if dtype == jnp.int32:
      values = [-(2**30), 2**30]
    elif dtype == jnp.int16:
      values = [-30000, 30000]
    elif dtype == jnp.uint16:
      values = [0, 60000]
    else:
      values = [-60000, 60000]
    forecasts = jnp.asarray([values, values[::-1]], dtype=dtype)
    observations = jnp.asarray([100, 1000], dtype=dtype)
    f, o = np.asarray(forecasts, np.float64), np.asarray(
        observations, np.float64
    )
    expected = np.abs(f - o[:, None]).mean(-1) - 0.5 * np.abs(
        f[:, :, None] - f[:, None, :]
    ).mean((-1, -2))
    for direct in (False, True):
      for compiled in (False, True):
        with self.subTest(direct=direct, compiled=compiled):

          def score(x, y):
            return probabilistic_forecast.crps(
                x, y, ensemble_axis=-1, direct_broadcast=direct
            )

          actual = (jax.jit(score) if compiled else score)(
              forecasts, observations
          )
          np.testing.assert_allclose(actual, expected, rtol=1e-6)
          self.assertEqual(actual.dtype, jnp.float32)
          self.assertTrue(np.all(np.isfinite(actual)))
          self.assertTrue(np.all(actual >= 0))
    np.testing.assert_array_equal(forecasts, f)
    np.testing.assert_array_equal(observations, o)

  @parameterized.parameters(0, 1, -1)
  def test_axis_permutation_mixed_inputs_and_single_members(self, axis):
    original = np.array(
        [[-30000, 10000, 30000], [10000, -30000, 30000]], np.int16
    )
    forecasts = np.moveaxis(original, -1, axis)
    observations = jnp.array([2.0, -4.0], dtype=jnp.float32)
    f = original.astype(np.float64)
    expected = np.abs(f - np.asarray(observations)[:, None]).mean(
        -1
    ) - 0.5 * np.abs(f[:, :, None] - f[:, None, :]).mean((-1, -2))
    for direct in (False, True):
      actual = probabilistic_forecast.crps(
          forecasts, observations, ensemble_axis=axis, direct_broadcast=direct
      )
      np.testing.assert_allclose(actual, expected, rtol=1e-6)
      singleton = jnp.array([[-60000.0], [60000.0]], dtype=jnp.float16)
      target = jnp.array([60000.0, -60000.0], dtype=jnp.float16)
      np.testing.assert_array_equal(
          probabilistic_forecast.crps(
              singleton, target, ensemble_axis=-1, direct_broadcast=direct
          ),
          [120000.0, 120000.0],
      )

  def test_float64_and_float32_preserve_precision_with_x64_enabled(self):
    with jax.enable_x64():
      for dtype in (jnp.float32, jnp.float64):
        with self.subTest(dtype=dtype):
          forecasts = jnp.array([[1.0, 3.0, 6.0]], dtype=dtype)
          obs = jnp.array([2.0], dtype=dtype)
          expected = (
              np.abs(np.asarray(forecasts) - 2).mean()
              - np.abs(
                  np.asarray(forecasts)[:, :, None]
                  - np.asarray(forecasts)[:, None, :]
              ).mean()
              / 2
          )
          for direct in (False, True):
            actual = probabilistic_forecast.crps(
                forecasts, obs, ensemble_axis=-1, direct_broadcast=direct
            )
            self.assertEqual(actual.dtype, dtype)
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

  @parameterized.parameters(False, True)
  def test_low_precision_gradients_match_floating_reference(self, direct):
    forecasts = jnp.array([[-60000.0, 10000.0, 60000.0]], dtype=jnp.float16)
    observations = jnp.array([1000.0], dtype=jnp.float16)

    def loss(values):
      return probabilistic_forecast.crps(
          values, observations, ensemble_axis=-1, direct_broadcast=direct
      ).sum()

    value, gradient = jax.jit(jax.value_and_grad(loss))(forecasts)
    ref_value, ref_gradient = jax.value_and_grad(loss)(
        forecasts.astype(jnp.float32)
    )
    np.testing.assert_allclose(value, ref_value, rtol=1e-6)
    np.testing.assert_allclose(gradient, ref_gradient, rtol=1e-3, atol=1e-4)
    self.assertTrue(np.all(np.isfinite(gradient)))


if __name__ == "__main__":
  absltest.main()
