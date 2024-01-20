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
          np.ones(forecasts_shape), np.ones(observations_shape), ensemble_axis
      )

  @parameterized.parameters(
      {"test_shape": (1, 8, 8, 2, 4)}, {"test_shape": (1, 8, 2, 4)}
  )
  def test_mad_with_broadcast_output_shape(self, test_shape):
    out = probabilistic_forecast._mean_abs_diff_with_broadcast(
        np.ones(test_shape)
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
    out = probabilistic_forecast.crps(forecasts, obs, ensemble_axis=axis)
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
        forecasts, obs, threshold, ensemble_axis=-1
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


if __name__ == "__main__":
  absltest.main()
