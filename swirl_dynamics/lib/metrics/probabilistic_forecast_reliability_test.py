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

"""Tests for rank-histogram reliability with tied forecast values."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.metrics import probabilistic_forecast


class RankReliabilityTest(parameterized.TestCase):

  @parameterized.product(num_members=(1, 2, 5), jit=(False, True))
  def test_all_tied_members_are_uniform_over_eligible_ranks(
      self, num_members, jit
  ):
    forecasts = jnp.ones((12, num_members), dtype=jnp.float32)
    observations = jnp.ones((12,), dtype=jnp.float32)

    def score(f, o):
      return probabilistic_forecast.unreliability_score(f, o, ensemble_axis=-1)

    actual = (jax.jit(score) if jit else score)(forecasts, observations)
    np.testing.assert_allclose(actual, 0.0, atol=1e-6)

  @parameterized.parameters(-1, 0)
  def test_partial_ties_split_fractional_rank_counts(self, axis):
    forecasts = jnp.array([[0.0, 1.0], [1.0, 2.0]])
    observations = jnp.array([1.0, 1.0])
    if axis == 0:
      forecasts = jnp.asarray(np.moveaxis(np.asarray(forecasts), -1, axis))
    actual = probabilistic_forecast.unreliability_score(
        forecasts, observations, ensemble_axis=axis
    )
    # Each observation has two equally eligible rank bins: (1, 2), (0, 1).
    # Rank counts are (0.5, 1.0, 0.5), yielding unreliability 1/8.
    np.testing.assert_allclose(actual, 0.125, rtol=1e-6)

  def test_no_ties_preserve_existing_rank_score(self):
    forecasts = jnp.array([[-1.0, 1.0], [0.0, 2.0]])
    observations = jnp.array([0.0, 1.0])
    actual = probabilistic_forecast.unreliability_score(
        forecasts, observations, ensemble_axis=-1
    )
    # Both observations fall in rank bin 1, producing a score of 2.
    np.testing.assert_allclose(actual, 2.0, rtol=1e-6)

  def test_tie_handling_with_nontrailing_ensemble_axis(self):
    forecasts = jnp.ones((2, 3, 4))
    observations = jnp.ones((2, 4))
    actual = jax.jit(
        lambda f, o: probabilistic_forecast.unreliability_score(
            f, o, ensemble_axis=1
        )
    )(forecasts, observations)
    np.testing.assert_allclose(actual, 0.0, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
