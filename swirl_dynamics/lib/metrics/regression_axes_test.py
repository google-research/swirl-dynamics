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

"""Regression metrics honor overlapping sum and mean axes."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.metrics import regression


class RegressionAxesTest(parameterized.TestCase):

  @parameterized.product(
      absolute=(False, True),
      relative=(False, True),
      axes=(
          ((2,), (0, 2)),
          ((-1,), (0, 2)),
          ((2,), (0, -1)),
          ((1, -1), (0, 1, 2)),
      ),
  )
  def test_overlapping_axes_match_sum_first_reference(
      self, absolute, relative, axes
  ):
    sum_axes, mean_axes = axes
    truth = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10 + 1
    prediction = truth + np.linspace(0.1, 2.4, 24).astype(np.float32).reshape(
        truth.shape
    )
    error = np.abs(prediction - truth)
    normalizer = np.abs(truth)
    if not absolute:
      error = error**2
      normalizer = normalizer**2
    expected = error.sum(axis=sum_axes, keepdims=True)
    if relative:
      expected = expected / normalizer.sum(axis=sum_axes, keepdims=True)
    reduced_axes = tuple(sorted({a % truth.ndim for a in sum_axes + mean_axes}))
    expected = expected.mean(axis=reduced_axes)
    metric = (
        regression.mean_absolute_error
        if absolute
        else regression.mean_squared_error
    )

    def compute(values):
      return metric(
          values,
          jnp.asarray(truth),
          sum_axes=sum_axes,
          mean_axes=mean_axes,
          relative=relative,
      )

    for fn in (compute, jax.jit(compute)):
      actual = fn(jnp.asarray(prediction))
      np.testing.assert_allclose(actual, expected, rtol=1e-6)
      self.assertEqual(actual.shape, expected.shape)

  @parameterized.parameters(False, True)
  def test_rmse_and_gradients_match_disjoint_axes(self, relative):
    truth = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 1
    prediction = truth + 0.5

    def loss(values, axes):
      return jnp.sum(
          regression.mean_squared_error(
              values,
              truth,
              sum_axes=(-1,),
              mean_axes=axes,
              relative=relative,
              squared=False,
          )
      )

    reference = jax.value_and_grad(lambda x: loss(x, (0,)))
    overlapping = jax.value_and_grad(lambda x: loss(x, (0, 2)))
    expected = reference(prediction)
    for fn in (overlapping, jax.jit(overlapping)):
      for actual, wanted in zip(fn(prediction), expected):
        np.testing.assert_allclose(actual, wanted, rtol=1e-6)

  @parameterized.product(
      metric=(regression.mean_squared_error, regression.mean_absolute_error),
      axis=(3, -4),
  )
  def test_out_of_bounds_axes_are_not_wrapped(self, metric, axis):
    values = jnp.ones((2, 3, 4))
    with self.assertRaises((ValueError, IndexError)):
      metric(values, values, sum_axes=(2,), mean_axes=(axis,))

  @parameterized.parameters(
      regression.mean_squared_error, regression.mean_absolute_error
  )
  def test_scalar_without_reduction_axes(self, metric):
    actual = metric(
        jnp.array(3.0), jnp.array(1.0), mean_axes=(), relative=False
    )
    expected = 4.0 if metric is regression.mean_squared_error else 2.0
    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()
