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

"""Regression metrics must not wrap before averaging integer errors."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.metrics import regression


class RegressionPrecisionTest(parameterized.TestCase):

  @parameterized.product(
      case=[
          (jnp.uint8, 0, 255),
          (jnp.int16, 32767, -32768),
          (jnp.int32, 100000, -100000),
          (jnp.float16, 300.0, -300.0),
          (jnp.float32, 3.0, -2.0),
      ],
      metric=['mse', 'rmse', 'mae'],
      relative=[False, True],
  )
  def test_values_match_independent_float64_arithmetic(
      self, case, metric, relative
  ):
    dtype, p, t = case
    pred = jnp.full((2, 3), p, dtype)
    true = jnp.full((2, 3), t, dtype)
    a, b = np.asarray(pred, dtype=np.float64), np.asarray(
        true, dtype=np.float64
    )
    if metric == 'mae':
      fn = lambda x, y: regression.mean_absolute_error(
          x, y, sum_axes=(-1,), relative=relative
      )
      errors = np.abs(a - b).sum(axis=-1)
      if relative:
        errors /= np.abs(b).sum(axis=-1)
    else:
      fn = lambda x, y: regression.mean_squared_error(
          x, y, sum_axes=(-1,), relative=relative, squared=metric == 'mse'
      )
      errors = ((a - b) ** 2).sum(axis=-1)
      if relative:
        errors /= (b**2).sum(axis=-1)
    expected = errors.mean()
    if metric == 'rmse':
      expected = np.sqrt(expected)
    for evaluation in (fn, jax.jit(fn)):
      actual = evaluation(pred, true)
      self.assertTrue(bool(jnp.isfinite(actual)))
      np.testing.assert_allclose(actual, expected, rtol=2e-6)

  def test_float64_precision_is_retained_when_enabled(self):
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    try:
      p = jnp.array([1e10 + 1, 1e10 + 2], jnp.float64)
      t = jnp.array([1e10, 1e10], jnp.float64)
      actual = regression.mean_squared_error(p, t)
      self.assertEqual(actual.dtype, jnp.float64)
      self.assertEqual(float(actual), 2.5)
    finally:
      jax.config.update('jax_enable_x64', previous)

  def test_mixed_dtypes_are_promoted_before_subtraction(self):
    pred = jnp.array([0, 255], jnp.uint8)
    true = jnp.array([255, 0], jnp.int16)
    np.testing.assert_allclose(
        regression.mean_squared_error(pred, true), 65025.0
    )

  def test_float32_gradients_match_the_analytic_error(self):
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.zeros_like(x)
    for derivative in (
        jax.grad(lambda p: regression.mean_squared_error(p, target)),
        jax.jit(jax.grad(lambda p: regression.mean_squared_error(p, target))),
    ):
      np.testing.assert_allclose(derivative(x), 2 * x / x.size)


if __name__ == '__main__':
  absltest.main()
