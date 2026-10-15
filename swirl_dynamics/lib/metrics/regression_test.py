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
from swirl_dynamics.lib.metrics import regression as reg

_TEST_INPUT_SHAPE = (1, 2, 3, 4)


class RegressionMetricsTest(parameterized.TestCase):

  @parameterized.parameters(
      (reg.mean_squared_error, (1,), (2,), (1, 4)),
      (reg.mean_squared_error, (), (2,), (1, 2, 4)),
      (reg.mean_squared_error, (3,), None, ()),
      (reg.mean_absolute_error, (1,), (2,), (1, 4)),
      (reg.mean_absolute_error, (), (2,), (1, 2, 4)),
      (reg.mean_absolute_error, (3,), None, ()),
  )
  def test_mse_and_mae_shapes(self, metric, sum_axes, mean_axes, output_shape):
    xpred = np.ones(_TEST_INPUT_SHAPE)
    xtrue = 2 * np.ones(_TEST_INPUT_SHAPE)
    err = metric(pred=xpred, true=xtrue, sum_axes=sum_axes, mean_axes=mean_axes)
    self.assertEqual(err.shape, output_shape)

  @parameterized.parameters(
      (False, True, 4.0),
      (False, False, 2.0),
      (True, True, 0.25),
      (True, False, 0.5),
  )
  def test_mse_values(self, relative, squared, expected):
    xpred = np.ones(_TEST_INPUT_SHAPE)
    xtrue = 2 * np.ones(_TEST_INPUT_SHAPE)
    err = reg.mean_squared_error(
        pred=xpred,
        true=xtrue,
        sum_axes=(-1,),
        relative=relative,
        squared=squared,
    )
    self.assertAlmostEqual(err, expected, places=5)

  @parameterized.parameters((False, 12.0), (True, 0.75))
  def test_mae_values(self, relative, expected):
    xpred = np.ones(_TEST_INPUT_SHAPE)
    xtrue = 4 * np.ones(_TEST_INPUT_SHAPE)
    err = reg.mean_absolute_error(
        pred=xpred, true=xtrue, sum_axes=(-1,), relative=relative
    )
    self.assertAlmostEqual(err, expected, places=5)

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
        reg.mean_absolute_error if absolute else reg.mean_squared_error
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
          reg.mean_squared_error(
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
      metric=(reg.mean_squared_error, reg.mean_absolute_error),
      axis=(3, -4),
  )
  def test_out_of_bounds_axes_are_not_wrapped(self, metric, axis):
    values = jnp.ones((2, 3, 4))
    with self.assertRaises((ValueError, IndexError)):
      metric(values, values, sum_axes=(2,), mean_axes=(axis,))

  @parameterized.parameters(reg.mean_squared_error, reg.mean_absolute_error)
  def test_scalar_without_reduction_axes(self, metric):
    actual = metric(
        jnp.array(3.0), jnp.array(1.0), mean_axes=(), relative=False
    )
    expected = 4.0 if metric is reg.mean_squared_error else 2.0
    np.testing.assert_array_equal(actual, expected)

  @parameterized.product(
      case=[
          (jnp.uint8, 0, 255),
          (jnp.int16, 32767, -32768),
          (jnp.int32, 100000, -100000),
          (jnp.float16, 300.0, -300.0),
          (jnp.float32, 3.0, -2.0),
      ],
      metric=["mse", "rmse", "mae"],
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
    if metric == "mae":
      fn = lambda x, y: reg.mean_absolute_error(
          x, y, sum_axes=(-1,), relative=relative
      )
      errors = np.abs(a - b).sum(axis=-1)
      if relative:
        errors /= np.abs(b).sum(axis=-1)
    else:
      fn = lambda x, y: reg.mean_squared_error(
          x, y, sum_axes=(-1,), relative=relative, squared=metric == "mse"
      )
      errors = ((a - b) ** 2).sum(axis=-1)
      if relative:
        errors /= (b**2).sum(axis=-1)
    expected = errors.mean()
    if metric == "rmse":
      expected = np.sqrt(expected)
    for evaluation in (fn, jax.jit(fn)):
      actual = evaluation(pred, true)
      self.assertTrue(bool(jnp.isfinite(actual)))
      np.testing.assert_allclose(actual, expected, rtol=2e-6)

  def test_float64_precision_is_retained_when_enabled(self):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
      p = jnp.array([1e10 + 1, 1e10 + 2], jnp.float64)
      t = jnp.array([1e10, 1e10], jnp.float64)
      actual = reg.mean_squared_error(p, t)
      self.assertEqual(actual.dtype, jnp.float64)
      self.assertEqual(float(actual), 2.5)
    finally:
      jax.config.update("jax_enable_x64", previous)

  def test_mixed_dtypes_are_promoted_before_subtraction(self):
    pred = jnp.array([0, 255], jnp.uint8)
    true = jnp.array([255, 0], jnp.int16)
    np.testing.assert_allclose(reg.mean_squared_error(pred, true), 65025.0)

  def test_dtype_argument_controls_promotion(self):
    p = jnp.array([1.0, 2.0], jnp.float16)
    t = jnp.array([0.0, 0.0], jnp.float16)
    self.assertEqual(
        reg.mean_squared_error(p, t, dtype=jnp.float16).dtype, jnp.float16
    )
    self.assertEqual(
        reg.mean_absolute_error(p, t, dtype=jnp.float16).dtype, jnp.float16
    )

  def test_float32_gradients_match_the_analytic_error(self):
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.zeros_like(x)
    for derivative in (
        jax.grad(lambda p: reg.mean_squared_error(p, target)),
        jax.jit(jax.grad(lambda p: reg.mean_squared_error(p, target))),
    ):
      np.testing.assert_allclose(derivative(x), 2 * x / x.size)


if __name__ == "__main__":
  absltest.main()
