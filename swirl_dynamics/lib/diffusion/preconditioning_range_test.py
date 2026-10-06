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

"""Preconditioning should not overflow while its coefficients are representable."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import preconditioning


def reference(sigma, sigma_data, shape):
  sigma = np.broadcast_to(np.asarray(sigma, np.float64), (shape[0],))
  scale = np.hypot(sigma, sigma_data)
  expanded = (shape[0],) + (1,) * (len(shape) - 1)
  return (
      (1 / scale).reshape(expanded),
      (sigma_data * (sigma / scale)).reshape(expanded),
      (sigma_data / scale).reshape(expanded) ** 2,
      0.25 * np.log(sigma),
  )


class OnesNetwork(nn.Module):

  def __call__(self, x, sigma, cond, *, is_training):
    del sigma, cond, is_training
    return jnp.ones_like(x)


class DecoratedNetwork(nn.Module):
  sigma_data: float

  @preconditioning.precondition
  def __call__(self, x, sigma, cond, *, is_training):
    del sigma, cond, is_training
    return jnp.ones_like(x)


class PreconditioningRangeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('half_large_sigma', jnp.float16, [80.0, 300.0, 512.0], 0.5),
      ('half_large_data', jnp.float16, [0.5, 80.0, 300.0], 300.0),
      ('float_large_sigma', jnp.float32, [1e10, 1e19, 1e20], 0.5),
      ('float_large_data', jnp.float32, [1.0, 1e10, 1e20], 1e20),
      ('ordinary', jnp.float32, [0.01, 1.0, 80.0], 0.5),
  )
  def test_coefficients_match_float64_reference(self, dtype, sigma, sigma_data):
    sigma = jnp.asarray(sigma, dtype=dtype)
    shape = (3, 2, 4)
    expected = reference(sigma, sigma_data, shape)
    fn = lambda s: preconditioning.compute_denoising_preconditioners(
        s, sigma_data, shape
    )
    for evaluate in (fn, jax.jit(fn)):
      for actual, wanted in zip(evaluate(sigma), expected):
        self.assertTrue(bool(jnp.isfinite(actual).all()))
        self.assertEqual(actual.dtype, dtype)
        tolerance = 3e-3 if dtype == jnp.float16 else 3e-6
        absolute = (
            np.finfo(np.float16).smallest_subnormal
            if dtype == jnp.float16
            else np.finfo(np.float32).tiny
        )
        np.testing.assert_allclose(
            actual, wanted, rtol=tolerance, atol=absolute
        )

  def test_scalar_sigma_still_broadcasts_across_the_batch(self):
    sigma = jnp.array(512.0, dtype=jnp.float16)
    for actual, wanted in zip(
        preconditioning.compute_denoising_preconditioners(sigma, 100.0, (3, 2)),
        reference(sigma, 100.0, (3, 2)),
    ):
      np.testing.assert_allclose(actual, wanted, rtol=3e-3, atol=1e-7)

  @parameterized.parameters('wrapper', 'decorator')
  def test_real_flax_preconditioners_retain_large_noise_network_output(
      self, kind
  ):
    model = (
        preconditioning.Preconditioned(OnesNetwork(), sigma_data=100.0)
        if kind == 'wrapper'
        else DecoratedNetwork(sigma_data=100.0)
    )
    x = jnp.arange(6, dtype=jnp.float16).reshape(2, 3)
    sigma = jnp.array([300.0, 512.0], dtype=jnp.float16)
    variables = model.init(jax.random.key(0), x, sigma, None, is_training=False)
    _, c_out, c_skip, _ = reference(sigma, 100.0, x.shape)
    expected = c_skip * np.asarray(x) + c_out
    evaluate = lambda x, s: model.apply(
        variables, x, s, None, is_training=False
    )
    for fn in (evaluate, jax.jit(evaluate)):
      actual = fn(x, sigma)
      np.testing.assert_allclose(actual, expected, rtol=3e-3)
      self.assertEqual(actual.dtype, x.dtype)
    derivative = jax.jit(jax.grad(lambda s: evaluate(x, s).sum()))(sigma)
    self.assertTrue(bool(jnp.isfinite(derivative).all()))

  def test_normal_range_derivatives_match_the_original_formula(self):
    sigma = jnp.array([0.01, 0.3, 1.0, 80.0])

    def original(s):
      variance = 0.5**2 + s**2
      return jnp.sum(
          1 / jnp.sqrt(variance)
          + 0.5 * s / jnp.sqrt(variance)
          + 0.5**2 / variance
          + 0.25 * jnp.log(s)
      )

    def actual(s):
      return sum(
          jnp.sum(v)
          for v in preconditioning.compute_denoising_preconditioners(
              s, 0.5, (4, 1)
          )
      )

    for fn in (jax.value_and_grad(actual), jax.jit(jax.value_and_grad(actual))):
      for value, wanted in zip(fn(sigma), jax.value_and_grad(original)(sigma)):
        np.testing.assert_allclose(value, wanted, rtol=1e-5, atol=1e-6)

  def test_batch_shape_validation_is_preserved(self):
    for sigma in [jnp.ones((2, 1)), jnp.ones(3)]:
      with self.assertRaisesRegex(ValueError, 'sigma must be 1D'):
        preconditioning.compute_denoising_preconditioners(sigma, 0.5, (2, 3))


if __name__ == '__main__':
  absltest.main()
