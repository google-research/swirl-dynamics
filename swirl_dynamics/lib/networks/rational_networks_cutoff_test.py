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

"""Regression tests for rational denominator cutoff on batched inputs."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.networks import rational_networks

_CUTOFF = 0.25


def _unit_numerator_variables(num_features, unshared):
  # p(x) = 1 and q(x) = x, with exactly representable coefficients.
  numerator = jnp.array([0.0, 0.0, 0.0, 1.0])
  denominator = jnp.array([0.0, 1.0, 0.0])
  if unshared:
    numerator = jnp.broadcast_to(numerator, (num_features, 4))
    denominator = jnp.broadcast_to(denominator, (num_features, 3))
    return {'params': {'p_params': numerator, 'q_params': denominator}}
  return {'params': {'p_coeffs': numerator, 'q_coeffs': denominator}}


def _reference_cutoff(x, cutoff):
  return np.where(np.abs(x) < cutoff, np.where(x >= 0, cutoff, -cutoff), x)


class RationalCutoffTest(parameterized.TestCase):

  @parameterized.product(
      unshared=(False, True),
      input_shape=((2, 6), (2, 3, 6)),
      compiled=(False, True),
  )
  def test_cutoff_elementwise_on_batched_inputs(
      self, unshared, input_shape, compiled
  ):
    values = np.resize(
        np.array([-0.75, -0.125, 0.0, 0.125, 0.25, 0.75]),
        input_shape,
    ).astype(np.float32)
    layer_cls = (
        rational_networks.UnsharedRationalLayer
        if unshared
        else rational_networks.RationalLayer
    )
    module = layer_cls(cutoff=_CUTOFF)
    variables = _unit_numerator_variables(values.shape[-1], unshared)

    def fn(x):
      return module.apply(variables, x)

    actual = (jax.jit(fn) if compiled else fn)(jnp.asarray(values))
    expected = 1.0 / _reference_cutoff(values, _CUTOFF)
    self.assertEqual(actual.shape, values.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    self.assertTrue(np.all(np.isfinite(actual)))

  @parameterized.product(unshared=(False, True), compiled=(False, True))
  def test_batched_cutoff_gradients(self, unshared, compiled):
    values = jnp.array([[-0.5, -0.125, 0.0, 0.125, 0.5, 0.75]])
    layer_cls = (
        rational_networks.UnsharedRationalLayer
        if unshared
        else rational_networks.RationalLayer
    )
    layer = layer_cls(cutoff=_CUTOFF)
    variables = _unit_numerator_variables(values.shape[-1], unshared)

    def fn(x):
      return jnp.sum(layer.apply(variables, x))

    actual = (jax.jit(jax.grad(fn)) if compiled else jax.grad(fn))(values)
    expected = np.array([[-4, 0, 0, 0, -4, -16 / 9]])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(
      ('scalar', ()), ('vector', (6,)), ('tensor', (2, 3, 6))
  )
  def test_zero_cutoff_supports_scalar_and_arrays(self, shape):
    x = np.resize(np.array([-0.75, -0.1, 0.0, 0.1, 0.25, 0.75]), shape)
    actual = rational_networks.zero_cutoff(jnp.asarray(x), _CUTOFF)
    np.testing.assert_allclose(actual, _reference_cutoff(x, _CUTOFF))

  @parameterized.parameters(False, True)
  def test_default_cutoff_disabled(self, unshared):
    values = jnp.ones((2, 3, 4))
    cls = (
        rational_networks.UnsharedRationalLayer
        if unshared
        else rational_networks.RationalLayer
    )
    layer = cls()
    result = layer.apply(layer.init(jax.random.PRNGKey(1), values), values)
    self.assertEqual(result.shape, values.shape)
    self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
  absltest.main()
