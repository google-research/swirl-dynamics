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

"""Tests for batched and shared contrastive guidance conditions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import guidance

_STRENGTH = 0.75


def _denoise(x, sigma, cond):
  del sigma
  return 0.25 * x + cond['target'] + 2.0 * cond['context']


class ContrastiveConditionTest(parameterized.TestCase):

  @parameterized.product(
      condition_shape=((), (2,), (1, 2), (3, 2)),
      compiled=(False, True),
  )
  def test_condition_broadcasting(self, condition_shape, compiled):
    x = jnp.arange(6, dtype=jnp.float32).reshape((3, 2))
    target = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    context = jnp.array([[2.0, 1.0], [0.5, 1.5], [-1.0, 0.0]])
    cond = {'target': target, 'context': context}
    negatives = np.resize(np.array([-3.0, -1.0, -2.0]), condition_shape)
    negatives = jnp.asarray(negatives, dtype=jnp.float32)
    transform = guidance.ClassifierFreeContrastive(
        guidance_strength=_STRENGTH,
        cond_key='target',
        contrast_cond_key='negative',
    )
    fn = transform(_denoise, {'negative': negatives})
    actual = (jax.jit(fn) if compiled else fn)(x, jnp.array(0.1), cond)
    expected = (
        0.25 * np.asarray(x)
        + np.asarray(target)
        + 2 * np.asarray(context)
        + _STRENGTH
        * (np.asarray(target) - np.broadcast_to(negatives, target.shape))
    )
    self.assertEqual(actual.shape, x.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(False, True)
  def test_per_sample_negative_condition_gradients(self, compiled):
    x = jnp.zeros((3, 2), dtype=jnp.float32)
    cond = {
        'target': jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        'context': jnp.ones((3, 2)),
    }
    transform = guidance.ClassifierFreeContrastive(
        guidance_strength=_STRENGTH,
        cond_key='target',
        contrast_cond_key='negative',
    )

    def total_prediction(negatives):
      fn = transform(_denoise, {'negative': negatives})
      return jnp.sum(fn(x, jnp.array(0.1), cond))

    negatives = jnp.arange(6, dtype=jnp.float32).reshape((3, 2))
    grad_fn = jax.grad(total_prediction)
    actual = (jax.jit(grad_fn) if compiled else grad_fn)(negatives)
    np.testing.assert_allclose(actual, -_STRENGTH * np.ones((3, 2)))

  @parameterized.named_parameters(
      ('wrong_batch', (4, 2)),
      ('wrong_features', (3, 3)),
      ('extra_axis', (2, 1, 2)),
  )
  def test_incompatible_negative_shape_rejected(self, shape):
    transform = guidance.ClassifierFreeContrastive(
        guidance_strength=0.5, cond_key='target', contrast_cond_key='negative'
    )
    x = jnp.zeros((3, 2))
    cond = {'target': jnp.ones((3, 2)), 'context': jnp.ones((3, 2))}
    fn = transform(_denoise, {'negative': jnp.ones(shape)})
    with self.assertRaisesRegex(ValueError, 'cannot be broadcast'):
      fn(x, jnp.array(0.1), cond)


if __name__ == '__main__':
  absltest.main()
