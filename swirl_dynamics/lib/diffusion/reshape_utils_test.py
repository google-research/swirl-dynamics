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

"""Tests for the reshape utilities used in the ViViT-based models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from swirl_dynamics.lib.diffusion import reshape_utils


class ReshapeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('batch_axis', 0),
      ('channel_axis', 3),
      ('negative_axis', -1),
  )
  def test_inverse_2d_rejects_invalid_axis(self, axis):
    shape = (2, 3, 4, 2)
    valid_flattened = jnp.zeros((8, 3, 2))
    with self.assertRaisesRegex(ValueError, 'axis'):
      reshape_utils.reshape_to_2d_factorized(
          valid_flattened, axis=axis, two_d_shape=shape
      )

  @parameterized.named_parameters(
      ('batch_axis', 0),
      ('channel_axis', 4),
      ('negative_axis', -1),
  )
  def test_inverse_3d_rejects_invalid_axis(self, axis):
    shape = (2, 5, 3, 4, 2)
    valid_flattened = jnp.zeros((24, 5, 2))
    with self.assertRaisesRegex(ValueError, 'axis'):
      reshape_utils.reshape_to_3d_factorized(
          valid_flattened, axis=axis, three_d_shape=shape
      )

  @parameterized.parameters(1, 2)
  def test_inverse_2d_rejects_incorrect_token_channel_shape(self, axis):
    shape = (2, 3, 4, 2)
    expected = (8, 3, 2) if axis == 1 else (6, 4, 2)
    # Same total number of elements, but the token/channel axes are wrong.
    wrong = jnp.zeros((expected[0], 2 * expected[1], 1))
    with self.assertRaisesRegex(ValueError, 'shape'):
      reshape_utils.reshape_to_2d_factorized(
          wrong, axis=axis, two_d_shape=shape
      )

  @parameterized.parameters(1, 2, 3)
  def test_inverse_3d_rejects_incorrect_token_channel_shape(self, axis):
    shape = (2, 5, 3, 4, 2)
    expected = {1: (24, 5, 2), 2: (40, 3, 2), 3: (30, 4, 2)}[axis]
    wrong = jnp.zeros((expected[0], 2 * expected[1], 1))
    with self.assertRaisesRegex(ValueError, 'shape'):
      reshape_utils.reshape_to_3d_factorized(
          wrong, axis=axis, three_d_shape=shape
      )

  @parameterized.parameters(
      ((2, 4, 8, 3), 1),
      ((2, 8, 8, 3), 1),
      ((2, 8, 8, 3), 2),
  )
  def test_2d_to_1d_reshape_output_shape(self, shape, axis):
    batch_size, height, width, channel = shape

    x = jax.random.normal(jax.random.PRNGKey(0), shape)

    new_batch = (batch_size * width) if axis == 1 else (batch_size * height)
    new_token = height if axis == 1 else width

    out = reshape_utils.reshape_2d_to_1d_factorized(x, axis=axis)

    self.assertEqual(out.shape, (new_batch, new_token, channel))

  @parameterized.parameters(
      ((2, 4, 8, 2, 3), 1),
      ((2, 2, 8, 3, 1), 3),
      ((2, 8, 2, 8, 3), 2),
      ((2, 2, 8, 3, 1), 3),
      ((2, 8, 2, 8, 3), 1),
  )
  def test_3d_to_1d_reshape_output_shape(self, shape, axis):
    batch_size, time, height, width, channel = shape

    x = jax.random.normal(jax.random.PRNGKey(0), shape)

    new_batch = {
        '1': batch_size * height * width,
        '2': batch_size * time * width,
        '3': batch_size * time * height,
    }
    new_token = {'1': time, '2': height, '3': width}

    out = reshape_utils.reshape_3d_to_1d_factorized(x, axis=axis)

    self.assertEqual(
        out.shape, (new_batch[f'{axis}'], new_token[f'{axis}'], channel)
    )

  @parameterized.parameters(
      ((2, 4, 8, 3), 1),
      ((2, 2, 8, 3), 1),
      ((2, 8, 4, 3), 2),
  )
  def test_indentity_cycle_reshape_2d(self, shape, axis):

    x = jax.random.normal(jax.random.PRNGKey(0), shape)

    out_reshaped = reshape_utils.reshape_2d_to_1d_factorized(x, axis=axis)
    out_identity = reshape_utils.reshape_to_2d_factorized(
        out_reshaped, axis=axis, two_d_shape=shape
    )

    self.assertEqual(out_identity.tolist(), x.tolist())

  @parameterized.parameters(
      ((2, 4, 8, 2, 3), 1),
      ((2, 2, 8, 3, 1), 3),
      ((2, 8, 2, 8, 3), 2),
      ((2, 2, 8, 3, 1), 3),
      ((2, 8, 2, 8, 3), 1),
  )
  def test_indentity_cycle_reshape_3d(self, shape, axis):

    x = jax.random.normal(jax.random.PRNGKey(0), shape)

    out_reshaped = reshape_utils.reshape_3d_to_1d_factorized(x, axis=axis)
    out_identity = reshape_utils.reshape_to_3d_factorized(
        out_reshaped, axis=axis, three_d_shape=shape
    )

    self.assertEqual(out_identity.tolist(), x.tolist())


if __name__ == '__main__':
  absltest.main()
