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
from swirl_dynamics.lib.layers import upsample


class UpsampleLayersTest(parameterized.TestCase):

  @parameterized.parameters(
      ((8, 8, 8, 32), (8,), (8, 8, 64, 4)),
      ((8, 8, 8, 32), (4, 4), (8, 32, 32, 2)),
      ((8, 8, 8, 8, 128), (2, 2, 2, 2), (16, 16, 16, 16, 8)),
  )
  def test_channel_to_space_output_shape(
      self, input_shape, block_shape, expected_output_shape
  ):
    inputs = jnp.ones(input_shape)
    out = upsample.channel_to_space(inputs, block_shape)
    self.assertEqual(out.shape, expected_output_shape)


if __name__ == "__main__":
  absltest.main()
