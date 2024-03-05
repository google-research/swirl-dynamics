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
import jax
import numpy as np
from swirl_dynamics.lib.networks import unets


class NetworksTest(parameterized.TestCase):

  @parameterized.parameters(
      ((64,), "CIRCULAR", (2, 2, 2), False),
      ((64, 64), "CIRCULAR", (2, 2, 2), True),
      ((64, 64), "LATLON", (2, 2, 2), False),
      ((72, 144), "LATLON", (2, 2, 3), True),
  )
  def test_unet_output_shape(
      self, spatial_dims, padding, ds_ratio, use_pos_enc
  ):
    batch, channels = 2, 3
    x = np.random.randn(batch, *spatial_dims, channels)
    model = unets.UNet(
        out_channels=channels,
        num_channels=(4, 8, 12),
        downsample_ratio=ds_ratio,
        num_blocks=2,
        padding=padding,
        num_heads=4,
        use_position_encoding=use_pos_enc,
    )
    out, _ = model.init_with_output(
        jax.random.PRNGKey(42), x=x,  # is_training=True
    )
    self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
  absltest.main()
