# Copyright 2023 The swirl_dynamics Authors.
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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from swirl_dynamics.lib.diffusion import unets


class NetworksTest(parameterized.TestCase):

  @parameterized.parameters(
      ((64,), "CIRCULAR", (2, 2, 2)),
      ((64, 64), "CIRCULAR", (2, 2, 2)),
      ((64, 64), "LATLON", (2, 2, 2)),
      ((72, 144), "LATLON", (2, 2, 3)),
  )
  def test_unet_output_shape(self, spatial_dims, padding, ds_ratio):
    batch, channels = 2, 3
    x = np.random.randn(batch, *spatial_dims, channels)
    sigma = np.linspace(0, 1, batch)
    model = unets.UNet(
        out_channels=channels,
        num_channels=(4, 8, 12),
        downsample_ratio=ds_ratio,
        num_blocks=2,
        padding=padding,
        num_heads=4,
        use_position_encoding=False,
    )
    out, _ = model.init_with_output(
        jax.random.PRNGKey(42), x=x, sigma=sigma, is_training=True
    )
    self.assertEqual(out.shape, x.shape)

  @parameterized.parameters(((64,),), ((64, 64),))
  def test_preconditioned_denoiser_output_shape(self, spatial_dims):
    batch, channels = 2, 3
    x = np.random.randn(batch, *spatial_dims, channels)
    sigma = np.linspace(0, 1, batch)
    model = unets.PreconditionedDenoiser(
        out_channels=channels,
        num_channels=(4, 8, 12),
        downsample_ratio=(2, 2, 2),
        num_blocks=2,
        num_heads=4,
        sigma_data=1.0,
        use_position_encoding=False,
    )
    variables = model.init(
        jax.random.PRNGKey(42), x=x, sigma=sigma, is_training=True
    )
    out = jax.jit(functools.partial(model.apply, is_training=True))(
        variables, x, sigma
    )
    self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
  absltest.main()
