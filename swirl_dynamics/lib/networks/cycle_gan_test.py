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

# import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from swirl_dynamics.lib.networks import cycle_gan


class NetworksTest(parameterized.TestCase):

  @parameterized.parameters(
      ((32, 16), "CIRCULAR", 1, 2, 0, "deconv", True),
      ((32, 32), "CIRCULAR", 2, 1, 1, "deconv", False),
      ((32, 32), "LATLON", 3, 2, 0, "bilinear", False),
      ((16, 32), "LATLON", 1, 1, 2, "bilinear", True),
  )
  def test_generator_output_shape(
      self,
      spatial_dims,
      padding,
      num_layers,
      channels,
      n_res_blocks_level,
      upsample_mode,
      use_skips,
  ):
    batch_size = 2

    x = np.random.randn(batch_size, *spatial_dims, channels)
    model = cycle_gan.Generator(
        output_nc=channels,
        ngf=2,
        upsample_mode=upsample_mode,
        n_res_blocks=1,
        padding=padding,
        n_downsample_layers=num_layers,
        n_upsample_layers=num_layers,
        use_attention=True,
        n_res_blocks_level=n_res_blocks_level,
        use_position_encoding=False,
        use_skips=use_skips,
    )
    out, variables = model.init_with_output(
        jax.random.PRNGKey(42), x, is_training=False
    )
    out_training = model.apply(
        variables, x, is_training=True, rngs={"dropout": jax.random.PRNGKey(42)}
    )
    self.assertEqual(out.shape, x.shape)
    self.assertEqual(out_training.shape, x.shape)

  @parameterized.parameters(
      ((16, 8), "CIRCULAR", 1, 2),
      ((16, 16), "CIRCULAR", 2, 2),
      ((32, 32), "LATLON", 2, 1),
      ((8, 16), "LATLON", 1, 1),
  )
  def test_discriminator_output_shape(
      self, spatial_dims, padding, num_layers, input_channels
  ):
    batch_size = 2

    new_spatial_dims = (spatial_dims[0]//(2**(num_layers+1)),
                        spatial_dims[1]//(2**(num_layers+1)))
    x = np.random.randn(batch_size, *spatial_dims, input_channels)

    model = cycle_gan.Discriminator(
        base_features=2,
        n_layers=num_layers,
        padding=padding,
        use_bias=False,
        use_local=False,
    )
    out, _ = model.init_with_output(
        jax.random.PRNGKey(42), x
    )

    self.assertEqual(out.shape, (batch_size, *new_spatial_dims, 1))

  @parameterized.parameters(
      ((16, 8), (16, 8), "CIRCULAR", 1, True),
      ((16, 16), (16, 16), "CIRCULAR", 2, False),
      ((32, 32), (32, 32), "LATLON", 2, True),
      ((8, 16), (8, 16), "LATLON", 1, False),
  )
  def test_filtered_interpolation_output_shape(
      self, input_dims, output_dims, padding, output_nc, use_local
  ):
    batch_size = 2

    x = np.random.randn(batch_size, *input_dims, output_nc)

    model = cycle_gan.FilteredInterpolation(
        height=output_dims[0],
        width=output_dims[1],
        output_nc=output_nc,
        padding=padding,
        use_local=use_local,
    )

    out, _ = model.init_with_output(
        jax.random.PRNGKey(42), x
    )

    self.assertEqual(out.shape, (batch_size, *output_dims, output_nc))


if __name__ == "__main__":
  absltest.main()
