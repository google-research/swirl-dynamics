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
import jax.numpy as jnp
from swirl_dynamics.lib.diffusion import unets3d


class Unets3dTest(parameterized.TestCase):

  @parameterized.parameters(
      (unets3d.UNet3d, (False, False, False), (False, False, False)),
      (
          unets3d.PreconditionedDenoiser3d,
          (False, False, True),
          (False, False, True),
      ),
  )
  def test_unets3d_output_shape(self, model_cls, spatial_attn, temporal_attn):
    x = jnp.ones((2, 5, 33, 16, 4))
    network = model_cls(
        out_channels=4,
        resize_to_shape=(32, 16),
        num_channels=(8, 16, 32),
        downsample_ratio=(2, 2, 2),
        num_blocks=2,
        noise_embed_dim=32,
        padding="LONLAT",
        use_spatial_attention=spatial_attn,
        use_temporal_attention=temporal_attn,
        use_position_encoding=True,
        num_heads=2,
        cond_resize_method="cubic",
        cond_embed_dim=32,
    )
    out, _ = network.init_with_output(
        jax.random.PRNGKey(0), x, sigma=jnp.ones((2,)), is_training=True
    )
    self.assertEqual(out.shape, x.shape)

  def test_unets3d_with_channelwise_cond_output_shape(self):
    x = jnp.ones((2, 5, 33, 16, 4))
    cond = {
        "channel:cond1": jax.random.normal(
            jax.random.PRNGKey(42), (2, 1, 16, 8, 3)
        )
    }
    network = unets3d.PreconditionedDenoiser3d(
        out_channels=4,
        resize_to_shape=(32, 16),
        num_channels=(8, 16, 32),
        downsample_ratio=(2, 2, 2),
        num_blocks=2,
        noise_embed_dim=32,
        padding="LONLAT",
        use_spatial_attention=(False, False, True),
        use_temporal_attention=(True, True, True),
        use_position_encoding=True,
        num_heads=2,
        cond_resize_method="cubic",
        cond_embed_dim=32,
    )
    out, _ = network.init_with_output(
        jax.random.PRNGKey(0),
        x,
        sigma=jnp.ones((2,)),
        cond=cond,
        is_training=True,
    )
    self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
  absltest.main()
