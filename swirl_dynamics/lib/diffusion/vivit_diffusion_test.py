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

"""Tests for the light-weight ViViT Diffusion implementation."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import ml_collections
import numpy as np
from swirl_dynamics.lib.diffusion import vivit_diffusion


class VivitDiffusionTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          (16, 32, 32),
          (2, 2, 2),
          1,
          3,
          'factorized_self_attention_block',
          'time_space',
      ),
      (
          (16, 64, 64),
          (4, 4, 4),
          1,
          3,
          'factorized_self_attention_block',
          'space_time',
      ),
      (
          (32, 64, 64),
          (8, 8, 8),
          2,
          6,
          'factorized_3d_self_attention_block',
          'time_height_width',
      ),
      (
          (32, 32, 32),
          (4, 4, 4),
          2,
          6,
          'factorized_3d_self_attention_block',
          'height_width_time',
      ),
  )
  def test_vivit_diffusion_output_shape(
      self,
      spatial_dims,
      patch_size,
      output_features,
      channels,
      attention_type,
      attention_order,
  ):
    batch = 2

    x = np.random.randn(batch, *spatial_dims, channels)
    sigma = np.linspace(0, 1, batch)

    config_patches = ml_collections.ConfigDict()
    config_patches.size = patch_size

    temporal_encoding_config = ml_collections.ConfigDict()
    temporal_encoding_config.method = '3d_conv'
    temporal_encoding_config.kernel_init_method = 'central_frame_initializer'

    attention_config = ml_collections.ConfigDict()
    attention_config.type = attention_type
    attention_config.attention_order = attention_order
    attention_config.attention_kernel_init_method = 'xavier'

    vivit_model = vivit_diffusion.ViViTDiffusion(
        mlp_dim=4,
        num_layers=1,
        num_heads=1,
        output_features=output_features,
        patches=config_patches,
        hidden_size=6,
        temporal_encoding_config=temporal_encoding_config,
        attention_config=attention_config,
        positional_embedding='none'
    )

    out, _ = vivit_model.init_with_output(
        jax.random.PRNGKey(42),
        x=x,
        sigma=sigma,
        is_training=False,
    )
    self.assertEqual(out.shape, (batch, *spatial_dims, output_features))

  def test_vivit_with_channelwise_cond_output_shape(self):

    batch = 2
    channels = 4
    channels_cond = 3
    patch_size = (1, 2, 2)

    x = np.ones((batch, 2, 16, 16, channels))
    cond = {
        'channel:cond1': jax.random.normal(
            jax.random.PRNGKey(42), (batch, 2, 16, 8, channels_cond)
        )
    }

    sigma = np.linspace(0, 1, batch)

    config_patches = ml_collections.ConfigDict()
    config_patches.size = patch_size

    temporal_encoding_config = ml_collections.ConfigDict()
    temporal_encoding_config.method = '3d_conv'
    temporal_encoding_config.kernel_init_method = 'central_frame_initializer'

    attention_config = ml_collections.ConfigDict()
    attention_config.type = 'factorized_self_attention_block'
    attention_config.attention_order = 'time_space'
    attention_config.attention_kernel_init_method = 'xavier'

    vivit_model = vivit_diffusion.ViViTDiffusion(
        mlp_dim=4,
        num_layers=1,
        num_heads=1,
        output_features=channels,
        patches=config_patches,
        hidden_size=6,
        temporal_encoding_config=temporal_encoding_config,
        attention_config=attention_config,
    )
    out, _ = vivit_model.init_with_output(
        jax.random.PRNGKey(0),
        x,
        sigma=sigma,
        cond=cond,
        is_training=True,
    )
    self.assertEqual(out.shape, x.shape)

if __name__ == '__main__':
  absltest.main()
