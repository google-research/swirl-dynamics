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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import diffusion  # gpylint: disable=g-importing-member


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
    model = diffusion.unets.UNet(
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

  @parameterized.parameters(
      ((66,), (64,), "CIRCULAR", (2, 2, 2)),
      ((73, 140), (72, 144), "LATLON", (2, 2, 3)),
  )
  def test_unet_with_reshape(
      self, input_shape, resize_to_shape, padding, ds_ratio
  ):
    batch, channels = 2, 3
    x = np.random.randn(batch, *input_shape, channels)
    sigma = np.linspace(0, 1, batch)
    model = diffusion.unets.UNet(
        out_channels=channels,
        resize_to_shape=resize_to_shape,
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
    model = diffusion.unets.PreconditionedDenoiser(
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

  @parameterized.parameters(
      {"x_dims": (1, 16, 3), "c_dims": (1, 12, 5)},
      {"x_dims": (1, 16, 16, 3), "c_dims": (1, 32, 32, 6)},
      {"x_dims": (1, 16, 16, 3), "c_dims": (1, 4, 8, 6)},
  )
  def test_channelwise_conditioning_output_shape(self, x_dims, c_dims):
    x = jax.random.normal(jax.random.PRNGKey(42), x_dims)
    cond = {
        "channel:cond1": jax.random.normal(jax.random.PRNGKey(42), c_dims),
        "channel:cond2": jax.random.normal(jax.random.PRNGKey(42), x_dims),
    }
    sigma = jnp.array(0.5)
    model = diffusion.unets.PreconditionedDenoiser(
        out_channels=x_dims[-1],
        num_channels=(4, 8, 12),
        downsample_ratio=(2, 2, 2),
        num_blocks=2,
        num_heads=4,
        sigma_data=1.0,
        use_position_encoding=False,
        cond_embed_dim=128,
        cond_resize_method="cubic",
    )
    variables = model.init(
        jax.random.PRNGKey(42), x=x, sigma=sigma, cond=cond, is_training=True
    )
    # Check shape dict so that err message is easier to read when things break.
    shape_dict = jax.tree.map(jnp.shape, variables["params"])
    self.assertIn("InterpConvMerge_0", shape_dict)
    # First condition should be reshaped. Second one (correct shape) shoud not.
    self.assertIn("resize_channel:cond1", shape_dict["InterpConvMerge_0"])
    self.assertNotIn("resize_channel:cond2", shape_dict["InterpConvMerge_0"])
    # But they should both pass through the embedding convolution
    self.assertIn("conv2d_embed_channel:cond1", shape_dict["InterpConvMerge_0"])
    self.assertIn("conv2d_embed_channel:cond2", shape_dict["InterpConvMerge_0"])

    out = jax.jit(functools.partial(model.apply, is_training=True))(
        variables, x, sigma, cond
    )
    self.assertEqual(out.shape, x.shape)

  def test_preconditioned_merging_functions(self):
    x_dims = (1, 16, 8, 3)
    c_dims = (1, 8, 4, 6)
    x = jax.random.normal(jax.random.PRNGKey(42), x_dims)
    cond = {
        "channel:cond1": jax.random.normal(jax.random.PRNGKey(42), c_dims),
    }
    sigma = jnp.array(0.5)
    model = diffusion.unets.PreconditionedDenoiser(
        out_channels=x_dims[-1],
        num_channels=(4, 8, 12),
        downsample_ratio=(2, 2, 2),
        num_blocks=2,
        num_heads=4,
        sigma_data=1.0,
        use_position_encoding=False,
        cond_embed_dim=32,
        cond_resize_method="cubic",
        cond_merging_fn=diffusion.unets.AxialMLPInterpConvMerge,
    )
    variables = model.init(
        jax.random.PRNGKey(42), x=x, sigma=sigma, cond=cond, is_training=True
    )
    # Check shape dict so that err message is easier to read when things break.
    shape_dict = jax.tree.map(jnp.shape, variables["params"])
    self.assertIn("AxialMLPInterpConvMerge_0", shape_dict)
    self.assertIn("Axial2DMLP_0", shape_dict["AxialMLPInterpConvMerge_0"])
    self.assertIn("InterpConvMerge_0", shape_dict["AxialMLPInterpConvMerge_0"])

    out = jax.jit(functools.partial(model.apply, is_training=True))(
        variables, x, sigma, cond
    )
    self.assertEqual(out.shape, x.shape)

  def test_preconditioned_embedding_functions(self):
    x_dims = (1, 16, 8, 3)
    c_dims = (1, 8, 4, 6)
    x = jax.random.normal(jax.random.PRNGKey(42), x_dims)
    cond = {
        "channel:cond1": jax.random.normal(jax.random.PRNGKey(42), c_dims),
    }
    sigma = jnp.array(0.5)
    model = diffusion.unets.PreconditionedDenoiser(
        out_channels=x_dims[-1],
        num_channels=(4, 8, 12),
        downsample_ratio=(2, 2, 2),
        num_blocks=2,
        num_heads=4,
        sigma_data=1.0,
        use_position_encoding=False,
        cond_embed_dim=32,
        cond_resize_method="cubic",
        cond_embed_fn=diffusion.unets.EmbConvMerge,
    )
    variables = model.init(
        jax.random.PRNGKey(42), x=x, sigma=sigma, cond=cond, is_training=True
    )
    # Check shape dict so that err message is easier to read when things break.
    shape_dict = jax.tree.map(jnp.shape, variables["params"])
    self.assertIn("EmbConvMerge_0", shape_dict)
    self.assertIn(
        "level_0.embedding_downsample_conv", shape_dict["EmbConvMerge_0"]
    )
    self.assertIn(
        "cond_embedding.attention_block", shape_dict["EmbConvMerge_0"]
    )
    self.assertIn(
        "resize_embedding_channel:cond1", shape_dict["EmbConvMerge_0"]
    )

    out = jax.jit(functools.partial(model.apply, is_training=True))(
        variables, x, sigma, cond
    )
    self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
  absltest.main()
