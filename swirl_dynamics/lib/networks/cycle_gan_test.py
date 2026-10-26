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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.networks import cycle_gan

# Golden parameter paths of a fixed 2-D generator captured before the networks
# were generalised to N spatial dimensions (see `test_two_d_generator_parity`).
_GOLDEN_2D_PARAM_PATHS = (
    "params/ConvTranspose_0/bias",
    "params/ConvTranspose_0/kernel",
    "params/ConvTranspose_1/bias",
    "params/ConvTranspose_1/kernel",
    "params/Conv_0/bias",
    "params/Conv_0/kernel",
    "params/Conv_1/bias",
    "params/Conv_1/kernel",
    "params/Conv_2/bias",
    "params/Conv_2/kernel",
    "params/Conv_3/bias",
    "params/Conv_3/kernel",
    "params/Conv_4/bias",
    "params/Conv_4/kernel",
    "params/Conv_5/bias",
    "params/Conv_5/kernel",
    "params/GroupNorm_0/bias",
    "params/GroupNorm_0/scale",
    "params/GroupNorm_1/bias",
    "params/GroupNorm_1/scale",
    "params/GroupNorm_2/bias",
    "params/GroupNorm_2/scale",
    "params/GroupNorm_3/bias",
    "params/GroupNorm_3/scale",
    "params/GroupNorm_4/bias",
    "params/GroupNorm_4/scale",
    "params/GroupNorm_5/bias",
    "params/GroupNorm_5/scale",
    "params/GroupNorm_6/bias",
    "params/GroupNorm_6/scale",
    "params/resnet_block_core_number_0/Conv_0/bias",
    "params/resnet_block_core_number_0/Conv_0/kernel",
    "params/resnet_block_core_number_0/Conv_1/bias",
    "params/resnet_block_core_number_0/Conv_1/kernel",
    "params/resnet_block_core_number_0/GroupNorm_0/bias",
    "params/resnet_block_core_number_0/GroupNorm_0/scale",
    "params/resnet_block_core_number_0/GroupNorm_1/bias",
    "params/resnet_block_core_number_0/GroupNorm_1/scale",
)


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
    self.assertEqual(out_training.shape, x.shape)  # pyrefly: ignore[missing-attribute]

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

    new_spatial_dims = (
        spatial_dims[0] // (2 ** (num_layers + 1)),
        spatial_dims[1] // (2 ** (num_layers + 1)),
    )
    x = np.random.randn(batch_size, *spatial_dims, input_channels)

    model = cycle_gan.Discriminator(
        base_features=2,
        n_layers=num_layers,
        padding=padding,
        use_bias=False,
        use_local=False,
    )
    out, _ = model.init_with_output(jax.random.PRNGKey(42), x)

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

    out, _ = model.init_with_output(jax.random.PRNGKey(42), x)

    self.assertEqual(out.shape, (batch_size, *output_dims, output_nc))

  def test_two_d_generator_parity(self):
    # Guards the N-D generalisation: a fixed 2-D generator must keep its
    # parameter tree and match between 2-tuple and scalar kernel specs, as well
    # as between `target_shape` and legacy `(height, width)`.
    model = cycle_gan.Generator(ngf=2, n_res_blocks=1)
    scalar_model = cycle_gan.Generator(
        ngf=2,
        n_res_blocks=1,
        kernel_size_downsampling=3,
        kernel_size_upsampling=3,
        kernel_size_core=3,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32, 1))

    out, variables = model.init_with_output(
        jax.random.PRNGKey(42), x, is_training=False
    )
    out_scalar, scalar_variables = scalar_model.init_with_output(
        jax.random.PRNGKey(42), x, is_training=False
    )
    param_paths = tuple(
        "/".join(str(getattr(k, "key", k)) for k in path)
        for path, _ in jax.tree_util.tree_flatten_with_path(variables)[0]
    )
    scalar_param_paths = tuple(
        "/".join(str(getattr(k, "key", k)) for k in path)
        for path, _ in jax.tree_util.tree_flatten_with_path(scalar_variables)[0]
    )

    self.assertEqual(param_paths, _GOLDEN_2D_PARAM_PATHS)
    self.assertEqual(scalar_param_paths, _GOLDEN_2D_PARAM_PATHS)
    np.testing.assert_allclose(out, out_scalar, rtol=1e-6, atol=1e-6)

    interp_target, _ = cycle_gan.FilteredInterpolation(
        target_shape=(16, 16), output_nc=1
    ).init_with_output(jax.random.PRNGKey(42), x)
    interp_hw, _ = cycle_gan.FilteredInterpolation(
        height=16, width=16, output_nc=1
    ).init_with_output(jax.random.PRNGKey(42), x)
    np.testing.assert_allclose(interp_target, interp_hw, rtol=1e-6, atol=1e-6)

  @parameterized.named_parameters(
      ("deconv", "deconv", 1, True, False),
      ("bilinear", "bilinear", 1, True, False),
      ("two_channels_without_global_skip", "deconv", 2, False, False),
      ("attention_with_pos_encoding", "deconv", 1, True, True),
  )
  def test_generator_1d_output_shape(
      self, upsample_mode, output_nc, use_global_skip, use_attention
  ):
    x = np.random.randn(3, 48, 1)
    model = cycle_gan.Generator(
        output_nc=output_nc,
        ngf=2,
        n_res_blocks=1,
        upsample_mode=upsample_mode,
        use_global_skip=use_global_skip,
        use_attention=use_attention,
        use_position_encoding=use_attention,
        num_heads=2,
    )

    out, variables = model.init_with_output(
        jax.random.PRNGKey(0), x, is_training=False
    )
    out_training = jnp.asarray(
        model.apply(
            variables,
            x,
            is_training=True,
            rngs={"dropout": jax.random.PRNGKey(1)},
        )
    )

    self.assertEqual(out.shape, (3, 48, output_nc))
    self.assertEqual(out_training.shape, (3, 48, output_nc))

  def test_generator_1d_interpolated_shape_round_trips(self):
    x = np.random.randn(2, 48, 1)
    model = cycle_gan.Generator(ngf=2, n_res_blocks=1, interpolated_shape=(32,))

    out, _ = model.init_with_output(jax.random.PRNGKey(0), x, is_training=False)

    self.assertEqual(out.shape, x.shape)

  def test_discriminator_1d_output_shape(self):
    x = np.random.randn(2, 48, 1)
    model = cycle_gan.Discriminator(
        base_features=2, n_layers=2, padding="CIRCULAR"
    )

    out, _ = model.init_with_output(jax.random.PRNGKey(0), x)

    # Three stride-2 convolutions: 48 -> 24 -> 12 -> 6.
    self.assertEqual(out.shape, (2, 6, 1))

  @parameterized.parameters((1, 1), (3, 1), (1, 2))
  def test_filtered_interpolation_1d_target_shape(
      self, input_channels, output_nc
  ):
    x = np.random.randn(2, 48, input_channels)
    model = cycle_gan.FilteredInterpolation(
        target_shape=(24,), output_nc=output_nc
    )

    out, _ = model.init_with_output(jax.random.PRNGKey(0), x)

    self.assertEqual(out.shape, (2, 24, output_nc))

  @parameterized.parameters(((2, 1),), ((2, 8, 8, 8, 1),))
  def test_unsupported_spatial_rank_raises(self, shape):
    x = np.zeros(shape)
    gen = cycle_gan.Generator(ngf=2, n_res_blocks=1)
    dis = cycle_gan.Discriminator(base_features=2, n_layers=1)

    with self.assertRaisesRegex(ValueError, "1-D or 2-D"):
      gen.init(jax.random.PRNGKey(0), x, is_training=False)
    with self.assertRaisesRegex(ValueError, "1-D or 2-D"):
      dis.init(jax.random.PRNGKey(0), x)

  def test_global_skip_with_channel_mismatch_raises(self):
    # Previously the 1-channel input was silently broadcast onto the
    # 2-channel output.
    model = cycle_gan.Generator(output_nc=2, ngf=2, n_res_blocks=1)

    with self.assertRaisesRegex(ValueError, "use_global_skip"):
      model.init(
          jax.random.PRNGKey(0), np.zeros((1, 8, 8, 1)), is_training=False
      )

  def test_instance_norm_accepts_a_single_size_one_axis(self):
    # A 2-D map with one axis of size 1 still has several spatial elements.
    model = cycle_gan.Discriminator(
        base_features=2, n_layers=1, padding="CIRCULAR"
    )

    out, _ = model.init_with_output(
        jax.random.PRNGKey(0), np.zeros((2, 8, 2, 1))
    )

    self.assertEqual(out.shape, (2, 2, 1, 1))

  def test_degenerate_instance_norm_raises(self):
    # Two stride-2 convolutions reduce 4x4 to 1x1 before the first instance
    # norm, whose output (and hence the discriminator) would be identically 0
    # (the default `n_layers=3` hits the same case on 16x16 inputs).
    model = cycle_gan.Discriminator(
        base_features=2, n_layers=1, padding="CIRCULAR"
    )

    with self.assertRaisesRegex(ValueError, "spatial"):
      model.init(jax.random.PRNGKey(0), np.zeros((2, 4, 4, 1)))


if __name__ == "__main__":
  absltest.main()
