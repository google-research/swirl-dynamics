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

"""Unit tests for `swirl_dynamics.lib.flow_map.backbones`."""

from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import flow_map
from swirl_dynamics.lib.diffusion import unets


def _apply_arr(module: nn.Module, *args: Any, **kwargs: Any) -> jax.Array:
  """Calls `module.apply` and narrows the non-mutable return to `jax.Array`."""
  return cast(jax.Array, module.apply(*args, **kwargs))


class BackbonesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="min_one", min_freq=1.0, max_freq=100.0),
      dict(testcase_name="min_sub_one", min_freq=0.1, max_freq=100.0),
  )
  def test_fourier_embedding_exponential_spectrum_oracle(
      self, min_freq: float, max_freq: float
  ):
    """Regression test for B1: exponential log-frequencies span [min, max]."""
    dims = 16
    emb = flow_map.FourierEmbedding(
        dims=dims,
        min_freq=min_freq,
        max_freq=max_freq,
        projection=False,
        frequency_scaling="exponential",
        normalization=True,
    )
    x = jnp.asarray([0.001, 0.003], dtype=jnp.float32)
    variables = emb.init(jax.random.PRNGKey(0), x)
    out = _apply_arr(emb, variables, x)
    self.assertEqual(out.shape, (2, dims))

    # sin(pi * f_k * x) * sqrt(2) checks exact frequencies including endpoints.
    expected_freqs = np.exp(np.linspace(np.log(min_freq), np.log(max_freq), 8))
    expected_phase = np.pi * expected_freqs[None, :] * np.asarray(x)[:, None]
    expected = np.sqrt(2.0) * np.concatenate(
        [np.sin(expected_phase), np.cos(expected_phase)], axis=-1
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    # Verify frequencies are strictly increasing (never collapsed or inverted).
    sin_part = np.asarray(out[0, :8])
    self.assertTrue(np.all(np.diff(sin_part) > 0.0))

  def test_fourier_embedding_linear_bfloat16_jvp_and_validation(self):
    emb_lin = flow_map.FourierEmbedding(
        dims=8,
        frequency_scaling="linear",
        projection=True,
        dtype=jnp.bfloat16,
    )
    x = jnp.asarray([0.2, 0.7], dtype=jnp.bfloat16)
    variables = emb_lin.init(jax.random.PRNGKey(1), x)

    def _fn(t_in):
      return _apply_arr(emb_lin, variables, t_in)

    y, dy = jax.jvp(_fn, (x,), (jnp.ones_like(x),))
    self.assertEqual(y.dtype, jnp.bfloat16)
    self.assertTrue(bool(jnp.all(jnp.isfinite(y))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(dy))))

    with self.assertRaisesRegex(ValueError, "even integer >= 2"):
      flow_map.FourierEmbedding(dims=3).init(
          jax.random.PRNGKey(0), jnp.ones((2,))
      )
    with self.assertRaisesRegex(ValueError, "must be 1D"):
      flow_map.FourierEmbedding(dims=8).init(
          jax.random.PRNGKey(0), jnp.ones((2, 1))
      )
    with self.assertRaisesRegex(ValueError, "0 < min_freq <= max_freq"):
      flow_map.FourierEmbedding(dims=8, min_freq=0.0, max_freq=10.0).init(
          jax.random.PRNGKey(0), jnp.ones((2,))
      )
    with self.assertRaisesRegex(ValueError, "Unsupported frequency scaling"):
      flow_map.FourierEmbedding(
          dims=8, frequency_scaling=cast(Any, "invalid")
      ).init(jax.random.PRNGKey(0), jnp.ones((2,)))

  def test_merge_categorical_and_guidance_embeddings(self):
    rng = jax.random.PRNGKey(2)
    emb = jnp.ones((3, 16), dtype=jnp.float32)

    cat = flow_map.MergeCategoricalEmbCond(
        cond_key="label", num_classes=4, features_embedding=8
    )
    cond_cat = {"label": jnp.asarray([0, 2, 3], dtype=jnp.int32)}
    cat_vars = cat.init(rng, emb, cond_cat, is_training=False)
    cat_out = _apply_arr(cat, cat_vars, emb, cond_cat, is_training=False)
    self.assertEqual(cat_out.shape, (3, 16))
    self.assertFalse(bool(jnp.allclose(cat_out[0], cat_out[1])))

    guid = flow_map.MergeGuidanceEmbCond(
        label_key="label", num_classes=4, features_embedding=8
    )
    cond_full = {
        "label": jnp.asarray([0, 2, 4], dtype=jnp.int32),  # 4 == null class
        "cfg_scale": jnp.asarray([1.0, 3.0, 1.0], dtype=jnp.float32),
        "cfg_t_min": jnp.asarray([0.0, 0.2, 0.0], dtype=jnp.float32),
        "cfg_t_max": jnp.asarray([1.0, 0.8, 1.0], dtype=jnp.float32),
    }
    guid_vars = guid.init(rng, emb, cond_full, is_training=False)
    out_full = _apply_arr(guid, guid_vars, emb, cond_full, is_training=False)
    self.assertEqual(out_full.shape, (3, 16))
    self.assertFalse(bool(jnp.allclose(out_full[0], out_full[1])))

    # Absent keys default to null class (4) and (1.0, 0.0, 1.0), matching idx 2.
    out_defaults = _apply_arr(guid, guid_vars, emb, {}, is_training=False)
    np.testing.assert_allclose(
        out_defaults[2], out_full[2], rtol=1e-5, atol=1e-5
    )

    # Scalar 0D class label broadcasts across the batch.
    out_scalar_lbl = _apply_arr(
        guid,
        guid_vars,
        emb,
        {"label": jnp.asarray(0, dtype=jnp.int32)},
        is_training=False,
    )
    np.testing.assert_allclose(
        out_scalar_lbl[0], out_full[0], rtol=1e-5, atol=1e-5
    )

    # Scalar guidance value and label_key=None mode.
    guid_nocls = flow_map.MergeGuidanceEmbCond(
        label_key=None, scalar_keys=("cfg_scale",), features_embedding=8
    )
    cond_scalar = {"cfg_scale": jnp.asarray(2.0, dtype=jnp.float32)}
    nocls_vars = guid_nocls.init(rng, emb, cond_scalar, is_training=False)
    out_nocls = _apply_arr(
        guid_nocls, nocls_vars, emb, cond_scalar, is_training=False
    )
    self.assertEqual(out_nocls.shape, (3, 16))

  @parameterized.named_parameters(
      dict(testcase_name="small_ch_2", num_channels=(2,)),
      dict(testcase_name="small_ch_6", num_channels=(6,)),
  )
  def test_flow_map_unet_b2_regressions_scalar_s_resize_and_small_channels(
      self, num_channels: tuple[int, ...]
  ):
    """Regression test for B2: scalar s, 1D FilteredResize, small channels."""
    unet = flow_map.FlowMapUNet(
        out_channels=2,
        resize_to_shape=(8,),
        num_channels=num_channels,
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        use_position_encoding=False,
        time_embedding_merge="add",
        time_conditioning="t_dt",
        use_global_skip=True,
        cond_embed_fn=flow_map.MergeGuidanceEmbCond,
        cond_embed_kwargs={"num_classes": 3, "features_embedding": 8},
    )
    x = jnp.ones((2, 10, 2), dtype=jnp.float32)
    t_scalar = jnp.asarray(0.7, dtype=jnp.float32)
    s_scalar = jnp.asarray(0.2, dtype=jnp.float32)
    cond = {"label": jnp.asarray([0, 1], dtype=jnp.int32)}
    variables = unet.init(
        jax.random.PRNGKey(3), x, t_scalar, s_scalar, cond, is_training=False
    )
    out = _apply_arr(
        unet, variables, x, t_scalar, s_scalar, cond, is_training=False
    )
    self.assertEqual(out.shape, x.shape)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_flow_map_unet_cond_embed_kwargs_none_and_channel_cond(self):
    unet = flow_map.FlowMapUNet(
        out_channels=2,
        num_channels=(4,),
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        use_position_encoding=False,
        cond_merging_fn=unets.InterpConvMerge,
        cond_embed_dim=4,
        cond_embed_fn=flow_map.MergeGuidanceEmbCond,
        cond_embed_kwargs=None,
    )
    x = jnp.ones((2, 8, 2), dtype=jnp.float32)
    t = jnp.asarray([0.8, 0.5], dtype=jnp.float32)
    s = jnp.asarray([0.2, 0.1], dtype=jnp.float32)
    cond = {
        "channel:aux": jnp.ones((2, 8, 1), dtype=jnp.float32),
        "label": jnp.asarray([1, 2], dtype=jnp.int32),
    }
    variables = unet.init(
        jax.random.PRNGKey(4), x, t, s, cond, is_training=False
    )
    out = _apply_arr(unet, variables, x, t, s, cond, is_training=False)
    self.assertEqual(out.shape, x.shape)

    # Different input and output channels succeed when use_global_skip=False.
    unet_diff_ch = flow_map.FlowMapUNet(
        out_channels=4,
        num_channels=(4,),
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        use_position_encoding=False,
        use_global_skip=False,
    )
    diff_vars = unet_diff_ch.init(
        jax.random.PRNGKey(6), x, t, s, is_training=False
    )
    diff_out = _apply_arr(unet_diff_ch, diff_vars, x, t, s, is_training=False)
    self.assertEqual(diff_out.shape, (2, 8, 4))

  @parameterized.named_parameters(
      dict(testcase_name="1d_float32", shape=(2, 8, 2), dtype=jnp.float32),
      dict(testcase_name="2d_bfloat16", shape=(2, 8, 8, 3), dtype=jnp.bfloat16),
  )
  def test_rescaled_flow_map_and_mean_flow_unets_boundary_and_jvp(
      self, shape: tuple[int, ...], dtype: jnp.dtype
  ):
    rng = jax.random.PRNGKey(5)
    x = jax.random.normal(rng, shape, dtype=dtype)
    t = jnp.asarray([0.8, 0.6], dtype=jnp.float32)
    s = jnp.asarray([0.2, 0.1], dtype=jnp.float32)

    fm_unet = flow_map.RescaledFlowMapUNet(
        out_channels=shape[-1],
        num_channels=(8,),
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        use_position_encoding=False,
        time_rescale=10.0,
        dtype=dtype,
    )
    fm_vars = fm_unet.init(rng, x, t, s, is_training=False)
    # Exact boundary condition X_{s, s}(x_s) == x_s.
    at_diag = _apply_arr(fm_unet, fm_vars, x, s, s, is_training=False)
    np.testing.assert_allclose(
        np.asarray(at_diag, dtype=np.float32),
        np.asarray(x, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )

    mf_unet = flow_map.RescaledMeanFlowUNet(
        out_channels=shape[-1],
        num_channels=(8,),
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        use_position_encoding=False,
        time_conditioning="t_dt",
        dtype=dtype,
    )
    mf_vars = mf_unet.init(rng, x, t, s, is_training=False)

    def _u_fn(x_in, t_in, s_in):
      return _apply_arr(mf_unet, mf_vars, x_in, t_in, s_in, is_training=False)

    u_val, du_dt = jax.jvp(
        _u_fn,
        (x, t, s),
        (jnp.ones_like(x), jnp.ones_like(t), jnp.zeros_like(s)),
    )
    self.assertEqual(u_val.shape, shape)
    self.assertTrue(bool(jnp.all(jnp.isfinite(u_val))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(du_dt))))

    # Scalar times also broadcast cleanly through validate_flow_map_inputs.
    scalar_out = _apply_arr(
        mf_unet,
        mf_vars,
        x,
        jnp.asarray(0.7, dtype=jnp.float32),
        jnp.asarray(0.3, dtype=jnp.float32),
        is_training=False,
    )
    self.assertEqual(scalar_out.shape, shape)

  def test_validation_error_branches(self):
    x = jnp.ones((2, 8, 2), dtype=jnp.float32)
    t = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
    s = jnp.asarray([0.2, 0.2], dtype=jnp.float32)

    with self.assertRaisesRegex(ValueError, "Number of channels"):
      flow_map.validate_flow_map_inputs(x, t, s, out_channels=3)
    with self.assertRaisesRegex(ValueError, "Time dimensions"):
      flow_map.validate_flow_map_inputs(x, jnp.asarray(0.5), s, out_channels=2)
    with self.assertRaisesRegex(ValueError, "leading .* dimension"):
      flow_map.validate_flow_map_inputs(
          x, jnp.ones((3,)), jnp.ones((3,)), out_channels=2
      )

    base = flow_map.FlowMapUNet(
        out_channels=2,
        num_channels=(4,),
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        use_position_encoding=False,
    )
    with self.assertRaisesRegex(ValueError, "t must be 1D"):
      base.init(jax.random.PRNGKey(0), x, jnp.ones((3,)), s, is_training=False)
    with self.assertRaisesRegex(ValueError, "s must be 1D"):
      base.init(jax.random.PRNGKey(0), x, t, jnp.ones((3,)), is_training=False)

    bad_cond = flow_map.FlowMapUNet(
        out_channels=2,
        num_channels=(4,),
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        use_position_encoding=False,
        time_conditioning=cast(Any, "bad"),
    )
    with self.assertRaisesRegex(ValueError, "time_conditioning"):
      bad_cond.init(jax.random.PRNGKey(0), x, t, s, is_training=False)

    bad_merge = flow_map.FlowMapUNet(
        out_channels=2,
        num_channels=(4,),
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        use_position_encoding=False,
        time_embedding_merge=cast(Any, "bad"),
    )
    with self.assertRaisesRegex(ValueError, "time embedding merge"):
      bad_merge.init(jax.random.PRNGKey(0), x, t, s, is_training=False)


if __name__ == "__main__":
  absltest.main()
