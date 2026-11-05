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
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import flow_matching
from swirl_dynamics.lib.diffusion import unets


class _BatchNormDropoutFlow(nn.Module):
  out_channels: int
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, sigma, cond=None, is_training=False):
    del sigma
    h = nn.BatchNorm(use_running_average=not is_training)(x)
    if cond is not None and "cond_vec" in cond:
      h = h + nn.Dense(x.shape[-1])(cond["cond_vec"])
    if self.dropout_rate > 0.0:
      h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not is_training)
    return nn.Dense(self.out_channels)(h)


class _PureDropoutConstantFlow(nn.Module):

  @nn.compact
  def __call__(self, x, sigma, cond=None, is_training=False):
    del sigma, cond
    scale = self.param("scale", nn.initializers.ones, (1,))
    return nn.Dropout(rate=0.5)(
        jnp.ones_like(x) * scale, deterministic=not is_training
    )


class _DualHeadFlowScoreNet(nn.Module):
  out_channels: int

  @nn.compact
  def __call__(self, x, sigma, is_training=False):
    h = nn.BatchNorm(use_running_average=not is_training)(x)
    t_feat = sigma.reshape((x.shape[0],) + (x.ndim - 1) * (1,))
    v = nn.Dense(self.out_channels, name="flow_head")(h + t_feat)
    s = nn.Dense(self.out_channels, name="score_head")(h - t_feat)
    return v, s


class _StatefulMeanFlowNet(nn.Module):
  out_channels: int

  @nn.compact
  def __call__(self, x_s, t, s, cond=None, is_training=False):
    h = nn.BatchNorm(use_running_average=not is_training)(x_s)
    dt = (t - s).reshape((x_s.shape[0],) + (x_s.ndim - 1) * (1,))
    if cond is not None and "cond_vec" in cond:
      h = h + nn.Dense(x_s.shape[-1])(cond["cond_vec"])
    return nn.Dense(self.out_channels)(h) + dt


class _StatelessTeacherFlow(nn.Module):
  out_channels: int

  @nn.compact
  def __call__(self, x, sigma, cond=None, is_training=False):
    del sigma, is_training
    h = x
    if cond is not None and "cond_vec" in cond:
      h = h + nn.Dense(x.shape[-1])(cond["cond_vec"])
    return nn.Dense(self.out_channels)(h)


class _UnconditionalOnlyFlow(nn.Module):
  out_channels: int

  @nn.compact
  def __call__(self, x, sigma, *, is_training=False):
    del is_training
    t_feat = sigma.reshape((x.shape[0],) + (x.ndim - 1) * (1,))
    return nn.Dense(self.out_channels)(x) + t_feat


class _StrictDictCondFlow(nn.Module):
  out_channels: int

  @nn.compact
  def __call__(self, x, sigma, cond=None, *, is_training=False):
    del sigma, is_training
    if cond is not None and not isinstance(cond, dict):
      raise TypeError(f"Expected plain dict for cond, got {type(cond)}")
    return nn.Dense(self.out_channels)(x)


class _ConstantVelocityFlow(nn.Module):

  @nn.compact
  def __call__(self, x, sigma, cond=None, is_training=False):
    del sigma, cond, is_training
    scale = self.param("scale", nn.initializers.ones, (1,))
    return jnp.ones_like(x) * scale


class FlowMatchingTest(parameterized.TestCase):

  def test_public_exports_importable(self):
    self.assertNotEmpty(flow_matching.__all__)
    self.assertEqual(
        flow_matching.__all__, sorted(set(flow_matching.__all__))
    )
    for name in flow_matching.__all__:
      with self.subTest(symbol=name):
        self.assertTrue(hasattr(flow_matching, name))

  @parameterized.named_parameters(
      ("linear", flow_matching.LinearInterpolant, True),
      (
          "linear_sinusoidal",
          flow_matching.LinearInterpolantSinusoidalNoise,
          True,
      ),
      ("trigonometric", flow_matching.TrigonometricInterpolant, False),
      ("rectified_flow", flow_matching.RectifiedFlow, False),
  )
  def test_interpolants_time_derivative_and_losses(
      self, interp_cls, has_stochastic_noise
  ):
    interp = interp_cls(sigma=0.5) if has_stochastic_noise else interp_cls()
    self.assertNotEqual(interp, None)
    self.assertNotEqual(interp, "not_an_interpolant")
    self.assertEqual(interp, interp)

    alpha = lambda t: 1.0 - t
    beta = lambda t: t
    gamma = lambda t: 0.0 * t
    dot_a = lambda t: -jnp.ones_like(t)
    dot_b = jnp.ones_like
    dot_b_alt = lambda t: 2.0 * jnp.ones_like(t)
    custom_1 = flow_matching.StochasticInterpolant(
        alpha=alpha,
        alpha_dot=dot_a,
        beta=beta,
        beta_dot=dot_b,
        gamma=gamma,
        gamma_dot=gamma,
    )
    custom_2 = flow_matching.StochasticInterpolant(
        alpha=alpha,
        alpha_dot=dot_a,
        beta=beta,
        beta_dot=dot_b_alt,
        gamma=gamma,
        gamma_dot=gamma,
    )
    self.assertNotEqual(custom_1, custom_2)
    self.assertNotEqual(hash(custom_1), hash(custom_2))

    t = jnp.asarray([0.25, 0.7], dtype=jnp.float32)
    x_0 = jnp.asarray([[1.0, -0.5, 2.0], [0.2, 1.5, -1.0]], dtype=jnp.float32)
    x_1 = jnp.asarray([[-0.5, 2.0, 0.5], [1.2, -0.5, 0.0]], dtype=jnp.float32)
    z = jnp.asarray([[0.3, -0.7, 0.1], [-0.4, 0.2, 0.8]], dtype=jnp.float32)

    eps = 1e-3
    fd_deriv = (
        interp.calculate_interpolant(t + eps, x_0, x_1, z)
        - interp.calculate_interpolant(t - eps, x_0, x_1, z)
    ) / (2.0 * eps)
    exact_deriv = interp.calculate_time_derivative_interpolant(t, x_0, x_1, z)
    np.testing.assert_allclose(exact_deriv, fd_deriv, rtol=2e-3, atol=2e-3)

    v_loss = flow_matching.velocity_loss(exact_deriv, t, x_0, x_1, z, interp)
    d_loss = flow_matching.denoising_loss(z, t, x_0, x_1, z, interp)
    self.assertAlmostEqual(float(v_loss), 0.0, places=6)
    self.assertAlmostEqual(float(d_loss), 0.0, places=6)

    if has_stochastic_noise:
      target_score = interp.calculate_target_score(t, x_0, x_1, z)
      s_loss = flow_matching.score_loss(target_score, t, x_0, x_1, z, interp)
      self.assertAlmostEqual(float(s_loss), 0.0, places=6)

  def test_cond_sample_from_shape(self):
    self.assertIsNone(flow_matching.cond_sample_from_shape(None))
    nested_shape = flax.core.freeze({"a": (4,), "nested": {"b": (2, 3)}})
    sample = flow_matching.cond_sample_from_shape(
        nested_shape, batch_dims=(2,)
    )
    self.assertEqual(sample["a"].shape, (2, 4))
    self.assertEqual(sample["nested"]["b"].shape, (2, 2, 3))
    with self.assertRaisesRegex(TypeError, "Cannot initialize shape"):
      flow_matching.cond_sample_from_shape({"bad": 123})

  def test_stochastic_interpolant_and_conditional_models(self):
    input_shape = (6, 2)
    batch_size = 4
    rng = jax.random.PRNGKey(0)
    batch = {
        "x_0": jnp.zeros((batch_size,) + input_shape, dtype=jnp.float32),
        "x_1": jnp.ones((batch_size,) + input_shape, dtype=jnp.float32),
        "cond_vec": jnp.ones((batch_size, 6, 3), dtype=jnp.float32),
    }

    si_model = flow_matching.StochasticInterpolantModel(
        input_shape=input_shape,
        flow_model=_BatchNormDropoutFlow(out_channels=2),
        interpolant=flow_matching.LinearInterpolant(),
        loss_stochastic_interpolant=flow_matching.velocity_loss,
        use_same_noise_rng=False,
    )
    si_vars = si_model.initialize(rng)
    loss, (metrics, mutables) = jax.jit(si_model.loss_fn)(
        si_vars["params"],
        batch,
        rng,
        {"batch_stats": si_vars["batch_stats"]},
    )
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertIn("loss", metrics)
    self.assertIn("batch_stats", mutables)

    eval_metrics = jax.jit(si_model.eval_fn)(
        {"params": si_vars["params"], **mutables},
        {"x_0": batch["x_0"], "x_1": batch["x_1"]},
        rng,
    )
    self.assertLen(eval_metrics, si_model.num_eval_time_levels)

    # Verify eval_fn uses noising_process_flow for the target velocity.
    zero_noise = lambda key, shape: jnp.zeros(shape, dtype=jnp.float32)
    score_free_si = flow_matching.StochasticInterpolantModel(
        input_shape=input_shape,
        flow_model=_ConstantVelocityFlow(),
        interpolant=flow_matching.LinearInterpolant(sigma=1.0),
        noising_process_interp=jax.random.normal,
        noising_process_flow=zero_noise,
        use_same_noise_rng=False,
        loss_stochastic_interpolant=flow_matching.velocity_loss,
    )
    sf_vars = score_free_si.initialize(rng)
    sf_eval = score_free_si.eval_fn(
        sf_vars, {"x_0": batch["x_0"], "x_1": batch["x_1"]}, rng
    )
    for lvl_loss in sf_eval.values():
      self.assertAlmostEqual(float(lvl_loss), 0.0, places=6)

    score_free_cond = flow_matching.ConditionalStochasticInterpolantModel(
        input_shape=input_shape,
        cond_shape={"cond_vec": (6, 3)},
        flow_model=_ConstantVelocityFlow(),
        interpolant=flow_matching.LinearInterpolant(sigma=1.0),
        noising_process_interp=jax.random.normal,
        noising_process_flow=zero_noise,
        use_same_noise_rng=False,
        loss_stochastic_interpolant=flow_matching.velocity_loss,
    )
    sfc_vars = score_free_cond.initialize(rng)
    batch_with_metadata = {
        **batch,
        "step": jnp.int32(7),
        "unbatched_meta": jnp.asarray([1.0, 2.0]),
    }
    sfc_eval = score_free_cond.eval_fn(sfc_vars, batch_with_metadata, rng)
    for lvl_loss in sfc_eval.values():
      self.assertAlmostEqual(float(lvl_loss), 0.0, places=6)

    inf_fn = si_model.inference_fn(
        {"params": si_vars["params"], **mutables}, si_model.flow_model
    )
    self.assertEqual(
        inf_fn(batch["x_0"], 0.5).shape, (batch_size,) + input_shape
    )

    for cond_shape in ({"cond_vec": (6, 3)}, {}):
      cond_model = flow_matching.ConditionalStochasticInterpolantModel(
          input_shape=input_shape,
          cond_shape=cond_shape,
          flow_model=_BatchNormDropoutFlow(out_channels=2),
          interpolant=flow_matching.RectifiedFlow(),
          loss_stochastic_interpolant=flow_matching.velocity_loss,
      )
      cond_vars = cond_model.initialize(rng)
      c_loss, (_, c_mut) = jax.jit(cond_model.loss_fn)(
          cond_vars["params"],
          batch_with_metadata,
          rng,
          {"batch_stats": cond_vars["batch_stats"]},
      )
      self.assertTrue(bool(jnp.isfinite(c_loss)))
      c_eval = jax.jit(cond_model.eval_fn)(
          {"params": cond_vars["params"], **c_mut}, batch_with_metadata, rng
      )
      self.assertLen(c_eval, cond_model.num_eval_time_levels)

  def test_crps_and_flow_score_models(self):
    input_shape = (32, 1)
    batch_size = 32
    zero_noise = lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32)

    crps_bn = flow_matching.StochasticInterpolantCRPSModel(
        input_shape=(8, 1),
        flow_model=_BatchNormDropoutFlow(out_channels=1, dropout_rate=0.5),
        interpolant=flow_matching.LinearInterpolant(),
        loss_stochastic_interpolant=flow_matching.velocity_loss,
        noising_process_interp=zero_noise,
    )
    bn_vars = crps_bn.initialize(jax.random.PRNGKey(0))
    const_batch = {
        "x_0": jnp.full((8, 8, 1), 4.0, dtype=jnp.float32),
        "x_1": jnp.full((8, 8, 1), 4.0, dtype=jnp.float32),
    }
    _, (_, updated_mut) = crps_bn.loss_fn(
        bn_vars["params"],
        const_batch,
        jax.random.PRNGKey(1),
        {"batch_stats": bn_vars["batch_stats"]},
    )
    np.testing.assert_allclose(
        updated_mut["batch_stats"]["BatchNorm_0"]["mean"],
        np.full((1,), 0.04, dtype=np.float32),
        atol=1e-6,
    )

    crps_fair = flow_matching.StochasticInterpolantCRPSModel(
        input_shape=input_shape,
        flow_model=_PureDropoutConstantFlow(),
        interpolant=flow_matching.LinearInterpolant(),
        loss_stochastic_interpolant=flow_matching.velocity_loss,
        noising_process_interp=zero_noise,
        noising_process_flow=zero_noise,
    )
    fair_vars = crps_fair.initialize(jax.random.PRNGKey(2))
    fair_loss, _ = crps_fair.loss_fn(
        fair_vars["params"],
        {
            "x_0": jnp.zeros((batch_size,) + input_shape, dtype=jnp.float32),
            "x_1": jnp.ones((batch_size,) + input_shape, dtype=jnp.float32),
        },
        jax.random.PRNGKey(3),
        {},
    )
    self.assertAlmostEqual(float(fair_loss), 0.5, delta=0.08)

    fs_model = flow_matching.StochasticInterpolantFlowScoreModel(
        input_shape=(6, 2),
        flow_score_model=_DualHeadFlowScoreNet(out_channels=2),
        interpolant=flow_matching.LinearInterpolantSinusoidalNoise(),
        loss_stochastic_interpolant_flow=flow_matching.velocity_loss,
        loss_stochastic_interpolant_score=flow_matching.score_loss,
    )
    fs_vars = fs_model.initialize(jax.random.PRNGKey(4))
    fs_batch = {
        "x_0": jnp.zeros((4, 6, 2), dtype=jnp.float32),
        "x_1": jnp.ones((4, 6, 2), dtype=jnp.float32),
    }
    fs_loss, (fs_metrics, fs_mut) = jax.jit(fs_model.loss_fn)(
        fs_vars["params"],
        fs_batch,
        jax.random.PRNGKey(5),
        {"batch_stats": fs_vars["batch_stats"]},
    )
    self.assertTrue(bool(jnp.isfinite(fs_loss)))
    self.assertIn("loss_flow", fs_metrics)
    self.assertIn("loss_score", fs_metrics)
    fs_eval = jax.jit(fs_model.eval_fn)(
        {"params": fs_vars["params"], **fs_mut}, fs_batch, jax.random.PRNGKey(6)
    )
    self.assertLen(fs_eval, fs_model.num_eval_time_levels)

  def test_rescaled_wrappers_and_validation(self):
    rng = jax.random.PRNGKey(0)
    wrapped = flow_matching.TimeRescaled(
        network=_BatchNormDropoutFlow(out_channels=2), time_rescale=100.0
    )
    x_2d = jnp.ones((2, 4, 4, 2), dtype=jnp.float32)
    w_vars = wrapped.init(rng, x_2d, jnp.asarray(0.5), is_training=False)
    self.assertEqual(
        np.asarray(
            wrapped.apply(w_vars, x_2d, jnp.asarray(0.5), is_training=False)
        ).shape,
        (2, 4, 4, 2),
    )

    # Unconditional-only network (no `cond` parameter) and FrozenDict `cond`.
    uncond_wrapped = flow_matching.TimeRescaled(
        network=_UnconditionalOnlyFlow(out_channels=2), time_rescale=10.0
    )
    uw_vars = uncond_wrapped.init(
        rng, x_2d, jnp.asarray(0.5), is_training=False
    )
    uw_out = uncond_wrapped.apply(
        uw_vars, x_2d, jnp.asarray(0.5), is_training=False
    )
    self.assertEqual(np.asarray(uw_out).shape, (2, 4, 4, 2))

    cond_wrapped = flow_matching.TimeRescaled(
        network=_StrictDictCondFlow(out_channels=2), time_rescale=10.0
    )
    frozen_cond = flax.core.freeze(
        {"cond_vec": jnp.ones((2, 4, 4, 1), dtype=jnp.float32)}
    )
    cw_vars = cond_wrapped.init(
        rng, x_2d, jnp.asarray(0.5), cond=frozen_cond, is_training=False
    )
    cw_out = cond_wrapped.apply(
        cw_vars, x_2d, jnp.asarray(0.5), cond=frozen_cond, is_training=False
    )
    self.assertEqual(np.asarray(cw_out).shape, (2, 4, 4, 2))

    with self.assertRaisesRegex(ValueError, "sigma must be 1D"):
      wrapped.apply(
          w_vars, x_2d, jnp.ones((3,), dtype=jnp.float32), is_training=False
      )

    unet_2d = flow_matching.RescaledUnet(
        out_channels=2,
        num_channels=(4, 4),
        downsample_ratio=(2, 2),
        num_blocks=1,
        noise_embed_dim=8,
        use_attention=False,
        cond_merging_fn=unets.InterpConvMerge,
    )
    cond_frozen = flax.core.freeze(
        {"channel:low_res": jnp.ones((2, 4, 4, 1), dtype=jnp.float32)}
    )
    u2_vars = unet_2d.init(
        rng, x_2d, jnp.asarray(0.25), cond=cond_frozen, is_training=False
    )
    u2_out = unet_2d.apply(
        u2_vars, x_2d, jnp.asarray(0.25), cond=cond_frozen, is_training=False
    )
    self.assertEqual(np.asarray(u2_out).shape, (2, 4, 4, 2))
    with self.assertRaisesRegex(
        ValueError,
        r"Number of channels in the input \(3\) must match .* output \(2\)\.",
    ):
      unet_2d.init(
          rng,
          jnp.ones((2, 4, 4, 3), dtype=jnp.float32),
          jnp.ones((2,), dtype=jnp.float32),
          is_training=False,
      )

    unet_3d = flow_matching.RescaledUnet3d(
        out_channels=2,
        num_channels=(4,),
        downsample_ratio=(2,),
        num_blocks=1,
        noise_embed_dim=8,
        input_proj_channels=4,
        output_proj_channels=4,
        use_spatial_attention=False,
        use_temporal_attention=False,
        use_position_encoding=False,
        num_heads=1,
    )
    x_3d = jnp.ones((2, 2, 4, 4, 2), dtype=jnp.float32)
    u3_vars = unet_3d.init(rng, x_3d, jnp.asarray(0.5), is_training=False)
    self.assertEqual(
        np.asarray(
            unet_3d.apply(u3_vars, x_3d, jnp.asarray(0.5), is_training=False)
        ).shape,
        (2, 2, 4, 4, 2),
    )

  def test_mean_flow_models_deduplication_and_mutables(self):
    rng = jax.random.PRNGKey(0)
    samples = flow_matching.logit_normal_dist(rng, (8,), mean=-0.2, std=0.8)
    self.assertTrue(bool(jnp.all((samples > 0.0) & (samples < 1.0))))

    t_diag, s_diag = flow_matching.time_sampler_mean_flow(
        rng, batch_size=8, diagonal_sampling_rate=1.0
    )
    np.testing.assert_allclose(t_diag, s_diag, atol=1e-7)

    input_shape = (6, 2)
    batch_size = 4
    const_batch = {
        "x_0": jnp.full((batch_size,) + input_shape, 4.0, dtype=jnp.float32),
        "x_1": jnp.full((batch_size,) + input_shape, 4.0, dtype=jnp.float32),
        "cond_vec": jnp.ones((batch_size, 6, 3), dtype=jnp.float32),
    }

    mf_net = _StatefulMeanFlowNet(out_channels=2)
    teacher_net = _StatelessTeacherFlow(out_channels=2)
    stateful_teacher_net = _BatchNormDropoutFlow(out_channels=2)

    stateful_uncond_teacher_vars = stateful_teacher_net.init(
        rng,
        x=jnp.ones((1,) + input_shape),
        sigma=jnp.ones((1,)),
        is_training=False,
    )
    mf_model = flow_matching.MeanFlowModel(
        input_shape=input_shape,
        mean_flow_model=mf_net,
        flow_model=stateful_teacher_net,
        params_flow=stateful_uncond_teacher_vars,
        noising_process=lambda key, shape: jnp.zeros(shape, dtype=jnp.float32),
    )
    mf_vars = mf_model.initialize(rng)
    (mf_loss, (mf_metrics, mf_mut)), mf_grads = jax.jit(
        jax.value_and_grad(mf_model.loss_fn, has_aux=True)
    )(
        mf_vars["params"],
        const_batch,
        rng,
        {"batch_stats": mf_vars["batch_stats"]},
    )
    self.assertTrue(bool(jnp.isfinite(mf_loss)))
    self.assertIn("raw_mse", mf_metrics)
    self.assertTrue(
        all(
            bool(jnp.all(jnp.isfinite(g)))
            for g in jax.tree.leaves(mf_grads)
        )
    )
    np.testing.assert_allclose(
        mf_mut["batch_stats"]["BatchNorm_0"]["mean"],
        np.full((2,), 0.04, dtype=np.float32),
        atol=1e-6,
    )

    mf_eval = jax.jit(mf_model.eval_fn)(
        {"params": mf_vars["params"], **mf_mut}, const_batch, rng
    )
    self.assertIn("eval_loss", mf_eval)
    self.assertEqual(
        mf_eval["eval_samples"].shape, (batch_size,) + input_shape
    )

    mf_missing = flow_matching.MeanFlowModel(
        input_shape=input_shape,
        mean_flow_model=mf_net,
        flow_model=teacher_net,
        params_flow=None,
    )
    with self.assertRaisesRegex(
        ValueError, "`params_flow` must be provided when `flow_model` is set."
    ):
      mf_missing.loss_fn(
          mf_vars["params"],
          const_batch,
          rng,
          {"batch_stats": mf_vars["batch_stats"]},
      )

    cond_teacher_vars = stateful_teacher_net.init(
        rng,
        x=jnp.ones((1,) + input_shape),
        sigma=jnp.ones((1,)),
        cond={"cond_vec": jnp.ones((1, 6, 3))},
        is_training=False,
    )
    empty_teacher_vars = teacher_net.init(
        rng,
        x=jnp.ones((1,) + input_shape),
        sigma=jnp.ones((1,)),
        cond={},
        is_training=False,
    )
    for model_cls in (
        flow_matching.ConditionalMeanFlowModel,
        flow_matching.ConditionalSymmetricMeanFlowModel,
    ):
      for cond_shape in ({"cond_vec": (6, 3)}, {}):
        t_net = stateful_teacher_net if cond_shape else teacher_net
        t_vars = (
            cond_teacher_vars if cond_shape else empty_teacher_vars["params"]
        )
        c_model = model_cls(
            input_shape=input_shape,
            cond_shape=cond_shape,
            mean_flow_model=mf_net,
            flow_model=t_net,
            params_flow=t_vars,
            noising_process=lambda key, shape: jnp.zeros(
                shape, dtype=jnp.float32
            ),
        )
        c_vars = c_model.initialize(rng)
        (c_loss, (c_metrics, c_mut)), c_grads = jax.jit(
            jax.value_and_grad(c_model.loss_fn, has_aux=True)
        )(
            c_vars["params"],
            const_batch,
            rng,
            {"batch_stats": c_vars["batch_stats"]},
        )
        self.assertTrue(bool(jnp.isfinite(c_loss)))
        self.assertIn("raw_mse", c_metrics)
        if model_cls is flow_matching.ConditionalSymmetricMeanFlowModel:
          self.assertIn("loss_t", c_metrics)
          self.assertIn("loss_s", c_metrics)
        self.assertTrue(
            all(
                bool(jnp.all(jnp.isfinite(g)))
                for g in jax.tree.leaves(c_grads)
            )
        )
        np.testing.assert_allclose(
            c_mut["batch_stats"]["BatchNorm_0"]["mean"],
            np.full((2,), 0.04, dtype=np.float32),
            atol=1e-6,
        )
        c_eval = jax.jit(c_model.eval_fn)(
            {"params": c_vars["params"], **c_mut}, const_batch, rng
        )
        self.assertIn("eval_loss", c_eval)
        inf_fn = model_cls.inference_fn(
            {"params": c_vars["params"], **c_mut}, mf_net
        )
        self.assertEqual(
            inf_fn(
                const_batch["x_0"],
                1.0,
                0.0,
                {"cond_vec": const_batch["cond_vec"]} if cond_shape else {},
            ).shape,
            (batch_size,) + input_shape,
        )


if __name__ == "__main__":
  absltest.main()
