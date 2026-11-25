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

"""Unit tests for `swirl_dynamics.lib.flow_map.flow_map`."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import flow_map
from swirl_dynamics.lib import flow_matching


class _ConstantTeacherFlow(nn.Module):
  """Exact constant-velocity teacher `v(x, t) = drift` (optionally stateful)."""

  drift: float = 2.0
  stateful: bool = False

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      sigma: jax.Array,
      cond: flow_map.CondDict | None = None,
      *,
      is_training: bool,
  ) -> jax.Array:
    del sigma
    scale = self.param("scale", nn.initializers.ones, ())
    h = x
    if self.stateful:
      h = nn.BatchNorm(use_running_average=not is_training)(h)
    bias = self.drift * scale
    if cond is not None and "offset" in cond:
      bias = bias + cond["offset"].reshape((x.shape[0],) + (1,) * (x.ndim - 1))
    return jnp.zeros_like(h) + bias


class _ExactLinearFlowMap(nn.Module):
  """Exact flow map `X_{s, t}(x_s) = x_s + (t - s) * drift`."""

  drift: float = 2.0

  @nn.compact
  def __call__(
      self,
      x_s: jax.Array,
      t: jax.Array,
      s: jax.Array,
      cond: flow_map.CondDict | None = None,
      *,
      is_training: bool,
  ) -> jax.Array:
    del is_training
    scale = self.param("scale", nn.initializers.ones, ())
    dt = (t - s).reshape((x_s.shape[0],) + (1,) * (x_s.ndim - 1))
    bias = self.drift * scale
    if cond is not None and "offset" in cond:
      bias = bias + cond["offset"].reshape(
          (x_s.shape[0],) + (1,) * (x_s.ndim - 1)
      )
    return x_s + dt * bias


class _ExactMeanVelocityNet(nn.Module):
  """Exact average velocity `u(x_s, t, s) = drift` with optional scale param."""

  drift: float = 2.0
  stateful: bool = False

  @nn.compact
  def __call__(
      self,
      x_s: jax.Array,
      t: jax.Array,
      s: jax.Array,
      cond: flow_map.CondDict | None = None,
      *,
      is_training: bool,
  ) -> jax.Array:
    scale = self.param("scale", nn.initializers.ones, ())
    slope = self.param("slope", nn.initializers.zeros, ())
    h = x_s
    if self.stateful:
      h = nn.BatchNorm(use_running_average=not is_training)(h)
    dt = (t - s).reshape((x_s.shape[0],) + (1,) * (x_s.ndim - 1))
    out = jnp.zeros_like(h) + self.drift * scale + slope * (h + dt)
    if cond is not None and "offset" in cond:
      out = out + cond["offset"].reshape(
          (x_s.shape[0],) + (1,) * (x_s.ndim - 1)
      )
    return out


class _StatefulFlowMapNet(nn.Module):
  """Minimal BatchNorm two-time module for B3 mutables regression tests."""

  @nn.compact
  def __call__(
      self,
      x_s: jax.Array,
      t: jax.Array,
      s: jax.Array,
      cond: flow_map.CondDict | None = None,
      *,
      is_training: bool,
  ) -> jax.Array:
    h = nn.BatchNorm(use_running_average=not is_training)(x_s)
    dt = (t - s).reshape((x_s.shape[0],) + (1,) * (x_s.ndim - 1))
    w = self.param("w", nn.initializers.ones, ())
    out = x_s + dt * (w * h)
    if cond is not None and "offset" in cond:
      out = out + cond["offset"].reshape(
          (x_s.shape[0],) + (1,) * (x_s.ndim - 1)
      )
    return out


class FlowMapModelsTest(parameterized.TestCase):

  def test_lagrangian_flow_map_closed_form_zero_loss_oracle(self):
    """Exact linear drift `x_1 = x_0 + 2` yields zero Lagrangian loss."""
    teacher = _ConstantTeacherFlow(drift=2.0)
    x_dummy = jnp.ones((1, 4, 1))
    teacher_vars = teacher.init(
        jax.random.PRNGKey(0),
        x=x_dummy,
        sigma=jnp.ones((1,)),
        is_training=False,
    )

    model = flow_map.LagrangianFlowMapModel(
        input_shape=(4, 1),
        flow_map_model=_ExactLinearFlowMap(drift=2.0),
        flow_model=teacher,
        params_flow=teacher_vars["params"],
        interpolant=flow_matching.RectifiedFlow(),
        number_of_eval_steps=(1, 2),
    )
    variables = model.initialize(jax.random.PRNGKey(1))
    x_0 = jnp.ones((3, 4, 1))
    batch = {"x_0": x_0, "x_1": x_0 + 2.0}
    loss, (metrics, mutables) = model.loss_fn(
        variables["params"], batch, jax.random.PRNGKey(2), {}
    )
    self.assertAlmostEqual(float(loss), 0.0, places=6)
    self.assertAlmostEqual(float(metrics["loss"]), 0.0, places=6)
    self.assertEqual(mutables, {})

    eval_out = model.eval_fn(variables, batch, jax.random.PRNGKey(3))
    self.assertAlmostEqual(float(eval_out["eval_loss"]), 0.0, places=6)
    np.testing.assert_allclose(eval_out["1"], x_0 + 2.0, atol=1e-5)
    np.testing.assert_allclose(eval_out["2"], x_0 + 2.0, atol=1e-5)

    infer = model.inference_fn(variables, model.flow_map_model)
    np.testing.assert_allclose(infer(x_0, 1.0, 0.0), x_0 + 2.0, atol=1e-5)
    np.testing.assert_allclose(infer(x_0, 1, 0), x_0 + 2.0, atol=1e-5)
    np.testing.assert_allclose(
        infer(x_0, 1.0, 0.0, cond={"offset": jnp.zeros((3,))}),
        x_0 + 2.0,
        atol=1e-5,
    )

  def test_b3_batchnorm_mutables_and_stateful_teacher_lagrangian(self):
    """Regression test for B3: BatchNorm student & stateful teacher vars."""
    teacher = _ConstantTeacherFlow(drift=1.5, stateful=True)
    x_dummy = jnp.ones((1, 4, 2))
    cond_dummy = {"offset": jnp.zeros((1,))}
    teacher_vars = teacher.init(
        jax.random.PRNGKey(0),
        x=x_dummy,
        sigma=jnp.ones((1,)),
        cond=cond_dummy,
        is_training=False,
    )

    model = flow_map.ConditionalLagrangianFlowMapModel(
        input_shape=(4, 2),
        flow_map_model=_StatefulFlowMapNet(),
        flow_model=teacher,
        params_flow=teacher_vars,  # Full variable dict containing batch_stats.
        interpolant=flow_matching.RectifiedFlow(),
        cond_shape={"offset": ()},
        number_of_eval_steps=(1, 2),
    )
    variables = model.initialize(jax.random.PRNGKey(1))
    params = variables["params"]
    mutables = {"batch_stats": variables["batch_stats"]}
    batch = {
        "x_0": jnp.full((3, 4, 2), 4.0),
        "x_1": jnp.full((3, 4, 2), 6.0),
        "offset": jnp.asarray([0.5, -0.5, 1.0]),
    }

    (loss, (_, updated_mut)), grads = jax.value_and_grad(
        model.loss_fn, has_aux=True
    )(params, batch, jax.random.PRNGKey(2), mutables)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(grads["w"]))))
    # Running mean must advance from initial 0 toward batch mean (~5).
    old_mean = mutables["batch_stats"]["BatchNorm_0"]["mean"]
    new_mean = updated_mut["batch_stats"]["BatchNorm_0"]["mean"]
    self.assertGreater(float(jnp.max(jnp.abs(new_mean - old_mean))), 1e-3)

    eval_out = model.eval_fn(
        {"params": params, **updated_mut}, batch, jax.random.PRNGKey(3)
    )
    self.assertIn("eval_loss", eval_out)
    self.assertEqual(eval_out["1"].shape, (3, 4, 2))

    infer = model.inference_fn(
        {"params": params, **updated_mut}, model.flow_map_model
    )
    out = infer(batch["x_0"], 1.0, 0.0, cond={"offset": batch["offset"]})
    self.assertEqual(out.shape, (3, 4, 2))

  def test_conditional_lagrangian_cond_shape_none(self):
    teacher = _ConstantTeacherFlow(drift=1.0)
    teacher_vars = teacher.init(
        jax.random.PRNGKey(0),
        x=jnp.ones((1, 4, 1)),
        sigma=jnp.ones((1,)),
        cond=None,
        is_training=False,
    )
    model = flow_map.ConditionalLagrangianFlowMapModel(
        input_shape=(4, 1),
        flow_map_model=_ExactLinearFlowMap(drift=1.0),
        flow_model=teacher,
        params_flow=teacher_vars["params"],
        interpolant=flow_matching.RectifiedFlow(),
        cond_shape=None,
    )
    variables = model.initialize(jax.random.PRNGKey(1))
    batch = {"x_0": jnp.zeros((2, 4, 1)), "x_1": jnp.ones((2, 4, 1))}
    loss, _ = model.loss_fn(
        variables["params"], batch, jax.random.PRNGKey(2), {}
    )
    self.assertAlmostEqual(float(loss), 0.0, places=6)

  def test_lsd_closed_form_zero_loss_oracle_and_validation(self):
    """Exact constant-drift RectifiedFlow gives zero `loss_v` and `loss_x`."""
    model = flow_map.LagrangianSelfDistilledFlowMapModel(
        input_shape=(4, 1),
        mean_flow_model=_ExactMeanVelocityNet(drift=2.0),
        interpolant=flow_matching.RectifiedFlow(),
        number_of_eval_steps=(1, 2),
        weight_v=0.5,
    )
    variables = model.initialize(jax.random.PRNGKey(0))
    x_0 = jnp.ones((3, 4, 1))
    batch = {"x_0": x_0, "x_1": x_0 + 2.0}
    loss, (metrics, _) = model.loss_fn(
        variables["params"], batch, jax.random.PRNGKey(1), {}
    )
    self.assertAlmostEqual(float(loss), 0.0, places=6)
    self.assertAlmostEqual(float(metrics["loss_v"]), 0.0, places=6)
    self.assertAlmostEqual(float(metrics["loss_x"]), 0.0, places=6)

    eval_out = model.eval_fn(variables, batch, jax.random.PRNGKey(2))
    self.assertAlmostEqual(float(eval_out["eval_loss"]), 0.0, places=6)
    np.testing.assert_allclose(eval_out["1"], x_0 + 2.0, atol=1e-5)
    np.testing.assert_allclose(eval_out["2"], x_0 + 2.0, atol=1e-5)

    infer = model.inference_fn(variables, model.mean_flow_model)
    np.testing.assert_allclose(infer(x_0, 1.0, 0.0), x_0 + 2.0, atol=1e-5)
    np.testing.assert_allclose(infer(x_0, 1, 0), x_0 + 2.0, atol=1e-5)
    np.testing.assert_allclose(
        infer(x_0, 1.0, 0.0, cond={"offset": jnp.zeros((3,))}),
        x_0 + 2.0,
        atol=1e-5,
    )

    with self.assertRaisesRegex(ValueError, "between 0 and 1"):
      flow_map.LagrangianSelfDistilledFlowMapModel(
          input_shape=(4, 1),
          mean_flow_model=_ExactMeanVelocityNet(),
          interpolant=flow_matching.RectifiedFlow(),
          weight_v=1.5,
      )

  @parameterized.named_parameters(
      dict(testcase_name="weight_v_0", weight_v=0.0, per_sample=False),
      dict(
          testcase_name="weight_v_half_scalar", weight_v=0.5, per_sample=False
      ),
      dict(
          testcase_name="weight_v_half_persample", weight_v=0.5, per_sample=True
      ),
      dict(testcase_name="weight_v_1", weight_v=1.0, per_sample=True),
  )
  def test_b3_lsd_batchnorm_mutables_and_adaptive_norm_all_regimes(
      self, weight_v: float, per_sample: bool
  ):
    """Regression test for B3: LSD BatchNorm mutables across all `weight_v`."""
    model = flow_map.ConditionalLagrangianSelfDistilledFlowMapModel(
        input_shape=(4, 2),
        mean_flow_model=_ExactMeanVelocityNet(drift=1.0, stateful=True),
        interpolant=flow_matching.RectifiedFlow(),
        cond_shape={"offset": ()},
        weight_v=weight_v,
        use_adaptive_norm=True,
        per_sample_adaptive_norm=per_sample,
    )
    variables = model.initialize(jax.random.PRNGKey(0))
    params = variables["params"]
    mutables = {"batch_stats": variables["batch_stats"]}
    batch = {
        "x_0": jnp.full((3, 4, 2), 3.0),
        "x_1": jnp.full((3, 4, 2), 5.0),
        "offset": jnp.asarray([0.2, -0.1, 0.4]),
    }

    (loss, (metrics, updated_mut)), grads = jax.value_and_grad(
        model.loss_fn, has_aux=True
    )(params, batch, jax.random.PRNGKey(1), mutables)
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertIn("loss_v", metrics)
    self.assertIn("loss_x", metrics)
    self.assertTrue(bool(jnp.isfinite(grads["scale"])))

    old_mean = mutables["batch_stats"]["BatchNorm_0"]["mean"]
    new_mean = updated_mut["batch_stats"]["BatchNorm_0"]["mean"]
    self.assertGreater(float(jnp.max(jnp.abs(new_mean - old_mean))), 1e-3)

    infer = model.inference_fn(
        {"params": params, **updated_mut}, model.mean_flow_model
    )
    out = infer(batch["x_0"], 1.0, 0.0, cond={"offset": batch["offset"]})
    self.assertEqual(out.shape, (3, 4, 2))

  def test_lsd_stop_gradient_target_changes_gradient(self):
    """Verifies opt-in `stop_gradient_target` blocks gradient into target."""
    net = _ExactMeanVelocityNet(drift=1.0)
    common = dict(
        input_shape=(4, 1),
        mean_flow_model=net,
        interpolant=flow_matching.RectifiedFlow(),
        weight_v=0.0,
    )
    model_nosg = flow_map.ConditionalLagrangianSelfDistilledFlowMapModel(
        **common, cond_shape=None, stop_gradient_target=False
    )
    model_sg = flow_map.ConditionalLagrangianSelfDistilledFlowMapModel(
        **common, cond_shape=None, stop_gradient_target=True
    )
    variables = model_sg.initialize(jax.random.PRNGKey(0))
    params = {"scale": jnp.asarray(1.5), "slope": jnp.asarray(0.4)}
    batch = {
        "x_0": jnp.ones((3, 4, 1)),
        "x_1": jnp.full((3, 4, 1), 3.0),
    }

    grad_nosg = jax.grad(model_nosg.loss_fn, has_aux=True)(
        params, batch, jax.random.PRNGKey(4), {}
    )[0]
    grad_sg = jax.grad(model_sg.loss_fn, has_aux=True)(
        params, batch, jax.random.PRNGKey(4), {}
    )[0]
    self.assertFalse(
        bool(jnp.allclose(grad_nosg["slope"], grad_sg["slope"], atol=1e-4))
    )
    self.assertIsNotNone(variables)


if __name__ == "__main__":
  absltest.main()
