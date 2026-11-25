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

"""Unit tests for `swirl_dynamics.lib.flow_map.mean_flow`."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import flow_map
from swirl_dynamics.lib import flow_matching


class _LinearOdeTeacher(nn.Module):
  """Exact linear ODE velocity `v(x, t) = rate * x` (optionally stateful)."""

  rate: float = 0.5
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
      _ = nn.BatchNorm(use_running_average=not is_training)(h)
    rate = self.rate * scale
    if cond is not None and "bias" in cond:
      return rate * x + cond["bias"].reshape(
          (x.shape[0],) + (1,) * (x.ndim - 1)
      )
    return rate * x


class _ExactLinearOdeMeanFlow(nn.Module):
  """Exact average velocity `((1 - exp(-a * (t - s))) / (t - s)) * x_t`."""

  rate: float = 0.5

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
    del cond, is_training
    scale = self.param("scale", nn.initializers.ones, ())
    a = self.rate * scale
    dt = (t - s).reshape((x_s.shape[0],) + (1,) * (x_s.ndim - 1))
    # Smooth Taylor fallback near dt = 0 avoids 0/0 in forward-mode JVP.
    small = jnp.abs(dt) < 1e-4
    safe_dt = jnp.where(small, 1.0, dt)
    factor = jnp.where(
        small,
        a * (1.0 - 0.5 * a * dt + (a * dt) ** 2 / 6.0),
        -jnp.expm1(-a * safe_dt) / safe_dt,
    )
    return factor * x_s


class _StatefulMeanFlowNet(nn.Module):
  """BatchNorm mean-flow network verifying mutable state propagation."""

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
    w = self.param("w", nn.initializers.ones, ())
    dt = (t - s).reshape((x_s.shape[0],) + (1,) * (x_s.ndim - 1))
    out = w * h + 0.1 * dt
    if cond is not None and "bias" in cond:
      out = out + cond["bias"].reshape((x_s.shape[0],) + (1,) * (x_s.ndim - 1))
    return out


def _make_exponential_interpolant(
    rate: float = 0.5,
) -> flow_matching.StochasticInterpolant:
  """Deterministic exponential flow `x_t = exp(a * t) * x_0` for ODE oracles."""
  return flow_matching.StochasticInterpolant(
      alpha=lambda t: jnp.exp(rate * t),
      alpha_dot=lambda t: rate * jnp.exp(rate * t),
      beta=jnp.zeros_like,
      beta_dot=jnp.zeros_like,
      gamma=jnp.zeros_like,
      gamma_dot=jnp.zeros_like,
  )


class MeanFlowModelsTest(parameterized.TestCase):

  def test_all_38_public_exports_sorted(self):
    self.assertEqual(flow_map.__all__, sorted(flow_map.__all__))
    self.assertLen(flow_map.__all__, 38)
    for name in flow_map.__all__:
      self.assertTrue(hasattr(flow_map, name), f"Missing export: {name}")

  def test_mean_flow_and_symmetric_linear_ode_zero_loss_oracle(self):
    """Exact linear ODE `dx/dt = 0.5 * x` gives zero MF and SymmetricMF loss."""
    rate = 0.5
    teacher = _LinearOdeTeacher(rate=rate)
    teacher_vars = teacher.init(
        jax.random.PRNGKey(0),
        x=jnp.ones((1, 4, 1)),
        sigma=jnp.ones((1,)),
        is_training=False,
    )
    interp = _make_exponential_interpolant(rate=rate)
    x_0 = jnp.asarray([[[1.0], [-2.0], [0.5], [1.5]]] * 3, dtype=jnp.float32)
    x_1 = x_0 * jnp.exp(rate)
    batch = {"x_0": x_0, "x_1": x_1}

    # Unconditional MeanFlowModel with teacher distillation.
    mf_model = flow_map.MeanFlowModel(
        input_shape=(4, 1),
        mean_flow_model=_ExactLinearOdeMeanFlow(rate=rate),
        flow_model=teacher,
        params_flow=teacher_vars["params"],
        interpolant=interp,
    )
    mf_vars = mf_model.initialize(jax.random.PRNGKey(1))
    loss, (metrics, _) = mf_model.loss_fn(
        mf_vars["params"], batch, jax.random.PRNGKey(2), {}
    )
    self.assertAlmostEqual(float(loss), 0.0, places=5)
    self.assertAlmostEqual(float(metrics["raw_mse"]), 0.0, places=5)

    eval_out = mf_model.eval_fn(mf_vars, batch, jax.random.PRNGKey(3))
    self.assertAlmostEqual(float(eval_out["eval_loss"]), 0.0, places=5)
    np.testing.assert_allclose(eval_out["eval_samples"], x_0, atol=1e-4)

    infer = mf_model.inference_fn(mf_vars, mf_model.mean_flow_model)
    samples_k = flow_map.sample_backward_mean_flow(infer, x_1, num_steps=4)
    np.testing.assert_allclose(samples_k, x_0, rtol=1e-4, atol=1e-4)
    # Optional `cond` and integer times work through MeanFlowModel inference.
    u_with_cond = infer(x_1, 1, 0, cond={"bias": jnp.zeros((3,))})
    np.testing.assert_allclose(x_1 - u_with_cond, x_0, rtol=1e-4, atol=1e-4)

    # ConditionalSymmetricMeanFlowModel with exact linear ODE satisfies both
    # t-direction and s-direction identities simultaneously.
    sym_model = flow_map.ConditionalSymmetricMeanFlowModel(
        input_shape=(4, 1),
        mean_flow_model=_ExactLinearOdeMeanFlow(rate=rate),
        flow_model=teacher,
        params_flow=teacher_vars,
        interpolant=interp,
        cond_shape=None,
    )
    sym_vars = sym_model.initialize(jax.random.PRNGKey(4))
    sym_loss, (sym_metrics, _) = sym_model.loss_fn(
        sym_vars["params"], batch, jax.random.PRNGKey(5), {}
    )
    self.assertAlmostEqual(float(sym_loss), 0.0, places=5)
    self.assertAlmostEqual(float(sym_metrics["loss_t"]), 0.0, places=5)
    self.assertAlmostEqual(float(sym_metrics["loss_s"]), 0.0, places=5)

  def test_batchnorm_mutables_teacher_and_validation(self):
    teacher = _LinearOdeTeacher(rate=0.5, stateful=True)
    cond_dummy = {"bias": jnp.zeros((1,))}
    teacher_vars = teacher.init(
        jax.random.PRNGKey(0),
        x=jnp.ones((1, 4, 2)),
        sigma=jnp.ones((1,)),
        cond=cond_dummy,
        is_training=False,
    )
    batch = {
        "x_0": jnp.full((3, 4, 2), 2.0),
        "x_1": jnp.full((3, 4, 2), 5.0),
        "bias": jnp.asarray([0.1, -0.2, 0.3]),
    }

    for model_cls in (
        flow_map.ConditionalMeanFlowModel,
        flow_map.ConditionalSymmetricMeanFlowModel,
    ):
      with self.subTest(model=model_cls.__name__):
        model = model_cls(
            input_shape=(4, 2),
            mean_flow_model=_StatefulMeanFlowNet(),
            flow_model=teacher,
            params_flow=teacher_vars,
            cond_shape={"bias": ()},
        )
        variables = model.initialize(jax.random.PRNGKey(1))
        params = variables["params"]
        mutables = {"batch_stats": variables["batch_stats"]}
        (loss, (_, updated_mut)), grads = jax.value_and_grad(
            model.loss_fn, has_aux=True
        )(params, batch, jax.random.PRNGKey(2), mutables)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(jnp.isfinite(grads["w"])))
        old_mean = mutables["batch_stats"]["BatchNorm_0"]["mean"]
        new_mean = updated_mut["batch_stats"]["BatchNorm_0"]["mean"]
        self.assertGreater(float(jnp.max(jnp.abs(new_mean - old_mean))), 1e-3)

        infer = model.inference_fn(
            {"params": params, **updated_mut}, model.mean_flow_model
        )
        out = infer(batch["x_1"], 1.0, 0.0, cond={"bias": batch["bias"]})
        self.assertEqual(out.shape, (3, 4, 2))

    # Direct interpolant velocity (no teacher) + missing params_flow ValueError.
    no_teacher = flow_map.MeanFlowModel(
        input_shape=(4, 2),
        mean_flow_model=_StatefulMeanFlowNet(),
    )
    nt_vars = no_teacher.initialize(jax.random.PRNGKey(3))
    nt_loss, _ = no_teacher.loss_fn(
        nt_vars["params"],
        batch,
        jax.random.PRNGKey(4),
        {"batch_stats": nt_vars["batch_stats"]},
    )
    self.assertTrue(bool(jnp.isfinite(nt_loss)))

    bad_teacher = flow_map.MeanFlowModel(
        input_shape=(4, 2),
        mean_flow_model=_StatefulMeanFlowNet(),
        flow_model=teacher,
        params_flow=None,
    )
    with self.assertRaisesRegex(ValueError, "params_flow"):
      bad_teacher.loss_fn(
          nt_vars["params"],
          batch,
          jax.random.PRNGKey(4),
          {"batch_stats": nt_vars["batch_stats"]},
      )

  def test_mixed_precision_bfloat16_jvp_tangent_safety(self):
    """Mixed bfloat16 inputs and float32 teacher velocity produce finite JVP."""
    teacher = _LinearOdeTeacher(rate=0.5)
    teacher_vars = teacher.init(
        jax.random.PRNGKey(0),
        x=jnp.ones((1, 4, 2), dtype=jnp.float32),
        sigma=jnp.ones((1,), dtype=jnp.float32),
        is_training=False,
    )
    model = flow_map.MeanFlowModel(
        input_shape=(4, 2),
        mean_flow_model=flow_map.RescaledMeanFlowUNet(
            out_channels=2,
            num_channels=(4,),
            downsample_ratio=(2,),
            num_blocks=1,
            noise_embed_dim=8,
            use_attention=False,
            use_position_encoding=False,
            dtype=jnp.bfloat16,
        ),
        flow_model=teacher,
        params_flow=teacher_vars["params"],
    )
    variables = model.initialize(jax.random.PRNGKey(1))
    batch_bf16 = {
        "x_0": jnp.zeros((2, 4, 2), dtype=jnp.bfloat16),
        "x_1": jnp.ones((2, 4, 2), dtype=jnp.bfloat16),
    }
    (loss, _), grads = jax.value_and_grad(model.loss_fn, has_aux=True)(
        variables["params"], batch_bf16, jax.random.PRNGKey(2), {}
    )
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertTrue(
        all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))
    )
    eval_out = model.eval_fn(variables, batch_bf16, jax.random.PRNGKey(3))
    self.assertTrue(bool(jnp.isfinite(eval_out["eval_loss"])))


if __name__ == "__main__":
  absltest.main()
