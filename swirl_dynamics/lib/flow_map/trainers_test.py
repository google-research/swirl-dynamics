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

"""Unit tests for `swirl_dynamics.lib.flow_map.trainers`."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from swirl_dynamics.lib import flow_map
from swirl_dynamics.lib import flow_matching


class _DummyTeacher(nn.Module):
  """Minimal linear teacher velocity for trainer tests."""

  @nn.compact
  def __call__(
      self,
      x: jax.Array,
      sigma: jax.Array,
      cond: flow_map.CondDict | None = None,
      *,
      is_training: bool,
  ) -> jax.Array:
    del sigma, cond, is_training
    return jnp.ones_like(x)


class _BatchNormFlowNet(nn.Module):
  """Minimal BatchNorm two-time network for B4 EMA & mutables tests."""

  use_batch_norm: bool = True

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
    h = x_s
    if self.use_batch_norm:
      h = nn.BatchNorm(use_running_average=not is_training)(h)
    w = self.param("w", nn.initializers.ones, ())
    b = self.param("b", nn.initializers.zeros, ())
    dt = (t - s).reshape((x_s.shape[0],) + (1,) * (x_s.ndim - 1))
    out = w * h + b + 0.5 * dt
    if cond is not None and "offset" in cond:
      out = out + cond["offset"].reshape(
          (x_s.shape[0],) + (1,) * (x_s.ndim - 1)
      )
    return out


def _make_batch(batch_size: int = 4) -> dict[str, np.ndarray]:
  return {
      "x_0": np.full((batch_size, 4, 2), 2.0, dtype=np.float32),
      "x_1": np.full((batch_size, 4, 2), 4.0, dtype=np.float32),
      "offset": np.linspace(-0.5, 0.5, batch_size, dtype=np.float32),
  }


class FlowMapTrainersTest(parameterized.TestCase):

  def test_b4_ema_variables_preserves_batchnorm_and_orbax_dict_state(self):
    """Regression test for B4: ema_variables keeps mutables & dict EMA."""
    model = flow_map.MeanFlowModel(
        input_shape=(4, 2),
        mean_flow_model=_BatchNormFlowNet(use_batch_norm=True),
    )
    trainer = flow_map.MeanFlowTrainer(
        ema_decay=0.5,
        model=model,
        rng=jax.random.PRNGKey(0),
        optimizer=optax.sgd(learning_rate=0.1),
    )
    init_params = jax.tree_util.tree_map(np.asarray, trainer.train_state.params)
    batch = _make_batch(4)
    trainer.train(iter([batch, batch]), num_steps=2)

    state = trainer.train_state
    ema_vars = state.ema_variables
    self.assertIn("params", ema_vars)
    self.assertIn("batch_stats", ema_vars)
    self.assertFalse(
        np.allclose(np.asarray(ema_vars["params"]["w"]), init_params["w"])
    )
    # BatchNorm running mean advanced from 0 and is preserved in ema_variables.
    bn_mean = ema_vars["batch_stats"]["BatchNorm_0"]["mean"]
    self.assertGreater(float(jnp.max(jnp.abs(bn_mean))), 1e-3)

    # Simulate Orbax checkpoint restore where `ema_state` is a plain dict.
    assert isinstance(state.ema_state, optax.EmaState)
    dict_state = state.replace(
        ema_state={"count": state.ema_state.count, "ema": state.ema_state.ema}
    )
    dict_ema_vars = dict_state.ema_variables
    self.assertIn("batch_stats", dict_ema_vars)
    np.testing.assert_allclose(
        dict_ema_vars["params"]["w"], ema_vars["params"]["w"]
    )

    # Inference with both use_ema=True (dict restored) and use_ema=False.
    infer_ema = flow_map.MeanFlowTrainer.inference_fn_from_state_dict(
        dict_state, model.mean_flow_model, use_ema=True
    )
    infer_raw = flow_map.MeanFlowTrainer.inference_fn_from_state_dict(
        state, model.mean_flow_model, use_ema=False
    )
    x = jnp.ones((2, 4, 2))
    self.assertEqual(infer_ema(x, 1.0, 0.0).shape, (2, 4, 2))
    self.assertEqual(infer_raw(x, 1.0, 0.0).shape, (2, 4, 2))

    # None ema_state raises ValueError; dict ema_state update raises TypeError.
    none_state = state.replace(ema_state=None)
    with self.assertRaisesRegex(ValueError, "EMA state is none"):
      _ = none_state.ema_variables
    with self.assertRaisesRegex(TypeError, "optax.EmaState"):
      trainer.update_train_state(dict_state, state.params, state.flax_mutables)

  def test_b4_lsd_trainer_inference_applies_x_plus_dt_u_with_cond(self):
    """Regression test for B4: LSD trainer delegates to LSD inference_fn."""
    net = _BatchNormFlowNet(use_batch_norm=False)
    model = flow_map.ConditionalLagrangianSelfDistilledFlowMapModel(
        input_shape=(4, 2),
        mean_flow_model=net,
        interpolant=flow_matching.RectifiedFlow(),
        cond_shape={"offset": ()},
        number_of_eval_steps=(1,),
    )
    trainer = flow_map.ConditionalLagrangianSelfDistilledFlowMapTrainer(
        ema_decay=0.9,
        model=model,
        rng=jax.random.PRNGKey(1),
        optimizer=optax.adam(1e-2),
    )
    batch = _make_batch(4)
    train_metrics = trainer.train(iter([batch]), num_steps=1)
    eval_metrics = trainer.eval(iter([batch]), num_steps=1)
    computed_train = train_metrics.compute()
    computed_eval = eval_metrics.compute()
    for key in (
        "train_loss",
        "train_loss_std",
        "train_loss_x",
        "train_loss_v",
    ):
      self.assertIn(key, computed_train)
      self.assertTrue(np.isfinite(computed_train[key]))
    self.assertIn("eval_loss", computed_eval)

    # Check exact forward map x + (t - s) * u_theta(x, t, s, cond).
    state = trainer.train_state
    lsd_trainer_cls = flow_map.LagrangianSelfDistilledFlowMapTrainer
    infer = lsd_trainer_cls.inference_fn_from_state_dict(
        state, net, use_ema=True
    )
    x = jnp.full((2, 4, 2), 3.0)
    cond = {"offset": jnp.asarray([0.25, -0.25])}
    actual = infer(x, 0.8, 0.3, cond=cond)
    raw_u = net.apply(
        state.ema_variables,
        x_s=x,
        t=jnp.asarray([0.8, 0.8]),
        s=jnp.asarray([0.3, 0.3]),
        cond=cond,
        is_training=False,
    )
    expected = x + 0.5 * jnp.asarray(raw_u)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name="lagrangian", family="lagrangian"),
      dict(testcase_name="lsd", family="lsd"),
      dict(testcase_name="mean_flow", family="mean_flow"),
  )
  def test_single_and_distributed_trainers(self, family: str):
    """Verifies single-device and `pmap` distributed train/eval steps."""
    num_devices = jax.local_device_count()
    self.assertGreaterEqual(num_devices, 1)
    batch = _make_batch(2 * num_devices)
    net = _BatchNormFlowNet(use_batch_norm=True)

    if family == "lagrangian":
      teacher = _DummyTeacher()
      teacher_vars = teacher.init(
          jax.random.PRNGKey(0),
          x=jnp.ones((1, 4, 2)),
          sigma=jnp.ones((1,)),
          is_training=False,
      )
      model = flow_map.LagrangianFlowMapModel(
          input_shape=(4, 2),
          flow_map_model=net,
          flow_model=teacher,
          params_flow=teacher_vars,
          interpolant=flow_matching.RectifiedFlow(),
          number_of_eval_steps=(1,),
      )
      single_cls = flow_map.LagrangianFlowMapTrainer
      dist_cls = flow_map.DistributedLagrangianFlowMapTrainer
    elif family == "lsd":
      model = flow_map.LagrangianSelfDistilledFlowMapModel(
          input_shape=(4, 2),
          mean_flow_model=net,
          interpolant=flow_matching.RectifiedFlow(),
          number_of_eval_steps=(1,),
      )
      single_cls = flow_map.LagrangianSelfDistilledFlowMapTrainer
      dist_cls = flow_map.DistributedLagrangianSelfDistilledFlowMapTrainer
    else:
      model = flow_map.MeanFlowModel(
          input_shape=(4, 2),
          mean_flow_model=net,
      )
      single_cls = flow_map.MeanFlowTrainer
      dist_cls = flow_map.DistributedMeanFlowTrainer

    single = single_cls(
        ema_decay=0.9,
        model=model,
        rng=jax.random.PRNGKey(10),
        optimizer=optax.adam(1e-2),
    )
    single_train = single.train(iter([batch]), num_steps=1).compute()
    single_eval = single.eval(iter([batch]), num_steps=1).compute()
    self.assertTrue(np.isfinite(single_train["train_loss"]))
    self.assertTrue(np.isfinite(single_eval["eval_loss"]))
    infer_fn = single_cls.inference_fn_from_state_dict(
        single.train_state, net, use_ema=True
    )
    self.assertEqual(infer_fn(jnp.ones((2, 4, 2)), 1.0, 0.0).shape, (2, 4, 2))

    dist = dist_cls(
        ema_decay=0.9,
        model=model,
        rng=jax.random.PRNGKey(11),
        optimizer=optax.adam(1e-2),
    )
    dist_train = dist.train(iter([batch]), num_steps=1).compute()
    dist_eval = dist.eval(iter([batch]), num_steps=1).compute()
    self.assertTrue(np.isfinite(dist_train["train_loss"]))
    self.assertTrue(np.isfinite(dist_eval["eval_loss"]))
    self.assertEqual(dist.train_state.step.shape, (num_devices,))
    unrep = flax.jax_utils.unreplicate(dist.train_state)
    self.assertEqual(int(unrep.step), 1)

  def test_base_ema_trainer_default_metrics(self):
    model = flow_map.MeanFlowModel(
        input_shape=(4, 2),
        mean_flow_model=_BatchNormFlowNet(use_batch_norm=False),
    )
    base = flow_map.EmaFlowMapTrainer(
        ema_decay=0.9,
        model=model,
        rng=jax.random.PRNGKey(20),
        optimizer=optax.sgd(0.05),
    )
    batch = _make_batch(4)
    tm = base.train(iter([batch]), num_steps=1).compute()
    em = base.eval(iter([batch]), num_steps=1).compute()
    self.assertIn("train_loss", tm)
    self.assertIn("eval_loss", em)


if __name__ == "__main__":
  absltest.main()
