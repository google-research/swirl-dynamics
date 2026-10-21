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
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import flow_matching


class _DummyMeanFlowNet(nn.Module):
  out_channels: int = 3

  @nn.compact
  def __call__(self, x_s, t, s=None, cond=None, is_training=False):
    del cond, is_training
    dt = (t - (0.0 if s is None else s)).reshape(
        (x_s.shape[0],) + (x_s.ndim - 1) * (1,)
    )
    return nn.Dense(self.out_channels)(x_s) + dt


class FlowMatchingTest(parameterized.TestCase):

  def test_rectified_flow_interpolant_and_velocity_loss(self):
    interp = flow_matching.RectifiedFlow()
    t = jnp.asarray([0.25, 0.75])
    x0 = jnp.zeros((2, 4))
    x1 = jnp.ones((2, 4))
    z = jnp.zeros((2, 4))
    xt = interp(t, x0, x1, z)
    self.assertEqual(xt.shape, (2, 4))
    loss = flow_matching.velocity_loss(x1 - x0, t, x0, x1, z, interp)
    self.assertAlmostEqual(float(loss), 0.0, places=6)

  def test_mean_flow_model_multichannel(self):
    net = _DummyMeanFlowNet(out_channels=3)
    model = flow_matching.MeanFlowModel(
        input_shape=(4, 4, 3),
        mean_flow_model=net,
    )
    rng = jax.random.PRNGKey(0)
    variables = model.initialize(rng)
    batch = {
        "x_0": jnp.zeros((2, 4, 4, 3)),
        "x_1": jnp.ones((2, 4, 4, 3)),
    }
    loss, (metrics, _) = jax.jit(model.loss_fn)(
        variables["params"], batch, rng, {}
    )
    self.assertTrue(bool(jnp.isfinite(loss)))
    self.assertIn("raw_mse", metrics)


if __name__ == "__main__":
  absltest.main()
