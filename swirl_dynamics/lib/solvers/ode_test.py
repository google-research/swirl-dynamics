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
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.solvers import ode


class TestAutonomous(nn.Module):

  @nn.compact
  def __call__(self, x, flag=True):
    return x + 1 if flag else x


class TestNonAutonomous(nn.Module):

  @nn.compact
  def __call__(self, x, t, flag=True):
    return x + t + 1 if flag else x + t


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters((5, True, 6), (5, False, 5))
  def test_module_to_autonomous_dynamics(self, x, flag, outputs):
    module = TestAutonomous()
    func = ode.nn_module_to_dynamics(module, autonomous=True, flag=flag)
    out = func(x=x * np.ones((10,)), t=jnp.array(0), params={})
    np.testing.assert_array_equal(out, outputs * np.ones((10,)))

  @parameterized.parameters((6, 8, True, 15), (6, 8, False, 14))
  def test_module_to_non_autonomous_dynamics(self, x, t, flag, outputs):
    module = TestNonAutonomous()
    func = ode.nn_module_to_dynamics(module, autonomous=False, flag=flag)
    out = func(x=x * np.ones((10,)), t=jnp.array(t), params={})
    np.testing.assert_array_equal(out, outputs * np.ones((10,)))


def dummy_ode_dynamics(x, t, params):
  del t, params
  return jnp.ones_like(x)


def dummy_ode_dynamics_return_original_num_channel(x, t, params, num_channels):
  """Assumes ode dynamics returns channel dim of 1."""
  del t, params
  return jnp.ones(x.shape[:-1] + (num_channels,))


class OdeSolversTest(parameterized.TestCase):

  @parameterized.parameters(
      {"solver": ode.ExplicitEuler(), "backward": False},
      {"solver": ode.ExplicitEuler(), "backward": True},
      {"solver": ode.HeunsMethod(), "backward": False},
      {"solver": ode.HeunsMethod(), "backward": True},
      {"solver": ode.RungeKutta4(), "backward": False},
      {"solver": ode.RungeKutta4(), "backward": True},
      {"solver": ode.DoPri45(), "backward": False},
  )
  def test_output_shape_and_value(self, solver, backward):
    dt = 0.1
    num_steps = 10
    x_dim = 5
    sign = -1 if backward else 1
    tspan = jnp.arange(num_steps) * dt * sign
    out = solver(dummy_ode_dynamics, jnp.zeros((x_dim,)), tspan, {})
    self.assertEqual(out.shape, (num_steps, x_dim))
    np.testing.assert_allclose(out[-1], np.ones((x_dim,)) * tspan[-1])

  def test_output_shape_and_value_one_step_direct(self):
    dt = 0.1
    num_steps = 10
    x_dim = 5
    tspan = jnp.arange(num_steps) * dt
    solver = ode.OneStepDirect()
    out = solver(dummy_ode_dynamics, jnp.zeros((x_dim,)), tspan, {})
    self.assertEqual(out.shape, (num_steps, x_dim))
    np.testing.assert_array_equal(out[-1], np.ones((x_dim,)))

  @parameterized.parameters(
      {"time_axis_pos": 1, "x_dim": (5,)},
      {"time_axis_pos": 2, "x_dim": (5, 5)},
  )
  def test_move_time_axis_pos(self, time_axis_pos, x_dim):
    dt = 0.1
    num_steps = 10
    batch_sz = 6
    input_shape = (batch_sz,) + x_dim
    tspan = jnp.arange(num_steps) * dt
    solver = ode.ExplicitEuler(time_axis_pos=time_axis_pos)
    expected_output_shape = (
        input_shape[:time_axis_pos] + (num_steps,) + input_shape[time_axis_pos:]
    )
    out = solver(dummy_ode_dynamics, jnp.zeros(input_shape), tspan, {})
    self.assertEqual(out.shape, expected_output_shape)
    np.testing.assert_allclose(
        jnp.moveaxis(out, time_axis_pos, 1)[:, -1],
        np.ones(input_shape) * tspan[-1],
    )

  @parameterized.parameters(
      {"time_axis_pos": 1, "x_dim": (5,)},
      {"time_axis_pos": 2, "x_dim": (5, 5)},
  )
  def test_move_time_axis_pos_one_step_direct(self, time_axis_pos, x_dim):
    dt = 0.1
    num_steps = 10
    batch_sz = 6
    input_shape = (batch_sz,) + x_dim
    tspan = jnp.arange(num_steps) * dt
    solver = ode.OneStepDirect(time_axis_pos=time_axis_pos)
    out = solver(dummy_ode_dynamics, jnp.zeros(input_shape), tspan, {})
    expected_output_shape = (
        input_shape[:time_axis_pos] + (num_steps,) + input_shape[time_axis_pos:]
    )
    self.assertEqual(out.shape, expected_output_shape)
    np.testing.assert_array_equal(
        jnp.moveaxis(out, time_axis_pos, 1)[:, -1], np.ones(input_shape)
    )

  @parameterized.parameters((np.arange(10) * -1,), (np.zeros(10),))
  def test_dopri45_backward_error(self, tspan):
    tspan = jnp.asarray(tspan)
    with self.assertRaises(ValueError):
      ode.DoPri45()(dummy_ode_dynamics, jnp.array(0), tspan, {})


class MultiStepOdeSolversTest(parameterized.TestCase):

  @parameterized.product(
      time_axis_pos=(0, 1, 2),
      batch_size=(1, 8),
      state_dim=((512,), (64, 64), (32, 32, 32)),
      channels=(1, 2, 3),
      num_lookback_steps=(2, 4, 8),
  )
  def test_stacked_output_shape_and_value(
      self,
      time_axis_pos,
      batch_size,
      state_dim,
      channels,
      num_lookback_steps,
  ):
    # Setup initial shape with time in axis 0
    input_shape = [num_lookback_steps, batch_size]
    input_shape.extend(state_dim)
    input_shape.append(channels)
    # Re-order shape to match time_axis_pos (for time_axis_pos=0, this is no OP)
    input_shape[0] = input_shape[time_axis_pos]
    input_shape[time_axis_pos] = num_lookback_steps

    rng = jax.random.PRNGKey(0)
    input_state = jax.random.normal(rng, input_shape)
    input_state_stacked = ode.MultiStepScanOdeSolver(
        time_axis_pos=time_axis_pos
    ).stack_timesteps_along_channel_dim(input_state)
    expected_output_shape = (
        input_shape[:time_axis_pos]
        + input_shape[time_axis_pos + 1 : -1]
        + [channels * num_lookback_steps]
    )
    # Check that expected shapes match
    self.assertEqual(input_state_stacked.shape, tuple(expected_output_shape))
    # Check that timesteps correctly concatenated along channel dim
    for w in range(num_lookback_steps):
      c_start = channels * w
      c_end = channels * (w + 1)
      np.testing.assert_array_equal(
          jnp.moveaxis(input_state, time_axis_pos, 0)[w, ...],
          input_state_stacked[..., c_start:c_end],
      )

  @parameterized.product(
      time_axis_pos=(0, 1, 2),
      batch_size=(1, 8),
      state_dim=((512,), (64, 64), (32, 32, 32)),
      channels=(1, 2, 3),
      num_lookback_steps=(2, 4, 8),
  )
  def test_output_shape_and_value_multi_step_direct(
      self,
      time_axis_pos,
      batch_size,
      state_dim,
      channels,
      num_lookback_steps,
  ):
    # Setup initial shape with time in axis 0
    input_shape = [num_lookback_steps, batch_size]
    input_shape.extend(state_dim)
    input_shape.append(channels)
    # Re-order shape to match time_axis_pos (for time_axis_pos=0, this is no OP)
    input_shape[0] = input_shape[time_axis_pos]
    input_shape[time_axis_pos] = num_lookback_steps
    dt = 0.1
    num_steps = 10
    tspan = jnp.arange(num_steps) * dt
    solver = ode.MultiStepDirect(time_axis_pos=time_axis_pos)
    out = solver(
        functools.partial(
            dummy_ode_dynamics_return_original_num_channel,
            num_channels=channels,
        ),
        jnp.zeros(input_shape),
        tspan,
        {},
    )
    expected_output_shape = (
        input_shape[:time_axis_pos]
        + [num_steps]
        + input_shape[time_axis_pos + 1 :]
    )
    out = jnp.take(
        out,
        np.arange(num_lookback_steps - 1, out.shape[time_axis_pos]),
        axis=time_axis_pos,
    )
    self.assertEqual(out.shape, tuple(expected_output_shape))
    np.testing.assert_array_equal(
        jnp.take(
            out,
            np.arange(1, out.shape[time_axis_pos]),
            axis=time_axis_pos,
        ),
        jnp.take(
            np.ones(expected_output_shape),
            np.arange(1, out.shape[time_axis_pos]),
            axis=time_axis_pos,
        ),
    )


if __name__ == "__main__":
  absltest.main()
