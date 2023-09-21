# Copyright 2023 The swirl_dynamics Authors.
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
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.solvers import utils


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
    func = utils.nn_module_to_ode_dynamics(module, autonomous=True, flag=flag)
    out = func(x=x * np.ones((10,)), t=jnp.array(0), params={})
    np.testing.assert_array_equal(out, outputs * np.ones((10,)))

  @parameterized.parameters((6, 8, True, 15), (6, 8, False, 14))
  def test_module_to_non_autonomous_dynamics(self, x, t, flag, outputs):
    module = TestNonAutonomous()
    func = utils.nn_module_to_ode_dynamics(module, autonomous=False, flag=flag)
    out = func(x=x * np.ones((10,)), t=jnp.array(t), params={})
    np.testing.assert_array_equal(out, outputs * np.ones((10,)))


if __name__ == "__main__":
  absltest.main()
