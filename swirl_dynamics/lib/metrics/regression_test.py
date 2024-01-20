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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from swirl_dynamics.lib.metrics import regression as reg

_TEST_INPUT_SHAPE = (1, 2, 3, 4)


class RegressionMetricsTest(parameterized.TestCase):

  @parameterized.parameters(
      (reg.mean_squared_error, (1,), (2,), (1, 4)),
      (reg.mean_squared_error, (), (2,), (1, 2, 4)),
      (reg.mean_squared_error, (3,), None, ()),
      (reg.mean_absolute_error, (1,), (2,), (1, 4)),
      (reg.mean_absolute_error, (), (2,), (1, 2, 4)),
      (reg.mean_absolute_error, (3,), None, ()),
  )
  def test_mse_and_mae_shapes(self, metric, sum_axes, mean_axes, output_shape):
    xpred = np.ones(_TEST_INPUT_SHAPE)
    xtrue = 2 * np.ones(_TEST_INPUT_SHAPE)
    err = metric(pred=xpred, true=xtrue, sum_axes=sum_axes, mean_axes=mean_axes)
    self.assertEqual(err.shape, output_shape)

  @parameterized.parameters(
      (False, True, 4.0),
      (False, False, 2.0),
      (True, True, 0.25),
      (True, False, 0.5),
  )
  def test_mse_values(self, relative, squared, expected):
    xpred = np.ones(_TEST_INPUT_SHAPE)
    xtrue = 2 * np.ones(_TEST_INPUT_SHAPE)
    err = reg.mean_squared_error(
        pred=xpred,
        true=xtrue,
        sum_axes=(-1,),
        relative=relative,
        squared=squared,
    )
    self.assertAlmostEqual(err, expected, places=5)

  @parameterized.parameters((False, 12.0), (True, 0.75))
  def test_mae_values(self, relative, expected):
    xpred = np.ones(_TEST_INPUT_SHAPE)
    xtrue = 4 * np.ones(_TEST_INPUT_SHAPE)
    err = reg.mean_absolute_error(
        pred=xpred, true=xtrue, sum_axes=(-1,), relative=relative
    )
    self.assertAlmostEqual(err, expected, places=5)


if __name__ == "__main__":
  absltest.main()
