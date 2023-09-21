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
import numpy as np
from swirl_dynamics.lib.metrics import regression


class RegressionMetricsTest(parameterized.TestCase):

  @parameterized.parameters(
      {  # absolute error along a single axis
          "input_shape": (1, 2, 3),
          "norm_axis": 2,
          "relative": False,
          "output_shape": (1, 2),
          "mean_value": np.sqrt(3),
      },
      {  # relative error along multiple axes
          "input_shape": (1, 2, 3, 4),
          "norm_axis": (1, 2),
          "relative": True,
          "output_shape": (1, 4),
          "mean_value": 0.5,
      },
  )
  def test_l2_err_shape_and_value(
      self, input_shape, norm_axis, relative, output_shape, mean_value
  ):
    xpred = np.ones(input_shape)
    xtrue = 2 * np.ones(input_shape)
    err = regression.l2_err(
        pred=xpred, true=xtrue, norm_axis=norm_axis, relative=relative
    )
    with self.subTest(checking="shape"):
      self.assertEqual(err.shape, output_shape)

    with self.subTest(checking="value"):
      self.assertTrue(np.allclose(np.mean(err), mean_value))


if __name__ == "__main__":
  absltest.main()
