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
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.layers import residual


class ResidualLayersTest(parameterized.TestCase):

  @parameterized.parameters(((8, 8, 8, 8), True), ((8, 8, 8, 8, 8), False))
  def test_combine_res_with_skip_output_shape(self, input_shape, project_skip):
    skip = res = jnp.ones(input_shape)
    model = residual.CombineResidualWithSkip(project_skip=project_skip)
    out, var = model.init_with_output(
        jax.random.PRNGKey(42), skip=skip, residual=res
    )
    self.assertEqual(out.shape, input_shape)
    # If project_skip = False, variables should be an empty dict.
    if not project_skip:
      self.assertFalse(var)
    else:
      self.assertTrue(var)


if __name__ == "__main__":
  absltest.main()
