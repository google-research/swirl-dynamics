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

"""Tests Hyper U-nets used to capture the dynamics."""

from absl.testing import absltest
import jax
import numpy as np
from swirl_dynamics.lib.networks import hyper_unet
from swirl_dynamics.lib.networks import utils


class HyperUNetTest(absltest.TestCase):

  def test_number_params(self):
    # Testing that the network has the correct number of parameters.

    flat_layer_shapes = (10, 10, 10)
    embed_dims = (1, 1, 1)
    hyper_unet_test = hyper_unet.HyperUnet(
        flat_layer_shapes=flat_layer_shapes, embed_dims=embed_dims
    )

    # Initializing the network and comparing the number of parameters.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (sum(flat_layer_shapes),))
    output, params = hyper_unet_test.init_with_output(rng, x)
    num_params = utils.flat_dim(params['params'])
    np.testing.assert_equal(num_params, 488)
    np.testing.assert_array_equal(output.shape, (sum(flat_layer_shapes),))


if __name__ == '__main__':
  absltest.main()
