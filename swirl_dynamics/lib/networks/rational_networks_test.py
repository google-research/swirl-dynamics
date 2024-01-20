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

"""Tests for the rational networks and other functions."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.networks import rational_networks
from swirl_dynamics.lib.networks import utils


class RationalNetworksTest(absltest.TestCase):

  def test_rational_default_init(self):
    """Function to test that the rational networks are properly initialized."""
    rat_net = rational_networks.RationalLayer()

    # Generating the input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (10,))
    params = rat_net.init(rng, x)['params']

    # Checking that the arrays are the same up to 1e-5.
    expected_p_params = jnp.array([1.1915, 1.5957, 0.5, 0.0218])
    expected_q_params = jnp.array([2.383, 0.0, 1.0])
    self.assertSequenceAlmostEqual(
        params['p_coeffs'], expected_p_params, places=5
    )
    self.assertSequenceAlmostEqual(
        params['q_coeffs'], expected_q_params, places=5
    )

  def test_unshared_rational_default_init(self):
    """Function to test that the rational networks are properly initialized."""
    rat_net = rational_networks.UnsharedRationalLayer()

    # Generating the input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (10,))
    params = rat_net.init(rng, x)['params']

    # Checking that the arrays are the same up to 1e-5.
    expected_p_params = jnp.array([1.1915, 1.5957, 0.5, 0.0218]) * jnp.ones(
        (x.shape[-1], 1)
    )
    expected_q_params = jnp.array([2.383, 0.0, 1.0]) * jnp.ones(
        (x.shape[-1], 1)
    )
    self.assertSequenceAlmostEqual(
        params['p_params'].reshape((-1,)),
        expected_p_params.reshape((-1,)),
        places=5,
    )
    self.assertSequenceAlmostEqual(
        params['q_params'].reshape((-1,)),
        expected_q_params.reshape((-1,)),
        places=5,
    )


class RationalMLPTest(absltest.TestCase):

  def test_number_params(self):
    """Tests that the network has the correct number of parameters."""
    features = (2, 2)
    periodic_mlp_small = rational_networks.RationalMLP(features=features)

    # Generating an input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1,))
    params = periodic_mlp_small.init(rng, x)['params']

    # Computing the number of parameters and checking they are correct.
    num_params = utils.flat_dim(params)
    self.assertEqual(num_params, 17)


if __name__ == '__main__':
  absltest.main()
