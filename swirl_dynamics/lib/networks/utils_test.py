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

"""Tests for the utilities for the model libraries."""
from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.networks import utils

jax.config.update('jax_enable_x64', True)


class UtilsTest(absltest.TestCase):

  def test_flatten_params_with_toy_example(self):
    """Test the flatten and unflatten functions with a toy example."""
    # Creating a dummy parameter for a simple network.
    model_vars = flax.core.freeze(
        {
            'params': {
                'dense': {'bias': jnp.zeros(5), 'kernel': jnp.ones((5, 5))}
            }
        }
    )

    # Computing number of parameters.
    n_params = utils.flat_dim(model_vars['params'])
    # Checking that the number of parameters is correct.
    self.assertEqual(n_params, 30)

    params_flat, shapes, tree_def = utils.flatten_params(model_vars['params'])

    # Checking the parameters flat are correctly handled.
    params_flat_ref = jnp.concatenate(
        [jnp.zeros(5), jnp.ones((5, 5)).reshape((-1,))]
    )
    np.testing.assert_array_equal(params_flat, params_flat_ref)

    # Checking that the shapes have been correctly computed.
    shapes_ref = [(5,), (5, 5)]
    self.assertEqual(shapes, shapes_ref)

    # Checking that we can reconstruct the parameters frozen dictionary.
    model_params = utils.unflatten_params(params_flat, shapes, tree_def)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, model_params, model_vars['params']
    )

  def test_vmean_toy_example(self):
    """Test the vmean function with a random toy example."""

    # Defining a small set of random seeds.
    rngs = jax.random.split(jax.random.PRNGKey(0), (10))

    # Creating a set for random parameters.
    model_vars_vmapped = jax.vmap(self.rand_model_vars, in_axes=(0, None))(
        rngs, 2
    )

    # Computing the mean along the vmap.
    model_vars_mean = utils.vmean(model_vars_vmapped)

    # Computing the reference mean parameters.
    model_vars_ref = flax.core.freeze(
        {
            'params': {
                'dense': {
                    'bias': jnp.array(
                        jnp.mean(
                            jax.vmap(jax.random.normal, in_axes=(0, None))(
                                rngs, (2,)
                            ),
                            axis=0,
                        )
                    ),
                    'kernel': jnp.array(
                        jnp.mean(
                            jax.vmap(jax.random.normal, in_axes=(0, None))(
                                rngs, (2, 2)
                            ),
                            axis=0,
                        )
                    ),
                }
            }
        }
    )

    # Checking the result.
    jax.tree_util.tree_map(
        np.testing.assert_array_almost_equal, model_vars_mean, model_vars_ref
    )

  def rand_model_vars(self, rng, n_size):
    """Function to create a random simple parameters dictionary.

    Args:
      rng: random seed.
      n_size: internal size of the kernel and bias.

    Returns:
      A small FrozenDict with one kernel matriz of size (n_size, n_size),
      and a bias vector of size (n_size,).
    """
    model_vars = flax.core.freeze(
        {
            'params': {
                'dense': {
                    'bias': jax.random.normal(rng, (n_size,)),
                    'kernel': jax.random.normal(rng, (n_size, n_size)),
                }
            }
        }
    )
    return model_vars


if __name__ == '__main__':
  absltest.main()
