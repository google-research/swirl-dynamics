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

"""Tests for the utilities for the model libraries."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.networks import utils

jax.config.update('jax_enable_x64', True)


class UtilsTest(parameterized.TestCase):

  def test_flatten_params_with_toy_example(self):
    """Test the flatten and unflatten functions with a toy example."""
    model_vars = flax.core.freeze(
        {
            'params': {
                'dense': {'bias': jnp.zeros(5), 'kernel': jnp.ones((5, 5))}
            }
        }
    )

    n_params = utils.flat_dim(model_vars['params'])
    self.assertEqual(n_params, 30)

    params_flat, shapes, tree_def = utils.flatten_params(model_vars['params'])

    params_flat_ref = jnp.concatenate(
        [jnp.zeros(5), jnp.ones((5, 5)).reshape((-1,))]
    )
    np.testing.assert_array_equal(params_flat, params_flat_ref)

    shapes_ref = [(5,), (5, 5)]
    self.assertEqual(shapes, shapes_ref)

    model_params = utils.unflatten_params(params_flat, shapes, tree_def)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, model_params, model_vars['params']
    )

  def test_negative_from_axis_raises(self):
    with self.assertRaisesRegex(ValueError, 'must be non-negative'):
      utils.flatten_params({'w': jnp.ones((2, 3))}, from_axis=-1)

  @parameterized.named_parameters(
      ('unbatched', (), 0),
      ('one_batch_axis', (2,), 1),
      ('two_batch_axes', (2, 3), 2),
      ('singleton_batch', (1,), 1),
      ('empty_batch', (0,), 1),
  )
  def test_restores_shapes_values_and_tree(self, batch_shape, from_axis):
    shapes = [batch_shape + (3, 4), batch_shape + (2,)]
    tree = {
        'layer': {
            'weight': (
                jnp.arange(np.prod(shapes[0]), dtype=jnp.float32).reshape(
                    shapes[0]
                )
            )
        },
        'bias': (
            jnp.arange(np.prod(shapes[1]), dtype=jnp.float32).reshape(
                shapes[1]
            ),
        ),
    }
    flattened, original_shapes, structure = utils.flatten_params(
        tree, from_axis=from_axis
    )
    for restore in (
        utils.unflatten_params,
        jax.jit(utils.unflatten_params, static_argnums=(1, 2)),
    ):
      recovered = restore(flattened, tuple(original_shapes), structure)
      self.assertEqual(jax.tree_util.tree_structure(recovered), structure)
      for expected, actual in zip(
          jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(recovered)
      ):
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)

  @parameterized.parameters(1, 2)
  def test_batched_roundtrip_preserves_gradients(self, from_axis):
    batch = (2,) if from_axis == 1 else (2, 3)
    tree = {
        'w': (
            jnp.arange(np.prod(batch) * 6, dtype=jnp.float32).reshape(
                batch + (2, 3)
            )
        ),
        'b': jnp.ones(batch + (2,)),
    }
    vector, shapes, structure = utils.flatten_params(tree, from_axis=from_axis)

    def objective(values):
      restored = utils.unflatten_params(values, shapes, structure)
      return sum(
          jnp.square(leaf).sum() for leaf in jax.tree_util.tree_leaves(restored)
      )

    for grad in (jax.grad(objective), jax.jit(jax.grad(objective))):
      np.testing.assert_allclose(grad(vector), 2 * vector, rtol=0, atol=0)

  def test_vmapped_unbatched_restoration_matches_batched_restoration(self):
    tree = {
        'w': jnp.arange(24.0).reshape(2, 3, 4),
        'b': jnp.arange(6.0).reshape(2, 3),
    }
    vector, shapes, structure = utils.flatten_params(tree, from_axis=1)
    unbatched_shapes = [shape[1:] for shape in shapes]
    mapped = jax.vmap(
        lambda values: utils.unflatten_params(
            values, unbatched_shapes, structure
        )
    )(vector)
    direct = utils.unflatten_params(vector, shapes, structure)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(mapped), jax.tree_util.tree_leaves(direct)
    ):
      np.testing.assert_array_equal(actual, expected)

  def test_batched_flax_parameters_preserve_model_predictions(self):
    model = nn.Dense(features=2)
    inputs = jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    variables = jax.vmap(model.init)(keys, inputs)
    flattened, shapes, structure = utils.flatten_params(variables, from_axis=1)
    reconstructed = utils.unflatten_params(flattened, shapes, structure)
    expected = jax.vmap(model.apply)(variables, inputs)
    actual = jax.jit(jax.vmap(model.apply))(reconstructed, inputs)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-7
    )

  def test_unbatched_scalars_and_zero_sized_leaves(self):
    tree = {
        'scalar': jnp.array(2.0),
        'empty': jnp.empty((0, 3)),
        'v': jnp.array([1.0, 2.0]),
    }
    vector, shapes, structure = utils.flatten_params(tree)
    recovered = utils.unflatten_params(vector, shapes, structure)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(recovered)
    ):
      np.testing.assert_array_equal(actual, expected)

  def test_vmean_toy_example(self):
    """Test the vmean function with a random toy example."""
    rngs = jax.random.split(jax.random.PRNGKey(0), (10))
    model_vars_vmapped = jax.vmap(self.rand_model_vars, in_axes=(0, None))(
        rngs, 2
    )
    model_vars_mean = utils.vmean(model_vars_vmapped)
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
    jax.tree_util.tree_map(
        np.testing.assert_array_almost_equal, model_vars_mean, model_vars_ref
    )

  def rand_model_vars(self, rng, n_size):
    """Function to create a random simple parameters dictionary."""
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
