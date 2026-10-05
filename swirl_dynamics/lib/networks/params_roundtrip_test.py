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

"""Roundtrip and derivative tests for parameter-tree vectorization."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.networks import utils


class ParameterRoundtripTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('unbatched', (), 0),
      ('one_batch_axis', (2,), 1),
      ('two_batch_axes', (2, 3), 2),
      ('singleton_batch', (1,), 1),
      ('negative_axis', (2, 3), -1),
  )
  def test_restores_shapes_values_and_tree(self, batch_shape, from_axis):
    if from_axis == -1:
      shapes = [batch_shape + (4,), batch_shape + (2,)]
    else:
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
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

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


if __name__ == '__main__':
  absltest.main()
