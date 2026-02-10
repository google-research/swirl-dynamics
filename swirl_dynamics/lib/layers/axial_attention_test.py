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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.layers import axial_attention


class AxialAttentionLayersTest(parameterized.TestCase):

  @parameterized.parameters(((8, 8), -2), ((8, 8, 8), -2), ((8, 8, 8), -3))
  def test_self_attn_output_shape(self, input_shape, axis):
    inputs = jnp.ones(input_shape)
    model = axial_attention.AxialSelfAttention(
        num_heads=4,
        attention_axis=axis,
    )
    out, _ = model.init_with_output(jax.random.PRNGKey(42), inputs=inputs)
    self.assertEqual(out.shape, input_shape)

  @parameterized.parameters(((8, 8), -2), ((8, 8, 8), -2), ((8, 8, 8), -3))
  def test_pos_embedding_output_shape(self, input_shape, axis):
    inputs = jnp.ones(input_shape)
    model = axial_attention.AddAxialPositionEmbedding(position_axis=axis)
    out, _ = model.init_with_output(jax.random.PRNGKey(42), inputs=inputs)
    self.assertEqual(out.shape, input_shape)

  @parameterized.parameters(
      (jnp.float32, jnp.float32),
      (jnp.float32, jnp.bfloat16),
      (jnp.bfloat16, jnp.float32),
      (jnp.bfloat16, jnp.bfloat16),
  )
  def test_axial_self_attention_dtypes(self, dtype, param_dtype):
    input_shape = (2, 8, 8, 4)
    inputs = jnp.ones(input_shape, dtype=dtype)
    model = axial_attention.AxialSelfAttention(
        num_heads=2,
        attention_axis=-2,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    variables = model.init(jax.random.PRNGKey(42), inputs)

    # Check parameter dtypes
    params = variables["params"]
    for leaf in jax.tree_util.tree_leaves(params):
      self.assertEqual(leaf.dtype, param_dtype)

    # Check output dtype
    out = model.apply(variables, inputs)
    self.assertEqual(out.dtype, dtype)
    self.assertEqual(out.shape, input_shape)


if __name__ == "__main__":
  absltest.main()
