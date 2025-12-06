# Copyright 2025 The swirl_dynamics Authors.
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
import numpy as np
from swirl_dynamics.lib.layers import convolutions


class ConvLayersTest(parameterized.TestCase):

  @parameterized.parameters(
      ((8, 4, 8, 8), "latlon"), ((8, 8, 8, 4, 8), "lonlat")
  )
  def test_latlon_conv_output_shape_and_equivariance(
      self, input_shape, padding
  ):
    num_features = 6
    inputs = jax.random.normal(jax.random.PRNGKey(0), input_shape)
    model = convolutions.ConvLayer(
        features=num_features, padding=padding, kernel_size=(3, 3)
    )
    out, var = model.init_with_output(jax.random.PRNGKey(42), inputs=inputs)
    self.assertEqual(out.shape, input_shape[:-1] + (num_features,))

    # Test equivariance in the longitudinal direction.
    lon_axis = -3 if padding == "lonlat" else -2
    rolled_inputs = jnp.roll(inputs, shift=3, axis=lon_axis)
    out_ = model.apply(var, inputs=rolled_inputs)
    np.testing.assert_allclose(
        jnp.roll(out, shift=3, axis=lon_axis), out_, atol=1e-6
    )

  @parameterized.parameters(
      ((8, 8, 8, 8), (2, 2), (8, 4, 4, 8)),
      ((8, 8, 8, 8, 8), (2, 2, 2, 2), (4, 4, 4, 4, 4)),
  )
  def test_downsample_conv_output_shape(
      self, input_shape, ratios, expected_output_shape
  ):
    num_features = 6
    inputs = jnp.ones(input_shape)
    model = convolutions.DownsampleConv(features=num_features, ratios=ratios)
    out, _ = model.init_with_output(jax.random.PRNGKey(42), inputs=inputs)
    self.assertEqual(out.shape, expected_output_shape[:-1] + (num_features,))

  @parameterized.product(
      order=("latlon", "lonlat"),
      dtype=(jnp.float32, jnp.bfloat16),
      param_dtype=(jnp.float32, jnp.bfloat16),
  )
  def test_latlon_conv_dtypes(self, order, dtype, param_dtype):
    input_shape = (2, 8, 8, 4)
    inputs = jnp.ones(input_shape, dtype=dtype)
    model = convolutions.LatLonConv(
        features=4,
        kernel_size=(3, 3),
        order=order,
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
