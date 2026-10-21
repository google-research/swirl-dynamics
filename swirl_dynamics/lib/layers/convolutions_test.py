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
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
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

  @parameterized.parameters(((5, 3),), ((3, 5),))
  def test_latlon_conv_rectangular_kernel_orders_are_consistent(
      self, kernel_size_latlon
  ):
    # Transposing the spatial axes, the kernel and the kernel parameters must
    # map the "latlon" convolution onto the "lonlat" one exactly; in particular
    # both orders must preserve the spatial shape.
    num_features = 3
    inputs_latlon = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 9, 3))
    inputs_lonlat = jnp.swapaxes(inputs_latlon, 1, 2)
    kernel_size_lonlat = kernel_size_latlon[::-1]

    model_latlon = convolutions.LatLonConv(
        features=num_features, kernel_size=kernel_size_latlon, order="latlon"
    )
    model_lonlat = convolutions.LatLonConv(
        features=num_features, kernel_size=kernel_size_lonlat, order="lonlat"
    )
    out_latlon, variables = model_latlon.init_with_output(
        jax.random.PRNGKey(42), inputs_latlon
    )
    variables_lonlat = jax.tree_util.tree_map(
        lambda leaf: jnp.swapaxes(leaf, 0, 1) if leaf.ndim == 4 else leaf,
        variables,
    )
    out_lonlat = jnp.asarray(
        model_lonlat.apply(variables_lonlat, inputs_lonlat)
    )

    self.assertEqual(out_latlon.shape, (2, 16, 9, num_features))
    self.assertEqual(out_lonlat.shape, (2, 9, 16, num_features))
    np.testing.assert_allclose(
        jnp.swapaxes(out_latlon, 1, 2), out_lonlat, atol=1e-5
    )

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
    self.assertEqual(out.dtype, dtype)  # pyrefly: ignore[missing-attribute]
    self.assertEqual(out.shape, input_shape)  # pyrefly: ignore[missing-attribute]

  @parameterized.parameters("latlon", "lonlat")
  def test_latlon_conv_nonsquare_kernel_lonlat_preserves_shape(self, order):
    input_shape = (2, 8, 6, 4)
    inputs = jnp.ones(input_shape, dtype=jnp.float32)
    model = convolutions.LatLonConv(
        features=5,
        kernel_size=(3, 5),
        order=order,
    )
    out, _ = model.init_with_output(jax.random.PRNGKey(0), inputs)
    self.assertEqual(out.shape, (2, 8, 6, 5))


if __name__ == "__main__":
  absltest.main()
