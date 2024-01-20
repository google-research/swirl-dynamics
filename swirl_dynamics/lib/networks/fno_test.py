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
from swirl_dynamics.lib.networks import fno


class FnoTest(parameterized.TestCase):

  @parameterized.product(
      n_dim=(1, 2, 3),
      num_modes=(2, 3),
      spatial_dim=(8, 9),
      domain_size=(None, 16),
      contract_fn=(fno.ContractFnType.DENSE,),
      separable=(True, False),
      weights_dtype=(jnp.complex64,),
  )
  def test_spectral_conv(
      self,
      n_dim,
      num_modes,
      spatial_dim,
      domain_size,
      contract_fn,
      separable,
      weights_dtype,
  ):
    batch_sz = 2
    in_channels, out_channels = 5, 5
    input_shape = (batch_sz,) + (spatial_dim,) * n_dim + (in_channels,)
    inputs = jax.random.normal(jax.random.PRNGKey(0), input_shape)
    domain_size = domain_size or spatial_dim
    layer = fno.SpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        num_modes=(num_modes,) * n_dim,
        domain_size=(domain_size,) * n_dim,
        contract_fn=contract_fn,
        separable=separable,
        weights_dtype=weights_dtype,
    )
    layer_vars = layer.init(jax.random.PRNGKey(0), inputs)
    out = jax.jit(layer.apply)(layer_vars, inputs)
    out_shape = (batch_sz,) + (domain_size,) * n_dim + (out_channels,)
    self.assertEqual(out.shape, out_shape)

  @parameterized.product(
      n_dim=(1, 2, 3),
      skip_type=("linear", "soft-gate", "identity"),
  )
  def test_fno_res_block(self, n_dim, skip_type):
    batch_sz = 2
    num_channels = 6
    spatial_dim = 12
    num_modes = (4,) * n_dim
    input_shape = (batch_sz,) + (spatial_dim,) * n_dim + (num_channels,)
    inputs = jax.random.normal(jax.random.PRNGKey(0), input_shape)
    block = fno.FnoResBlock(
        out_channels=num_channels,
        num_modes=num_modes,
        num_layers=4,
        contract_fn=fno.ContractFnType.DENSE,
        separable=False,
        skip_type=skip_type,
        param_dtype=jnp.complex64,
    )
    block_vars = block.init(jax.random.PRNGKey(0), inputs)
    out = jax.jit(block.apply)(block_vars, inputs)
    out_shape = (batch_sz,) + (spatial_dim,) * n_dim + (num_channels,)
    self.assertEqual(out.shape, out_shape)

  @parameterized.product(
      n_dim=(1, 2, 3),
  )
  def test_fno(self, n_dim):
    batch_sz = 2
    in_channels, out_channels, hidden_channels = 3, 1, 16
    projection_channels = lifting_channels = 24
    spatial_dim = 12
    num_modes = (4,) * n_dim
    input_shape = (batch_sz,) + (spatial_dim,) * n_dim + (in_channels,)
    inputs = jax.random.normal(jax.random.PRNGKey(0), input_shape)
    model = fno.Fno(
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_modes=num_modes,
        lifting_channels=lifting_channels,
        projection_channels=projection_channels,
    )
    model_vars = model.init(jax.random.PRNGKey(0), inputs)
    out = jax.jit(model.apply)(model_vars, inputs)
    out_shape = (batch_sz,) + (spatial_dim,) * n_dim + (out_channels,)
    self.assertEqual(out.shape, out_shape)

  def test_fno_2d(self):
    batch_sz = 2
    in_channels, out_channels = 1, 1
    num_modes = (4, 4)
    width = 5
    spatial_dim = 12
    input_shape = (batch_sz,) + (spatial_dim,) * 2 + (in_channels,)
    inputs = jax.random.normal(jax.random.PRNGKey(0), input_shape)
    model = fno.Fno2d(
        out_channels=out_channels, num_modes=num_modes, width=width
    )
    model_vars = model.init(jax.random.PRNGKey(0), inputs)
    out = jax.jit(model.apply)(model_vars, inputs)
    out_shape = (batch_sz,) + (spatial_dim,) * 2 + (out_channels,)
    self.assertEqual(out.shape, out_shape)


if __name__ == "__main__":
  absltest.main()
