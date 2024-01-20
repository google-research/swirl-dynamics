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

"""Tests for the Convolution-based Modules."""


from absl.testing import absltest
from absl.testing import parameterized
import jax
from swirl_dynamics.lib.networks import convnets


class DilatedBlockTest(parameterized.TestCase):
  """Testing the DilatedBlock building block of the PeriodicConvNetModel."""

  @parameterized.named_parameters(
      (f':input_dim={i}', i)
      for i in ((512,), (64, 64))
  )
  def test_output_shapes(
      self,
      input_dim=(512,),
      kernel_size=5,
      num_levels=4,
      num_channels=48,
      batch_size=2,
      input_channels=1,
  ):
    d_block = convnets.DilatedBlock(
        num_channels=num_channels,
        kernel_size=(kernel_size,),
        num_levels=num_levels
    )
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, ((batch_size,) + input_dim + (input_channels,)))
    params = d_block.init(rng, x)['params']
    y = d_block.apply({'params': params}, x)
    self.assertEqual(y.shape, ((batch_size,) + input_dim + (num_channels,)))


class PeriodicConvNetModelTest(parameterized.TestCase):
  """Testing the PeriodicConvNetModel."""

  @parameterized.named_parameters(
      (f':input_dim={i}', i)
      for i in ((512,), (64, 64))
  )
  def test_output_shapes(
      self,
      input_dim=(512,),
      batch_size=2,
      input_channels=3,
      latent_dim=48,
      num_levels=4,
      num_processors=4,
      encoder_kernel_size=(5,),
      decoder_kernel_size=(5,),
      processor_kernel_size=(5,)

  ):
    test_model = convnets.PeriodicConvNetModel(
        latent_dim=latent_dim,
        num_levels=num_levels,
        num_processors=num_processors,
        encoder_kernel_size=encoder_kernel_size,
        decoder_kernel_size=decoder_kernel_size,
        processor_kernel_size=processor_kernel_size,
    )
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, ((batch_size,) + input_dim + (input_channels,)))
    params = test_model.init(rng, x)['params']
    y = test_model.apply({'params': params}, x)
    self.assertEqual(y.shape, ((batch_size,) + input_dim + (input_channels,)))


if __name__ == '__main__':
  absltest.main()
