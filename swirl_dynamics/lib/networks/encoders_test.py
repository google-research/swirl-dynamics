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

"""Tests for the Non-Linear Fourier networks."""


from absl.testing import absltest
from absl.testing import parameterized
import jax
from swirl_dynamics.lib.networks import encoders
from swirl_dynamics.lib.networks import utils


class ResNetTest(parameterized.TestCase):
  """Testing the ResNet building block of the encoder."""

  @parameterized.named_parameters(
      (f':features={s};kernel_size={i}', s, i)  # pylint: disable=g-complex-comprehension
      for s in (2, 5)
      for i in (1, 2, 4)
  )
  def test_number_params_and_batch_stats(self, features, kernel_size):
    # Testing that the network has the correct number of parameters.

    resnet_test = encoders.ResNetBlock1D(
        features=features, kernel_size=(kernel_size,)
    )

    # Generating an input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (10, features))
    params_dict = resnet_test.init(rng, x)

    params = params_dict['params']
    batch_stats = params_dict['batch_stats']

    # Computing the number of parameters and batch stats.
    num_params = utils.flat_dim(params)
    num_params_expected = 2 * kernel_size * features**2 + 4 * features
    self.assertEqual(num_params, num_params_expected)

    num_batch_stats = utils.flat_dim(batch_stats)
    num_batch_stats_expected = 4 * features
    self.assertEqual(num_batch_stats, num_batch_stats_expected)

  @parameterized.named_parameters(
      (f':dim_in={s};features={i}', s, i)  # pylint: disable=g-complex-comprehension
      for s in (4, 8)
      for i in (2, 4)
  )
  def test_output_downsample(self, dim_in, features):
    # Testing that the output has the correct size.

    resnet_test = encoders.ResNetBlock1D(
        features=features, kernel_size=(2,), downsample=True
    )

    # Generating an input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (dim_in, features))
    init_values = resnet_test.init(rng, x)

    params = init_values['params']
    batch_stats = init_values['batch_stats']

    y = resnet_test.apply(
        {'params': params, 'batch_stats': batch_stats}, x, is_training=False
    )
    dim_expected = (dim_in / 2, features)
    self.assertSequenceEqual(y.shape, dim_expected)


class EncoderResNetTest(parameterized.TestCase):

  def test_number_params_and_batch_stats(self):
    # Test of the currect number of parameters and batch stats.

    encoder_model = encoders.EncoderResNet(
        filters=1, dim_out=2, num_levels=2, num_resnet_blocks=1
    )

    # Generating an input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (32, 1))
    init_values = encoder_model.init(rng, x)

    params = init_values['params']
    batch_stats = init_values['batch_stats']

    # Computing the number of parameters and batch stats, compared to the
    # reference number computed by hand.
    num_params = utils.flat_dim(params)
    self.assertEqual(num_params, 89)
    num_batch_stats = utils.flat_dim(batch_stats)
    self.assertEqual(num_batch_stats, 12)

  @parameterized.parameters((1,), (2,), (3,))
  def test_output_dimension(self, dim_out):
    encoder_model = encoders.EncoderResNet(
        filters=1, dim_out=dim_out, num_levels=2, num_resnet_blocks=1
    )

    # Generating an input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (32, 1))
    init_values = encoder_model.init(rng, x)

    params = init_values['params']
    batch_stats = init_values['batch_stats']

    y = encoder_model.apply(
        {'params': params, 'batch_stats': batch_stats}, x, is_training=False
    )
    dim_expected = (1, dim_out)
    self.assertSequenceEqual(y.shape, dim_expected)


if __name__ == '__main__':
  absltest.main()
