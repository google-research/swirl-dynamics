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
# pylint: disable=undefined-variable, g-complex-comprehension

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.networks import nonlinear_fourier
from swirl_dynamics.lib.networks import utils

jax.config.update('jax_enable_x64', True)


class NonLinearFourierTest(absltest.TestCase):

  def test_number_params(self):
    # Testing that the network has the correct number of parameters.

    features = (2, 2)
    periodic_mlp_small = nonlinear_fourier.NonLinearFourier(features=features)

    # Generating an input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1,))
    params = periodic_mlp_small.init(rng, x)['params']

    # Computing the number of parameters and checking they are correct.
    num_params = utils.flat_dim(params)
    self.assertEqual(num_params, 44)

  def test_freqs(self):
    """Testing that the frequencies are properly initialized."""
    features = (2, 2)
    periodic_mlp_dyadic = nonlinear_fourier.NonLinearFourier(
        features=features, dyadic=True, train_freqs=True, zero_freq=False
    )
    periodic_mlp = nonlinear_fourier.NonLinearFourier(
        features=features, dyadic=False, train_freqs=True, zero_freq=True
    )

    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1,))

    # Checking the initialization of the dyadic frequencies.
    params_dyadic = periodic_mlp_dyadic.init(rng, x)['params']

    freqs_expected = (
        jnp.pi
        * jnp.repeat(2 ** jnp.linspace(0, 2, 3, dtype=jnp.float32), 2)
        .reshape((-1, 2))
        .T
    )
    np.testing.assert_array_equal(params_dyadic['frequencies'], freqs_expected)

    # Checking the initialization of the equispaced frequencies.
    params = periodic_mlp.init(rng, x)['params']

    freqs_expected = (
        jnp.pi
        * jnp.repeat(jnp.linspace(1, 3, 3, dtype=jnp.float32), 2)
        .reshape((-1, 2))
        .T
    )
    np.testing.assert_array_equal(params['frequencies'], freqs_expected)
    # Checking that the zero-th frequency coefficient is properly initialized.
    np.testing.assert_array_equal(params['const'], np.array([0.0]))


class MLPTest(parameterized.TestCase):
  """Testing the MLP within the nonlinear Fourier class."""

  @parameterized.named_parameters(
      (f':input_dim={s};num_neurons={i}', s, i)
      for s in (2, 5)
      for i in ((2, 2), (10, 10, 10, 10))
  )
  def test_number_params_and_shape(self, input_dim, num_neurons):
    """Testing that we have the correct number of parameters and shape."""
    mlp_small = nonlinear_fourier.MLP(num_neurons)

    # Generating an input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (input_dim,))
    params = mlp_small.init(rng, x)['params']

    # Computing the number of parameters and checking they are correct.
    num_params = utils.flat_dim(params)
    neurons = jnp.array((input_dim,) + num_neurons)
    total_params = jnp.sum(neurons[:-1] * neurons[1:]) + jnp.sum(neurons[1:])
    self.assertEqual(num_params, total_params)

  @parameterized.named_parameters(
      (f':dtype={s}', s) for s in (jnp.float32, jnp.float64)
  )
  def test_type_params_and_outputs(self, dtype):
    """Testing that network output and parameters have the correct types."""
    num_neurons = (10, 10)
    mlp_small = nonlinear_fourier.MLP(num_neurons, dtype=dtype)

    # Generating an input and initializing the network.
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (10,), dtype=dtype)
    params = mlp_small.init(rng, x)['params']

    # Running the model through the input and checking types.
    y = mlp_small.apply({'params': params}, x)
    self.assertEqual(params['Dense_0']['kernel'].dtype, dtype)
    self.assertEqual(y.dtype, dtype)
    self.assertEqual(y.shape, (10,))


class NonLinearFourier2DTest(absltest.TestCase):
  """Testing the two-dimension nonlinear Fourier network."""

  def test_number_params_and_output(self):
    """Testing that we have the correct number of parameters and output."""

    num_features = (5, 5)

    per_mlp_2d = nonlinear_fourier.NonLinearFourier2D(
        features=num_features, num_freqs=2, dyadic=True
    )

    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (2,))
    params = per_mlp_2d.init(rng, x)['params']

    # Computing the number of parameters.
    num_params = utils.flat_dim(params)
    # Computing the number of parameters is given by:
    # first layer :  2*(2*num_freqs) phases
    # second layer:  (2*num_freqs +1)^2  * num_features + num_features
    # second layer:  num_features * num_features + num_features
    # third layer:   (2*num_freqs +1)^2  * num_features + num_features
    # So in this case    2*(2 * 2).                         = 8
    #                  + (2 * 2 + 1)^2 * 5 + 5              = 130
    #                  + 5 * 5 + 5                          = 30
    #                  + 5 * (2 * 2 + 1)^2 + (2 * 2 + 1)^2  = 150
    #                           Total                       = 318
    self.assertEqual(num_params, 318)

    # Evaluating the network on a batch sample.
    x_array = jax.random.normal(rng, (5, 2))

    y_array = jax.vmap(per_mlp_2d.apply, in_axes=(None, 0), out_axes=0)(
        {'params': params}, x_array
    )

    y_expected = jnp.array(
        [0.61469799, 2.84619846, 2.78081641, 0.43919809, -0.22430251]
    )

    self.assertSequenceAlmostEqual(y_array, y_expected, delta=1e-5)


if __name__ == '__main__':
  absltest.main()
