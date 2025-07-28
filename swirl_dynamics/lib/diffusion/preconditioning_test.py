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

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import preconditioning

Array: TypeAlias = jax.Array


def _broadcastable(y: Array, x_shape: tuple[int, ...]) -> Array:
  """Returns version of `y` that is broadcastable with `x_shape`.

  `y` is assumed to be either a scalar or 1D. If `y` is 1D, this function
  assumes that `y` aligns with the leading dimension of `x_shape`.

  Args:
    y: The 1D array to be broadcasted.
    x_shape: The shape of the array that the returned version of `y` will be
      broadcasted with.

  Returns:
    A version of `y` that is broadcastable with `x_shape`.
  """
  if y.ndim < 1:
    y = jnp.broadcast_to(y, (x_shape[0],))
  y = jnp.expand_dims(y, axis=tuple(range(1, len(x_shape))))
  return y


def _simple_function(x: Array, sigma: Array) -> Array:
  """Function to be used as a simple network call."""
  return 2.0 * x * sigma


class SimpleNetwork(nn.Module):
  """A simple network."""

  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: Mapping[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    del cond, is_training
    sigma = _broadcastable(sigma, x.shape)
    return _simple_function(x, sigma)


class SimpleNetworkWithDecorator(nn.Module):
  """A simple network that uses the preconditioning decorator."""

  sigma_data: float

  @preconditioning.precondition
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: Mapping[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    del cond, is_training
    sigma = _broadcastable(sigma, x.shape)
    return _simple_function(x, sigma)


def _expected_output(x: Array, sigma: Array, sigma_data: float) -> Array:
  """Returns the expected output of the preconditioned network."""
  total_var = sigma_data**2 + sigma**2
  c_skip = sigma_data**2 / total_var
  c_out = sigma * sigma_data / jnp.sqrt(total_var)
  c_in = 1 / jnp.sqrt(total_var)
  c_noise = 0.25 * jnp.log(sigma)
  expected_f_x = 2.0 * c_in * x * c_noise
  expected_output = c_skip * x + c_out * expected_f_x
  return expected_output


class PreconditioningTest(parameterized.TestCase):

  def test_compute_denoising_preconditioners(self):
    sigma = jnp.array([0.4, 0.5])
    sigma_data = 1.0
    x_shape = (2, 3, 4)
    c_in, c_out, c_skip, c_noise = (
        preconditioning.compute_denoising_preconditioners(
            sigma, sigma_data, x_shape
        )
    )

    x_ndim = len(x_shape)
    self.assertEqual(c_in.ndim, x_ndim)
    self.assertEqual(c_out.ndim, x_ndim)
    self.assertEqual(c_skip.ndim, x_ndim)
    self.assertEqual(c_noise.shape, sigma.shape)

    # Verify that c_in, c_out, c_skip are broadcastable with x
    # by trying the multiplication.
    try:
      x = jnp.zeros(x_shape)
      _ = c_in * x
      _ = c_out * x
      _ = c_skip * x
    except ValueError as e:
      self.fail(f"Broadcast failed: {e}")

  @parameterized.parameters(
      {"sigma": 0.1},
      {"sigma": [0.2, 0.3]},
  )
  def test_precondition_module(self, sigma: float | Sequence[float]):
    sigma = jnp.array(sigma)
    sigma_data = 1.5
    simple_network = SimpleNetwork()
    preconditioned_network = preconditioning.Preconditioned(
        network=simple_network, sigma_data=sigma_data
    )
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    output = preconditioned_network(x, sigma, is_training=False)

    # Verification.
    sigma_b = _broadcastable(sigma, x.shape)
    expected_output = _expected_output(x, sigma_b, sigma_data)
    np.testing.assert_allclose(output, expected_output)

  @parameterized.parameters(
      {"sigma": 0.1},
      {"sigma": [0.2, 0.3]},
  )
  def test_precondition_decorator(self, sigma: float | Sequence[float]):
    sigma = jnp.array(sigma)
    sigma_data = 1.5
    network = SimpleNetworkWithDecorator(sigma_data=sigma_data)
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    output = network(x, sigma, is_training=False)

    # Verification.
    sigma_b = _broadcastable(sigma, x.shape)
    expected_output = _expected_output(x, sigma_b, sigma_data)
    np.testing.assert_allclose(output, expected_output)


if __name__ == "__main__":
  absltest.main()
