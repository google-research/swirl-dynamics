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
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import preconditioning


class SimpleNetwork(nn.Module):
  """A simple network without internal variables."""

  def __call__(
      self,
      x: jax.Array,
      sigma: jax.Array,
      cond: dict[str, jax.Array] | None = None,
      *,
      is_training: bool,
  ) -> jax.Array:
    del cond, is_training
    sigma = jnp.expand_dims(sigma, axis=tuple(range(1, x.ndim)))
    return 2.0 * x * sigma


class PreconditioningTest(absltest.TestCase):

  def test_preconditioning(self):
    sigma = jnp.array(0.4)
    simple_network = SimpleNetwork()
    preconditioned_network = preconditioning.Preconditioned(
        network=simple_network, sigma_data=1.0
    )

    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output = preconditioned_network(x, sigma, is_training=False)

    # Validation.
    c_skip = 1 / (1 + sigma**2)
    c_out = sigma / jnp.sqrt(1 + sigma**2)
    c_in = 1 / jnp.sqrt(1 + sigma**2)
    c_noise = 0.25 * jnp.log(sigma)
    expected_f_x = 2.0 * c_in * x * c_noise
    expected_output = c_skip * x + c_out * expected_f_x
    np.testing.assert_allclose(output, expected_output, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
