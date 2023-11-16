# Copyright 2023 The swirl_dynamics Authors.
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
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import guidance


class GuidanceTransformsTest(parameterized.TestCase):

  @parameterized.parameters(
      {"test_dim": (4, 16), "ds_ratios": (None, 8), "guide_shape": (4, 2)},
      {
          "test_dim": (4, 16, 16),
          "ds_ratios": (None, 3, 2),
          "guide_shape": (4, 6, 8),
      },
  )
  def test_super_resolution(self, test_dim, ds_ratios, guide_shape):
    superresolve = guidance.InfillFromSlices(
        slices=tuple(slice(None, None, r) for r in ds_ratios),
    )

    def _dummy_denoiser(x, sigma, cond=None):
      del sigma, cond
      return jnp.ones_like(x)

    guided_denoiser = superresolve(
        _dummy_denoiser, {"observed_slices": jnp.array(0.0)}
    )
    denoised = guided_denoiser(jnp.ones(test_dim), jnp.array(0.1), None)
    guided_elements = denoised[superresolve.slices]
    self.assertEqual(denoised.shape, test_dim)
    self.assertEqual(guided_elements.shape, guide_shape)

    expected = np.ones(test_dim)
    expected[superresolve.slices] = 0.0
    np.testing.assert_allclose(denoised, expected)


if __name__ == "__main__":
  absltest.main()
