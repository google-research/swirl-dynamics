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

  @parameterized.parameters(
      {"mask_keys": None, "mask_value": 0, "expected": 13},
      {"mask_keys": None, "mask_value": 1, "expected": 12},
      {"mask_keys": ("0", "1", "2"), "mask_value": 0, "expected": 11.6},
  )
  def test_classifier_free_hybrid(self, mask_keys, mask_value, expected):
    cf_hybrid = guidance.ClassifierFreeHybrid(
        guidance_strength=0.2,
        cond_mask_keys=mask_keys,
        cond_mask_value=mask_value,
    )

    def _dummy_denoiser(x, sigma, cond):
      del sigma
      out = jnp.ones_like(x)
      for v in cond.values():
        out += v
      return out

    guided_denoiser = cf_hybrid(_dummy_denoiser, {})
    cond = {str(v): jnp.array(v) for v in range(5)}
    denoised = guided_denoiser(jnp.array(0), jnp.array(0.1), cond)
    self.assertAlmostEqual(denoised, expected, places=5)

  @parameterized.parameters(
      {"test_dim": (1, 2, 4, 4, 4, 1), "style": "swap"},
      {"test_dim": (1, 2, 4, 4, 4, 1), "style": "average"},
  )
  def test_frame_interlocking(self, test_dim, style):
    interlock = guidance.InterlockingFrames(style=style)

    def _dummy_denoiser(x, sigma, cond=None):
      del sigma, cond
      return jnp.ones_like(x)

    guided_denoiser = interlock(_dummy_denoiser)
    denoised = guided_denoiser(jnp.ones(test_dim), jnp.array(0.1), None)

    self.assertEqual(denoised.shape, test_dim)

if __name__ == "__main__":
  absltest.main()
