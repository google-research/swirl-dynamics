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

"""Prior-density consistency tests for probability-flow likelihoods."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.lib.diffusion import samplers


def make_sampler(sampler_type, scale, sigma_max, input_shape, times):
  sigma = diffusion.InvertibleSchedule(
      forward=lambda t: 1.0 + (sigma_max - 1.0) * t,
      inverse=lambda value: (value - 1.0) / (sigma_max - 1.0),
  )
  scheme = diffusion.Diffusion(scale=scale, sigma=sigma)
  return sampler_type(
      input_shape=input_shape,
      scheme=scheme,
      tspan=jnp.asarray(times),
      denoise_fn=lambda x, unused_sigma, unused_cond: jnp.zeros_like(x),
      apply_denoise_at_end=False,
  )


def normal_log_density(x, std):
  return (
      stats.norm.logpdf(np.asarray(x), scale=std)
      .reshape(x.shape[0], -1)
      .sum(axis=-1)
  )


class SamplerPriorTest(parameterized.TestCase):

  @parameterized.parameters(
      (samplers.OdeSampler, 0.25, 4.0),
      (samplers.OdeSampler, 1.0, 2.0),
      (samplers.OdeSampler, 2.0, 2.0),
      (samplers.ExponentialOdeSampler, 0.25, 4.0),
      (samplers.ExponentialOdeSampler, 1.0, 2.0),
      (samplers.ExponentialOdeSampler, 2.0, 2.0),
  )
  def test_terminal_only_density_matches_the_generation_prior(
      self, cls, scale, sigma
  ):
    sampler = make_sampler(
        cls, lambda t: jnp.ones_like(t) * scale, sigma, (2, 3), [1.0]
    )
    rng = jax.random.key(19)
    generated = sampler.generate(3, rng)
    init_rng, _ = jax.random.split(rng)
    std = scale * sigma
    expected_samples = std * jax.random.normal(init_rng, (3, 2, 3))
    np.testing.assert_allclose(
        generated, expected_samples, rtol=2e-6, atol=1e-6
    )
    fn = lambda x: sampler.compute_log_likelihood(x, rng=rng)
    expected = normal_log_density(generated, std)
    for evaluation in (fn, jax.jit(fn)):
      np.testing.assert_allclose(
          evaluation(generated), expected, rtol=3e-6, atol=1e-6
      )

  @parameterized.parameters(
      (samplers.OdeSampler, 0.5),
      (samplers.OdeSampler, 2.0),
      (samplers.OdeSampler, 16.0),
      (samplers.ExponentialOdeSampler, 0.5),
      (samplers.ExponentialOdeSampler, 2.0),
      (samplers.ExponentialOdeSampler, 16.0),
  )
  def test_constant_noise_amplitude_retains_the_same_density_through_flow(
      self, cls, std
  ):
    # In this analytic case s(t)*sigma(t) is constant and D=0, so flow and
    # log-density change are zero even though the two schedules vary.
    sampler = make_sampler(
        cls, lambda t: std / (1.0 + 3.0 * t), 4.0, (2,), [1.0, 0.75, 0.5, 0.25]
    )
    x = jnp.array([[-0.5, 0.2], [0.3, 0.1], [1.0, -1.0]]) * std
    rng = jax.random.key(21)
    fn = lambda values: sampler.compute_log_likelihood(values, rng=rng)
    for evaluation in (fn, jax.jit(fn)):
      np.testing.assert_allclose(
          evaluation(x), normal_log_density(x, std), rtol=3e-6, atol=1e-6
      )
    _, derivative = jax.jvp(fn, (x,), (jnp.ones_like(x),))
    np.testing.assert_allclose(
        derivative, -np.asarray(x).sum(axis=-1) / std**2, rtol=1e-5, atol=1e-6
    )


if __name__ == '__main__':
  absltest.main()
