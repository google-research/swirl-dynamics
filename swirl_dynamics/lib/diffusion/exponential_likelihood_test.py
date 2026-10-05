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

"""Analytic change-of-variables tests for exponential ODE likelihoods."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.lib.diffusion import samplers


def make_sampler(times, varying_scale=False, num_probes=None):
  sigma = diffusion.InvertibleSchedule(
      lambda t: 0.5 + 0.5 * t, lambda y: 2 * y - 1
  )
  scale = (lambda t: 0.75 + 0.25 * t) if varying_scale else jnp.ones_like
  return samplers.ExponentialOdeSampler(
      input_shape=(2,),
      scheme=diffusion.Diffusion(scale=scale, sigma=sigma),
      tspan=jnp.asarray(times),
      num_probes=num_probes,
      denoise_fn=lambda x, sigma, cond: jnp.broadcast_to(
          cond['center'], x.shape
      ),
  )


def log_normal(x, mean, std):
  return stats.norm.logpdf(
      np.asarray(x), loc=np.asarray(mean), scale=float(std)
  ).sum(axis=-1)


class ExponentialLikelihoodTest(parameterized.TestCase):

  @parameterized.product(varying_scale=(False, True), center=(0.0, 0.3))
  def test_forward_density_obeys_change_of_variables(
      self, varying_scale, center
  ):
    sampler = make_sampler([1.0, 0.0], varying_scale)
    x = jnp.array([[0.1, 0.2], [1.0, -1.0]])
    c = jnp.array([center, -center])
    params = dict(cond={'center': c}, guidance_inputs=None)
    initial_std = sampler.scheme.scale(0.0) * sampler.scheme.sigma(0.0)
    initial_mean = sampler.scheme.scale(0.0) * 0.5 * c
    initial_logp = jnp.asarray(log_normal(x, initial_mean, initial_std))
    y, logp = sampler.forward_step(
        x,
        jnp.array(0.0),
        jnp.array(1.0),
        params,
        logp0=initial_logp,
        rng=jax.random.key(0),
    )
    expected_y = (x - initial_mean) / initial_std
    np.testing.assert_allclose(y, expected_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        logp, log_normal(expected_y, 0.0, 1.0), rtol=2e-6, atol=2e-6
    )
    y_without_density, empty = sampler.forward_step(
        x, jnp.array(0.0), jnp.array(1.0), params
    )
    np.testing.assert_array_equal(y, y_without_density)
    self.assertIsNone(empty)

  @parameterized.product(
      varying_scale=(False, True), steps=(2, 5), probes=(None, 4)
  )
  def test_complete_likelihood_matches_analytic_pullback(
      self, varying_scale, steps, probes
  ):
    sampler = make_sampler(jnp.linspace(1.0, 0.2, steps), varying_scale, probes)
    x = jnp.array([[0.1, 0.2], [0.5, -0.25], [1.0, -1.0]])
    c = jnp.array([0.3, -0.2])
    start = sampler.tspan[-1]
    scale = sampler.scheme.scale(start)
    sigma = sampler.scheme.sigma(start)
    mean, std = scale * (1 - sigma) * c, scale * sigma
    expected = log_normal(x, mean, std)
    fn = lambda data: sampler.compute_log_likelihood(
        data, cond={'center': c}, rng=jax.random.key(7)
    )
    for run in (fn, jax.jit(fn)):
      np.testing.assert_allclose(run(x), expected, rtol=3e-6, atol=2e-6)
    _, derivative = jax.jvp(fn, (x,), (jnp.ones_like(x),))
    np.testing.assert_allclose(
        derivative,
        -np.sum(np.asarray(x - mean), axis=-1) / float(std) ** 2,
        rtol=1e-5,
        atol=2e-6,
    )

  def test_identity_denoiser_keeps_density_unchanged(self):
    sampler = make_sampler([1.0, 0.5, 0.0]).replace(
        denoise_fn=lambda x, sigma, cond: x
    )
    x = jnp.array([[0.1, 0.2], [1.0, -1.0]])
    params = dict(cond=None, guidance_inputs=None)
    logp = jnp.array([-1.0, -2.0])
    y, result = sampler.forward_step(
        x,
        jnp.array(0.0),
        jnp.array(1.0),
        params,
        logp0=logp,
        rng=jax.random.key(3),
    )
    np.testing.assert_allclose(y, x, rtol=0, atol=0)
    np.testing.assert_allclose(result, logp, rtol=0, atol=0)

  def test_terminal_only_likelihood_is_unchanged(self):
    sampler = make_sampler([1.0])
    x = jnp.array([[0.1, 0.2], [1.0, -1.0]])
    result = sampler.compute_log_likelihood(
        x, cond={'center': jnp.zeros(2)}, rng=jax.random.key(4)
    )
    np.testing.assert_allclose(result, log_normal(x, 0.0, 1.0), rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
