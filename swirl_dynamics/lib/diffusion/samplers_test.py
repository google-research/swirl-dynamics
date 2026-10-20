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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.lib.diffusion import samplers
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.lib.solvers import ode
from swirl_dynamics.lib.solvers import sde


class TimeStepSchedulersTest(parameterized.TestCase):

  def test_uniform_time(self):
    num_steps = 3
    end_time = 0.2
    expected = [1.0, 0.6, 0.2]
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    with self.subTest("end_time_test"):
      tspan = samplers.uniform_time(scheme, num_steps, end_time, end_sigma=None)
      np.testing.assert_allclose(tspan, np.asarray(expected), atol=1e-6)

    with self.subTest("end_sigma_test"):
      tspan = samplers.uniform_time(
          scheme, num_steps, end_time=None, end_sigma=sigma_schedule(end_time)  # pyrefly: ignore[bad-argument-type]
      )
      np.testing.assert_allclose(tspan, np.asarray(expected), atol=1e-6)

  @parameterized.product(end_time=[0.0, 0.2], compiled=[False, True])
  def test_uniform_time_explicit_endpoint(self, end_time, compiled):
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    schedule = functools.partial(samplers.uniform_time, scheme, num_steps=3)
    if compiled:
      schedule = jax.jit(schedule)
    tspan = schedule(end_time=jnp.asarray(end_time))
    np.testing.assert_allclose(
        tspan, np.linspace(1.0, end_time, 3), atol=1e-6
    )

  @parameterized.product(compiled=[False, True])
  def test_uniform_time_explicit_end_sigma(self, compiled):
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    schedule = functools.partial(samplers.uniform_time, scheme, num_steps=3)
    if compiled:
      schedule = jax.jit(schedule)
    tspan = schedule(end_time=None, end_sigma=jnp.asarray(0.1))
    expected_end = float(scheme.sigma.inverse(0.1))
    np.testing.assert_allclose(
        tspan, np.linspace(1.0, expected_end, 3), atol=1e-6
    )

  @parameterized.parameters((None, None), (0.0, 0.0), (0.2, 0.1))
  def test_uniform_time_requires_one_endpoint(self, end_time, end_sigma):
    scheme = diffusion.Diffusion.create_variance_exploding(
        diffusion.tangent_noise_schedule()
    )
    with self.assertRaisesRegex(ValueError, "Exactly one"):
      samplers.uniform_time(scheme, end_time=end_time, end_sigma=end_sigma)

  def test_exponential_noise_decay(self):
    num_steps = 4
    start_sigma, end_sigma = 100, 0.1
    expected_noise = np.asarray([100, 10, 1, 0.1])
    sigma_schedule = diffusion.tangent_noise_schedule(clip_max=start_sigma)
    expected_tspan = sigma_schedule.inverse(np.asarray(expected_noise))
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    tspan = samplers.exponential_noise_decay(scheme, num_steps, end_sigma)
    np.testing.assert_allclose(tspan, np.asarray(expected_tspan), atol=1e-6)

  def test_edm_noise_decay(self):
    num_steps = 3
    start_sigma, end_sigma = 100, 1
    expected_noise = np.asarray([100, 30.25, 1])
    sigma_schedule = diffusion.tangent_noise_schedule(clip_max=start_sigma)
    expected_tspan = sigma_schedule.inverse(expected_noise)
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    tspan = samplers.edm_noise_decay(
        scheme, rho=2, num_steps=num_steps, end_sigma=end_sigma
    )
    np.testing.assert_allclose(tspan, np.asarray(expected_tspan), atol=1e-6)

  @parameterized.named_parameters(
      ("exponential", samplers.exponential_noise_decay),
      ("edm", samplers.edm_noise_decay),
  )
  def test_noise_decay_single_step(self, decay_fn):
    scheme = diffusion.Diffusion.create_variance_exploding(
        diffusion.tangent_noise_schedule(clip_max=100)
    )
    tspan = decay_fn(scheme, num_steps=1, end_sigma=0.1)
    np.testing.assert_allclose(
        tspan, [diffusion.MAX_DIFFUSION_TIME], atol=1e-6
    )


class TestTransform:

  def __call__(self, denoise_fn, guidance_inputs):
    return lambda x, t, cond: denoise_fn(x, t, cond) + guidance_inputs["const"]


class SamplersTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          samplers.OdeSampler,
          {
              "integrator": ode.ExplicitEuler(),
              "return_full_paths": True,
          },
      ),
      (
          samplers.OdeSampler,
          {
              "integrator": ode.ExplicitEuler(),
              "apply_denoise_at_end": False,
              "return_full_paths": True,
          },
      ),
      (
          samplers.SdeSampler,
          {
              "integrator": sde.EulerMaruyama(iter_type="scan"),
              "return_full_paths": True,
          },
      ),
      (
          samplers.SdeSampler,
          {
              "integrator": sde.EulerMaruyama(iter_type="scan"),
              "apply_denoise_at_end": False,
              "return_full_paths": True,
          },
      ),
      (samplers.ExponentialOdeSampler, {}),
      (samplers.ExponentialSdeSampler, {}),
  )
  def test_sampler_output_shape(self, sampler, kwargs):
    input_shape = (5, 1)
    num_samples = 4
    num_steps = 8
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    sampler = sampler(
        input_shape=input_shape,
        tspan=samplers.exponential_noise_decay(scheme, num_steps),
        scheme=scheme,
        denoise_fn=lambda x, t, cond: x,
        guidance_transforms=(),
        **kwargs,
    )
    sample_paths = jax.jit(sampler.generate, static_argnums=0)(
        num_samples=num_samples, rng=jax.random.PRNGKey(0)
    )
    if kwargs.get("return_full_paths", False):
      expected_shape = (
          num_steps + int(kwargs.get("apply_denoise_at_end", True)),
          num_samples,
      ) + input_shape
    else:
      expected_shape = (num_samples,) + input_shape
    self.assertEqual(sample_paths.shape, expected_shape)

  @parameterized.parameters(
      (
          samplers.OdeSampler,
          {"integrator": ode.ExplicitEuler(), "return_full_paths": True},
      ),
      (
          samplers.SdeSampler,
          {"integrator": sde.EulerMaruyama(), "return_full_paths": True},
      ),
      (samplers.ExponentialOdeSampler, {}),
      (samplers.ExponentialSdeSampler, {}),
  )
  def test_unet_denoiser(self, sampler, kwargs):
    input_shape = (64, 64, 3)
    num_samples = 2
    num_steps = 4
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    unet = unets.PreconditionedDenoiser(
        out_channels=input_shape[-1],
        num_channels=(4, 8, 12),
        downsample_ratio=(2, 2, 2),
        num_blocks=2,
        num_heads=4,
        sigma_data=1.0,
        use_position_encoding=False,
    )
    variables = unet.init(
        jax.random.PRNGKey(42),
        x=jnp.ones((1,) + input_shape, dtype=jnp.float32),
        sigma=jnp.array(1.0, dtype=jnp.float32),
        is_training=False,
    )
    denoise_fn = functools.partial(unet.apply, variables, is_training=False)
    sampler = sampler(
        input_shape=input_shape,
        tspan=samplers.exponential_noise_decay(scheme, num_steps),
        scheme=scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        **kwargs,
    )
    samples_paths = jax.jit(sampler.generate, static_argnums=0)(
        num_samples=num_samples, rng=jax.random.PRNGKey(0)
    )
    if kwargs.get("return_full_paths", False):
      expected_shape = (
          num_steps + int(kwargs.get("apply_denoise_at_end", True)),
          num_samples,
      ) + input_shape
    else:
      expected_shape = (num_samples,) + input_shape
    self.assertEqual(samples_paths.shape, expected_shape)

  @parameterized.parameters(
      (samplers.OdeSampler, {"integrator": ode.HeunsMethod()}),
      (samplers.SdeSampler, {"integrator": sde.EulerMaruyama()}),
      (
          samplers.SdeSampler,
          {"integrator": sde.EulerMaruyama(iter_type="loop")},
      ),
      (samplers.ExponentialSdeSampler, {}),
      (samplers.ExponentialSdeSampler, {}),
  )
  def test_output_shape_with_guidance(self, sampler, kwargs):
    input_shape = (5, 1)
    num_samples = 4
    num_steps = 8
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    sampler = sampler(
        input_shape=input_shape,
        tspan=samplers.exponential_noise_decay(scheme, num_steps),
        scheme=scheme,
        denoise_fn=lambda x, t, cond: x * t,
        guidance_transforms=(TestTransform(),),
        **kwargs,
    )
    generate_fn = jax.jit(functools.partial(sampler.generate, num_samples))
    samples = generate_fn(
        rng=jax.random.PRNGKey(0),
        guidance_inputs={"const": jnp.ones(input_shape)},
    )
    self.assertEqual(samples.shape, (num_samples,) + input_shape)

  @parameterized.parameters(
      (samplers.OdeSampler, {"integrator": ode.HeunsMethod()}),
      (samplers.SdeSampler, {"integrator": sde.EulerMaruyama()}),
      (samplers.ExponentialOdeSampler, {}),
      (samplers.ExponentialSdeSampler, {}),
  )
  def test_output_shape_with_cond(self, sampler, kwargs):
    input_shape = (5, 1)
    num_samples = 4
    num_steps = 8
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    sampler = sampler(
        input_shape=input_shape,
        tspan=samplers.exponential_noise_decay(scheme, num_steps),
        scheme=scheme,
        denoise_fn=lambda x, t, cond: x * t + cond["bias"],
        guidance_transforms=(),
        **kwargs,
    )
    samples = jax.jit(sampler.generate, static_argnums=0)(
        num_samples=num_samples,
        rng=jax.random.PRNGKey(0),
        cond={"bias": jnp.ones(input_shape)},
    )
    self.assertEqual(samples.shape, (num_samples,) + input_shape)

  @parameterized.parameters(
      (samplers.OdeSampler, {"integrator": ode.HeunsMethod()}),
      (
          samplers.OdeSampler,
          {"integrator": ode.HeunsMethod(), "num_probes": 4},
      ),
      (samplers.ExponentialOdeSampler, {}),
      (samplers.ExponentialOdeSampler, {"num_probes": 4}),
  )
  def test_ode_log_likelihood_output_shape(self, sampler_cls, kwargs):
    input_shape = (8, 8, 3)
    batch_size = 4
    num_steps = 8
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    sampler = sampler_cls(
        input_shape=input_shape,
        tspan=samplers.exponential_noise_decay(scheme, num_steps),
        scheme=scheme,
        denoise_fn=lambda x, t, cond: x + cond["bias"],
        guidance_transforms=(),
        **kwargs,
    )

    log_likelihoods = sampler.compute_log_likelihood(
        inputs=jnp.ones((batch_size, *input_shape)),
        cond={"bias": jnp.ones((batch_size, *input_shape)) * 2.0},
        rng=jax.random.key(0),
    )
    self.assertEqual(log_likelihoods.shape, (batch_size,))

  @parameterized.parameters(
      (samplers.OdeSampler, True),
      (samplers.TStudentOdeSampler, False),
      (samplers.SdeSampler, True),
      (samplers.TStudentSdeSampler, False),
  )
  def test_noise_dist(self, sampler_cls, is_normal):
    input_shape = (8, 8, 3)
    num_steps = 8
    sigma_schedule = diffusion.tangent_noise_schedule()
    scheme = diffusion.Diffusion.create_variance_exploding(sigma_schedule)
    sampler = sampler_cls(
        input_shape=input_shape,
        tspan=samplers.exponential_noise_decay(scheme, num_steps),
        scheme=scheme,
        denoise_fn=lambda x, t, cond: x + cond["bias"],
    )
    self.assertEqual(is_normal, sampler.noise_dist is jax.random.normal)

  @staticmethod
  def _make_prior_sampler(sampler_type, scale, sigma_max, input_shape, times):
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

  @staticmethod
  def _normal_log_density(x, std):
    return (
        jax.scipy.stats.norm.logpdf(jnp.asarray(x), scale=std)
        .reshape(x.shape[0], -1)
        .sum(axis=-1)
    )

  @staticmethod
  def _make_exponential_likelihood_sampler(
      times, varying_scale=False, num_probes=None
  ):
    sigma = diffusion.InvertibleSchedule(
        lambda t: 0.5 + 0.5 * t, lambda y: 2 * y - 1
    )
    scale = (lambda t: 0.75 + 0.25 * t) if varying_scale else jnp.ones_like

    def denoise_fn(x, sigma, cond):
      del sigma
      assert cond is not None
      return jnp.broadcast_to(cond["center"], x.shape)

    return samplers.ExponentialOdeSampler(
        input_shape=(2,),
        scheme=diffusion.Diffusion(scale=scale, sigma=sigma),
        tspan=jnp.asarray(times),
        num_probes=num_probes,
        denoise_fn=denoise_fn,
    )

  @staticmethod
  def _log_normal(x, mean, std):
    return jax.scipy.stats.norm.logpdf(
        jnp.asarray(x), loc=jnp.asarray(mean), scale=std
    ).sum(axis=-1)

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
    sampler = self._make_prior_sampler(
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
    expected = self._normal_log_density(generated, std)
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
    sampler = self._make_prior_sampler(
        cls, lambda t: std / (1.0 + 3.0 * t), 4.0, (2,), [1.0, 0.75, 0.5, 0.25]
    )
    x = jnp.array([[-0.5, 0.2], [0.3, 0.1], [1.0, -1.0]]) * std
    rng = jax.random.key(21)
    fn = lambda values: sampler.compute_log_likelihood(values, rng=rng)
    for evaluation in (fn, jax.jit(fn)):
      np.testing.assert_allclose(
          evaluation(x), self._normal_log_density(x, std), rtol=3e-6, atol=1e-6
      )
    _, derivative = jax.jvp(fn, (x,), (jnp.ones_like(x),))
    np.testing.assert_allclose(
        derivative, -np.asarray(x).sum(axis=-1) / std**2, rtol=1e-5, atol=1e-6
    )

  @parameterized.product(varying_scale=(False, True), center=(0.0, 0.3))
  def test_forward_density_obeys_change_of_variables(
      self, varying_scale, center
  ):
    sampler = self._make_exponential_likelihood_sampler(
        [1.0, 0.0], varying_scale
    )
    x = jnp.array([[0.1, 0.2], [1.0, -1.0]])
    c = jnp.array([center, -center])
    params = dict(cond={"center": c}, guidance_inputs=None)
    initial_std = sampler.scheme.scale(0.0) * sampler.scheme.sigma(0.0)
    initial_mean = sampler.scheme.scale(0.0) * 0.5 * c
    initial_logp = self._log_normal(x, initial_mean, initial_std)
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
        logp, self._log_normal(expected_y, 0.0, 1.0), rtol=2e-6, atol=2e-6
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
    sampler = self._make_exponential_likelihood_sampler(
        jnp.linspace(1.0, 0.2, steps), varying_scale, probes
    )
    x = jnp.array([[0.1, 0.2], [0.5, -0.25], [1.0, -1.0]])
    c = jnp.array([0.3, -0.2])
    start = sampler.tspan[-1]
    scale = sampler.scheme.scale(start)
    sigma = sampler.scheme.sigma(start)
    mean, std = scale * (1 - sigma) * c, scale * sigma
    expected = self._log_normal(x, mean, std)
    fn = lambda data: sampler.compute_log_likelihood(
        data, cond={"center": c}, rng=jax.random.key(7)
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
    sampler = self._make_exponential_likelihood_sampler(
        [1.0, 0.5, 0.0]
    ).replace(denoise_fn=lambda x, sigma, cond: x)
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
    np.testing.assert_allclose(y, x, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result, logp, rtol=1e-6, atol=1e-6)

  def test_terminal_only_likelihood_is_unchanged(self):
    sampler = self._make_exponential_likelihood_sampler([1.0])
    x = jnp.array([[0.1, 0.2], [1.0, -1.0]])
    result = sampler.compute_log_likelihood(
        x, cond={"center": jnp.zeros(2)}, rng=jax.random.key(4)
    )
    np.testing.assert_allclose(
        result, self._log_normal(x, 0.0, 1.0), rtol=1e-6
    )


if __name__ == "__main__":
  absltest.main()
