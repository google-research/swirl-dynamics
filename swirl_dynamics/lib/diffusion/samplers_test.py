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
          scheme, num_steps, end_time=None, end_sigma=sigma_schedule(end_time)
      )
      np.testing.assert_allclose(tspan, np.asarray(expected), atol=1e-6)

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


if __name__ == "__main__":
  absltest.main()
