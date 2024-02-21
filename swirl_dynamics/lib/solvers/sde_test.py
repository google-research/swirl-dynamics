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
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib.solvers import sde


class SdeSolversTest(parameterized.TestCase):

  @parameterized.product(iter_type=("scan", "loop"), tspan_length=(1, 10))
  def test_euler_maruyama_constant_drift_no_diffusion(
      self, iter_type, tspan_length
  ):
    dt = 0.1
    x_dim = 5
    tspan = jnp.arange(tspan_length) * dt
    solver = sde.EulerMaruyama(iter_type=iter_type)
    out = solver(
        dynamics=sde.SdeDynamics(
            drift=lambda x, t, params: jnp.ones_like(x),
            diffusion=lambda x, t, params: jnp.zeros_like(x),
        ),
        x0=jnp.zeros((x_dim,)),
        tspan=tspan,
        rng=jax.random.PRNGKey(0),
        params={"drift": {}, "diffusion": {}},
    )
    final = out[-1] if iter_type == "scan" else out
    expected = jnp.ones(x_dim) * tspan[-1]
    np.testing.assert_allclose(final, expected, rtol=1e-5)

  def test_euler_maruyama_scan_loop_equivalence(self):
    dt = 0.1
    x_dim = 5
    tspan = jnp.arange(1, 10) * dt

    def solve(solver):
      return solver(
          dynamics=sde.SdeDynamics(
              drift=lambda x, t, params: params * x,
              diffusion=lambda x, t, params: params + x,
          ),
          x0=jnp.ones((x_dim,)),
          tspan=tspan,
          rng=jax.random.PRNGKey(12),
          params={"drift": 2, "diffusion": 1},
      )

    out_scan = solve(sde.EulerMaruyama(iter_type="scan"))
    out_loop = solve(sde.EulerMaruyama(iter_type="loop"))
    np.testing.assert_allclose(out_scan[-1], out_loop, rtol=1e-5)

  @parameterized.product(
      solver=(sde.EulerMaruyama(),),
      linear_coeffs=((0, 0.2), (1.5, 0.25)),
  )
  def test_linear_drift_and_diffusion(self, solver, linear_coeffs):
    dt = 1e-4
    tspan = jnp.arange(100) * dt
    rng = jax.random.PRNGKey(0)
    x0 = jnp.array(1, dtype=jnp.float64)
    mu, sigma = linear_coeffs
    dynamics_params = {"drift": mu, "diffusion": sigma}
    out = solver(
        sde.SdeDynamics(
            drift=lambda x, t, params: params * x,
            diffusion=lambda x, t, params: params * x,
        ),
        x0,
        tspan,
        rng,
        dynamics_params,
    )
    wiener_increments = jax.vmap(jax.random.normal, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(0), len(tspan)), tuple()
    )
    path = jnp.sqrt(dt) * jnp.concatenate(
        [jnp.zeros((1,)), jnp.cumsum(wiener_increments[:-1])]
    )
    # compare with analytical solution
    # dX_t = mu * X_t * dt + sigma * X_t * dW_t
    # => X_t = X_0 * exp(mu * t + sigma * W_t)
    expected = x0[None] * jnp.exp(mu * tspan + sigma * path)
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-2)

  @parameterized.product(
      solver=(sde.EulerMaruyama(),),
      linear_coeffs=((0.0, 0.2), (1.5, 0.25)),
  )
  def test_linear_drift_and_diffusion_grad(self, solver, linear_coeffs):
    dt = 1e-4
    tspan = jnp.arange(10) * dt
    rng = jax.random.PRNGKey(0)
    x0 = jnp.array(1, dtype=jnp.float64)
    mu, sigma = linear_coeffs
    sde_params = {"drift": mu, "diffusion": sigma}

    def solve(x0, tspan, rng, sde_params):
      return solver(
          sde.SdeDynamics(
              drift=lambda x, t, params: params * x,
              diffusion=lambda x, t, params: params * x,
          ),
          x0,
          tspan,
          rng,
          sde_params,
      )[-1]

    grad = jax.grad(solve, argnums=3)(x0, tspan, rng, sde_params)
    wiener_increments = jax.vmap(jax.random.normal, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(0), len(tspan)), tuple()
    )
    path = jnp.sqrt(dt) * jnp.concatenate(
        [jnp.zeros((1,)), jnp.cumsum(wiener_increments[:-1])]
    )
    # compare with analytical gradients
    # dX_t = mu * X_t * dt + c * X_t * dW_t
    # => dX_T/dmu = X_0 * T * exp(mu * T + sigma * W_T)
    # => dX_T/dsigma = X_0 * W_T * exp(mu * T + sigma * W_T)
    expected_grad_mu = (
        x0 * tspan[-1] * jnp.exp(mu * tspan[-1] + sigma * path[-1]),
    )
    expected_grad_sigma = (
        x0 * path[-1] * jnp.exp(mu * tspan[-1] + sigma * path[-1]),
    )
    np.testing.assert_allclose(
        grad["drift"], expected_grad_mu, atol=1e-3, rtol=1e-2
    )
    np.testing.assert_allclose(
        grad["diffusion"], expected_grad_sigma, atol=1e-3, rtol=1e-2
    )

  @parameterized.product(solver=(sde.EulerMaruyama(),))
  def test_backward_integration(self, solver):
    dt = 0.1
    num_steps = 10
    x_dim = 5
    tspan = -1 * jnp.arange(num_steps) * dt
    out = solver(
        dynamics=sde.SdeDynamics(
            drift=lambda x, t, params: jnp.ones_like(x),
            diffusion=lambda x, t, params: jnp.ones_like(x),
        ),
        x0=jnp.zeros((x_dim,)),
        tspan=tspan,
        rng=jax.random.PRNGKey(0),
        params={"drift": {}, "diffusion": {}},
    )
    wiener_increments = jax.vmap(jax.random.normal, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(0), len(tspan)), (x_dim,)
    )
    expected = np.ones(x_dim) * tspan[-1] + jnp.sqrt(dt) * jnp.sum(
        wiener_increments[:-1], axis=0
    )
    np.testing.assert_allclose(out[-1], expected, rtol=1e-5)

  def test_move_time_axis_pos(self):
    dt = 0.1
    num_steps = 10
    x_dim = 5
    batch_sz = 6
    tspan = jnp.arange(num_steps) * dt
    out = sde.EulerMaruyama(time_axis_pos=1)(
        dynamics=sde.SdeDynamics(
            drift=lambda x, t, params: jnp.ones_like(x),
            diffusion=lambda x, t, params: jnp.ones_like(x),
        ),
        x0=jnp.zeros((batch_sz, x_dim)),
        tspan=tspan,
        rng=jax.random.PRNGKey(1),
        params={"drift": {}, "diffusion": {}},
    )
    self.assertEqual(out.shape, (batch_sz, num_steps, x_dim))

  def test_invalid_params(self):
    with self.assertRaisesRegex(ValueError, "both `drift` and `diffusion`"):
      sde.EulerMaruyama()(
          dynamics=sde.SdeDynamics(
              drift=lambda x, t, params: jnp.ones_like(x),
              diffusion=lambda x, t, params: jnp.zeros_like(x),
          ),
          x0=jnp.ones((10,)),
          tspan=jnp.arange(10),
          rng=jax.random.PRNGKey(0),
          params={"drift": jnp.zeros((100,))},
      )


if __name__ == "__main__":
  jax.config.update("jax_enable_x64", True)
  absltest.main()
