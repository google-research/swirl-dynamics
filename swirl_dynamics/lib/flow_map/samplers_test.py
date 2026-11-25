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

"""Unit tests for `swirl_dynamics.lib.flow_map.samplers`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.lib import flow_map


def _exact_power_law_inv_cdf(
    u: np.ndarray, w_min: float, w_max: float, beta: float
) -> np.ndarray:
  """Reference float64 inverse CDF for `p(w) propto w**(-beta)`."""
  u64 = np.asarray(u, dtype=np.float64)
  log_ratio = np.log(w_max / w_min)
  if abs(beta - 1.0) < 1e-12:
    return w_min * np.exp(u64 * log_ratio)
  omb = 1.0 - beta
  a = omb * log_ratio
  if abs(a) <= 1e-6:
    log_off = (
        log_ratio
        * u64
        * (
            1.0
            + 0.5 * (1.0 - u64) * a
            + (1.0 / 6.0) * (1.0 - u64) * (1.0 - 2.0 * u64) * a * a
        )
    )
    return w_min * np.exp(log_off)
  if a > 0.0:
    log_num = a + np.log((1.0 - u64) * np.exp(-a) + u64)
  else:
    log_num = np.log((1.0 - u64) + u64 * np.exp(a))
  return w_min * np.exp(log_num / omb)


class SamplersTest(parameterized.TestCase):

  def test_public_exports_sorted(self):
    self.assertEqual(flow_map.__all__, sorted(flow_map.__all__))
    for name in flow_map.__all__:
      self.assertTrue(hasattr(flow_map, name))

  def test_logit_normal_dist_bounds_quantiles_and_validation(self):
    rng = jax.random.PRNGKey(0)
    samples = flow_map.logit_normal_dist(rng, (16384,), mean=-0.4, std=1.0)
    self.assertEqual(samples.shape, (16384,))
    self.assertTrue(bool(jnp.all((samples > 0.0) & (samples < 1.0))))
    # Median of sigmoid(N(mean, std^2)) equals sigmoid(mean).
    expected_median = float(jax.nn.sigmoid(-0.4))
    self.assertAlmostEqual(
        float(jnp.median(samples)), expected_median, delta=0.015
    )

    zero_std = flow_map.logit_normal_dist(rng, (8,), mean=0.5, std=0.0)
    np.testing.assert_allclose(
        zero_std, np.full((8,), float(jax.nn.sigmoid(0.5))), rtol=1e-6
    )

    with self.assertRaisesRegex(ValueError, "non-negative"):
      flow_map.logit_normal_dist(rng, (4,), std=-0.1)

  @parameterized.named_parameters(
      dict(testcase_name="off_diagonal", diag_rate=0.0, expected_eq=0.0),
      dict(testcase_name="half_diagonal", diag_rate=0.5, expected_eq=0.5),
      dict(testcase_name="all_diagonal", diag_rate=1.0, expected_eq=1.0),
  )
  def test_time_sampler_mean_flow_ordering_and_diagonal_rate(
      self, diag_rate: float, expected_eq: float
  ):
    rng = jax.random.PRNGKey(7)
    t, s = flow_map.time_sampler_mean_flow(
        rng,
        batch_size=8192,
        min_train_time=0.05,
        max_train_time=0.95,
        diagonal_sampling_rate=diag_rate,
    )
    self.assertEqual(t.shape, (8192,))
    self.assertEqual(s.shape, (8192,))
    self.assertTrue(bool(jnp.all(t >= s)))
    self.assertTrue(bool(jnp.all((s >= 0.05) & (t <= 0.95))))
    eq_rate = float(jnp.mean(t == s))
    self.assertAlmostEqual(eq_rate, expected_eq, delta=0.025)

  def test_time_sampler_flow_map_and_validation(self):
    rng = jax.random.PRNGKey(3)
    t, s = flow_map.time_sampler_flow_map(rng, batch_size=256)
    self.assertTrue(bool(jnp.all(t >= s)))
    self.assertLess(float(jnp.mean(t == s)), 0.02)

    with self.assertRaisesRegex(ValueError, "diagonal_sampling_rate"):
      flow_map.time_sampler_mean_flow(rng, 4, diagonal_sampling_rate=1.5)
    with self.assertRaisesRegex(ValueError, "max_train_time"):
      flow_map.time_sampler_mean_flow(
          rng, 4, min_train_time=0.8, max_train_time=0.2
      )

  @parameterized.named_parameters(
      dict(testcase_name="beta_neg20", beta=-20.0),
      dict(testcase_name="beta_zero", beta=0.0),
      dict(testcase_name="beta_half", beta=0.5),
      dict(testcase_name="beta_near_one_below", beta=1.0 - 2e-5),
      dict(testcase_name="beta_one", beta=1.0),
      dict(testcase_name="beta_near_one_above", beta=1.0 + 2e-5),
      dict(testcase_name="beta_two", beta=2.0),
      dict(testcase_name="beta_pos40", beta=40.0),
  )
  def test_sample_cfg_scale_exact_pointwise_inverse_cdf(self, beta: float):
    rng = jax.random.PRNGKey(11)
    w_min, w_max = 1.0, 8.0
    u = jax.random.uniform(rng, (1024,), dtype=jnp.float32)
    actual = flow_map.sample_cfg_scale(
        rng, (1024,), w_min=w_min, w_max=w_max, beta=beta
    )
    expected = _exact_power_law_inv_cdf(np.asarray(u), w_min, w_max, beta)
    self.assertTrue(bool(jnp.all(jnp.isfinite(actual))))
    self.assertTrue(bool(jnp.all((actual >= w_min) & (actual <= w_max))))
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-6)

  def test_sample_cfg_scale_degenerate_gradient_and_validation(self):
    rng = jax.random.PRNGKey(19)
    const = flow_map.sample_cfg_scale(rng, (16,), w_min=3.0, w_max=3.0)
    np.testing.assert_allclose(const, np.full((16,), 3.0), rtol=1e-6)

    def _mean_scale(beta: jax.Array) -> jax.Array:
      return jnp.mean(
          flow_map.sample_cfg_scale(rng, (64,), w_min=1.0, w_max=8.0, beta=beta)
      )

    grad_at_one = float(jax.grad(_mean_scale)(jnp.asarray(1.0)))
    eps = 1e-3
    fd_grad = float(
        (
            _mean_scale(jnp.asarray(1.0 + eps))
            - _mean_scale(jnp.asarray(1.0 - eps))
        )
        / (2.0 * eps)
    )
    self.assertTrue(np.isfinite(grad_at_one))
    self.assertAlmostEqual(grad_at_one, fd_grad, delta=2e-3)

    with self.assertRaisesRegex(ValueError, "0 < w_min <= w_max"):
      flow_map.sample_cfg_scale(rng, (4,), w_min=0.0, w_max=2.0)
    with self.assertRaisesRegex(ValueError, "0 < w_min <= w_max"):
      flow_map.sample_cfg_scale(rng, (4,), w_min=4.0, w_max=2.0)

  def test_sample_cfg_interval_ordering_and_validation(self):
    rng = jax.random.PRNGKey(23)
    t_min, t_max = flow_map.sample_cfg_interval(
        rng, 256, t_min_range=(0.1, 0.6), t_max_range=(0.4, 0.9)
    )
    self.assertEqual(t_min.shape, (256,))
    self.assertEqual(t_max.shape, (256,))
    self.assertTrue(bool(jnp.all(t_min <= t_max)))
    self.assertTrue(bool(jnp.all((t_min >= 0.1) & (t_max <= 0.9))))

    with self.assertRaisesRegex(ValueError, "Invalid interval ranges"):
      flow_map.sample_cfg_interval(rng, 4, t_min_range=(0.6, 0.2))
    with self.assertRaisesRegex(ValueError, "Invalid interval ranges"):
      flow_map.sample_cfg_interval(rng, 4, t_max_range=(0.9, 0.3))

  def test_adaptive_weighted_loss_exact_value_and_validation(self):
    err = jnp.asarray([[1.0, 1.0], [3.0, 3.0]], dtype=jnp.float32)
    loss, raw_mse = flow_map.adaptive_weighted_loss(
        err, norm_power=1.0, norm_regularization=1.0
    )
    # Per-sample MSEs are 1.0 and 9.0 -> weights 2.0 and 10.0.
    expected_loss = 0.5 * (1.0 / 2.0 + 9.0 / 10.0)
    self.assertAlmostEqual(float(loss), expected_loss, places=6)
    self.assertAlmostEqual(float(raw_mse), 5.0, places=6)

    # Stop-gradient on denominator: d/d(err_ij) of e_b^2 / sg(w_b)
    # = (2 * err_ij / (B * D)) / w_b.
    grad = jax.grad(
        lambda e: flow_map.adaptive_weighted_loss(
            e, norm_power=1.0, norm_regularization=1.0
        )[0]
    )(err)
    expected_grad = jnp.asarray([
        [1.0 / (2.0 * 2.0), 1.0 / (2.0 * 2.0)],
        [3.0 / (2.0 * 10.0), 3.0 / (2.0 * 10.0)],
    ])
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-6, atol=1e-6)

    with self.assertRaisesRegex(ValueError, "norm_regularization"):
      flow_map.adaptive_weighted_loss(err, norm_regularization=0.0)
    with self.assertRaisesRegex(ValueError, "leading batch axis"):
      flow_map.adaptive_weighted_loss(jnp.asarray(1.0))

  def test_compound_velocity_u_loss_v_loss_gradient_identity(self):
    """Verifies exact gradient identity between u-loss and v-loss."""
    rng = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(rng, 3)
    x_t = jax.random.normal(k1, (8, 4))
    t = jnp.linspace(0.3, 0.9, 8)
    s = jnp.linspace(0.1, 0.5, 8)
    v_tgt = jax.random.normal(k2, (8, 4))
    w_init = jax.random.normal(k3, (4, 4)) * 0.2
    b_init = jnp.zeros((4,))

    def _u_eval(params, x, t_in, s_in):
      w, b = params
      dt = (t_in - s_in)[:, None]
      return (x @ w + b) * (1.0 + 0.3 * t_in[:, None] - 0.2 * dt), None

    def _v_loss(params):
      compound_v, _, _, _ = flow_map.compute_compound_velocity(
          lambda x, ti, si: _u_eval(params, x, ti, si),
          x_t,
          t,
          s,
          v_tgt,
      )
      return flow_map.adaptive_weighted_loss(compound_v - v_tgt)[0]

    def _u_loss(params):
      u_ts, du_dt, _ = jax.jvp(
          lambda x, ti, si: _u_eval(params, x, ti, si),
          (x_t, t, s),
          (v_tgt, jnp.ones_like(t), jnp.zeros_like(s)),
          has_aux=True,
      )
      u_tgt = jax.lax.stop_gradient(v_tgt - (t - s)[:, None] * du_dt)
      return flow_map.adaptive_weighted_loss(u_ts - u_tgt)[0]

    grad_v = jax.grad(_v_loss)((w_init, b_init))
    grad_u = jax.grad(_u_loss)((w_init, b_init))
    for gv, gu in zip(grad_v, grad_u, strict=True):
      np.testing.assert_allclose(gv, gu, rtol=1e-6, atol=1e-6)

  def test_compute_guided_velocity_target_four_regimes(self):
    # Four batch samples covering all 4 regimes of Eq. 17:
    # 0: active in-interval (w=4.0) -> w_eff=4.0, coeff=0.75
    # 1: dropped class (w=4.0)      -> w_eff=1.0, coeff=0.0
    # 2: outside interval (t<t_min) -> w_eff=1.0, coeff=0.0
    # 3: active unit scale (w=1.0)  -> w_eff=1.0, coeff=0.0
    v_base = jnp.asarray([[2.0], [2.0], [2.0], [2.0]])
    v_cond = jnp.asarray([[6.0], [6.0], [6.0], [6.0]])
    v_uncond = jnp.asarray([[2.0], [2.0], [2.0], [2.0]])
    cfg_scale = jnp.asarray([4.0, 4.0, 4.0, 1.0])
    time_t = jnp.asarray([0.5, 0.5, 0.1, 0.5])
    t_min = jnp.asarray([0.2, 0.2, 0.2, 0.2])
    t_max = jnp.asarray([0.8, 0.8, 0.8, 0.8])
    is_dropped = jnp.asarray([False, True, False, False])

    v_tgt, omega_eff = flow_map.compute_guided_velocity_target(
        v_base,
        v_cond,
        v_uncond,
        cfg_scale,
        time_t,
        t_min,
        t_max,
        is_dropped,
    )
    np.testing.assert_allclose(omega_eff, [4.0, 1.0, 1.0, 1.0], atol=1e-6)
    # Fixed-point check: at w_eff=4, v_cond = 4*v_c - 3*v_u; when v_base = v_c
    # and v_cond - v_uncond = 4.0, v_tgt = 2.0 + 0.75 * 4.0 = 5.0.
    np.testing.assert_allclose(v_tgt, [[5.0], [2.0], [2.0], [2.0]], atol=1e-6)

    # Confirm exact stop-gradient on v_tgt.
    grad_fn = jax.grad(
        lambda vc: jnp.sum(
            flow_map.compute_guided_velocity_target(
                v_base,
                vc,
                v_uncond,
                cfg_scale,
                time_t,
                t_min,
                t_max,
                is_dropped,
            )[0]
        )
    )
    np.testing.assert_allclose(grad_fn(v_cond), np.zeros_like(v_cond))

  @parameterized.named_parameters(
      dict(testcase_name="one_step", num_steps=1, custom_grid=None),
      dict(testcase_name="two_steps", num_steps=2, custom_grid=None),
      dict(testcase_name="five_steps", num_steps=5, custom_grid=None),
      dict(
          testcase_name="custom_grid",
          num_steps=1,
          custom_grid=(0.0, 0.2, 0.7, 1.0),
      ),
  )
  def test_forward_and_backward_linear_ode_semigroup(
      self, num_steps: int, custom_grid: tuple[float, ...] | None
  ):
    """Exact semigroup check for linear ODE `dx/dt = a * x` (`a = 0.6`)."""
    a = 0.6
    x_0 = jnp.asarray([[1.0, -2.0], [0.5, 3.0]], dtype=jnp.float32)
    x_1_exact = x_0 * jnp.exp(a)

    # Exact forward flow map X_{s, t}(x) = exp(a * (t - s)) * x.
    def _exact_flow_map(x, t, s, cond=None):
      scale = cond["scale"][:, None] if cond is not None else a
      return x * jnp.exp(scale * (t - s)[:, None])

    fwd_grid = (
        jnp.asarray(custom_grid, dtype=jnp.float32)
        if custom_grid is not None
        else None
    )
    x_1_hat = flow_map.sample_forward_flow_map(
        _exact_flow_map, x_0, num_steps=num_steps, time_grid=fwd_grid
    )
    np.testing.assert_allclose(x_1_hat, x_1_exact, rtol=1e-5, atol=1e-5)

    # Conditional forward pass.
    cond = {"scale": jnp.asarray([a, a], dtype=jnp.float32)}
    x_1_cond = flow_map.sample_forward_flow_map(
        _exact_flow_map,
        x_0,
        num_steps=num_steps,
        cond=cond,
        time_grid=fwd_grid,
    )
    np.testing.assert_allclose(x_1_cond, x_1_exact, rtol=1e-5, atol=1e-5)

    # Exact backward average velocity u(x_t, t, s) = (x_t - x_s) / (t - s)
    # = ((1 - exp(-a * (t - s))) / (t - s)) * x_t.
    def _exact_mean_flow(x, t, s, cond=None):
      scale = cond["scale"][:, None] if cond is not None else a
      dt = (t - s)[:, None]
      return x * (-jnp.expm1(-scale * dt)) / jnp.maximum(dt, 1e-12)

    bwd_grid = (
        jnp.asarray(custom_grid[::-1], dtype=jnp.float32)
        if custom_grid is not None
        else None
    )
    x_0_hat = flow_map.sample_backward_mean_flow(
        _exact_mean_flow, x_1_exact, num_steps=num_steps, time_grid=bwd_grid
    )
    np.testing.assert_allclose(x_0_hat, x_0, rtol=1e-5, atol=1e-5)

    x_0_cond = flow_map.sample_backward_mean_flow(
        _exact_mean_flow,
        x_1_exact,
        num_steps=num_steps,
        cond=cond,
        time_grid=bwd_grid,
    )
    np.testing.assert_allclose(x_0_cond, x_0, rtol=1e-5, atol=1e-5)

  def test_sampler_grid_validation(self):
    x = jnp.zeros((2, 3))
    fn = lambda z, t, s: z
    with self.assertRaisesRegex(ValueError, "num_steps"):
      flow_map.sample_forward_flow_map(fn, x, num_steps=0)
    with self.assertRaisesRegex(ValueError, "time_grid"):
      flow_map.sample_backward_mean_flow(fn, x, time_grid=jnp.asarray([1.0]))
    with self.assertRaisesRegex(ValueError, "time_grid"):
      flow_map.sample_forward_flow_map(
          fn, x, time_grid=jnp.asarray([[0.0, 1.0]])
      )


if __name__ == "__main__":
  absltest.main()
