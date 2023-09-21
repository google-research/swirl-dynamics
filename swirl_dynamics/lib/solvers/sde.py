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

"""Solvers for stochastic differential equations (SDEs)."""

from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

import flax
import jax
import jax.numpy as jnp

Array = jax.Array
PyTree = Any
SdeParams = Mapping[str, PyTree]


class SdeCoefficientFn(Protocol):
  """A callable type for the drift or diffusion coefficients of a SDE."""

  def __call__(self, x: Array, t: Array, params: PyTree) -> Array:
    """Evaluates the drift or diffusion coefficients."""
    ...


class SdeDynamics(NamedTuple):
  """A pair of drift and diffusion functions that represent the SDE dynamics."""

  drift: SdeCoefficientFn
  diffusion: SdeCoefficientFn


def _check_sde_params_fields(params: SdeParams) -> None:
  if not ("drift" in params.keys() and "diffusion" in params.keys()):
    raise ValueError(
        "`params` must contain both `drift` and `diffusion` fields."
    )


class SdeSolver(Protocol):
  """A callable type implementing a SDE solver."""

  def __call__(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      tspan: Array,
      rng: jax.random.KeyArray,
      params: PyTree,
  ) -> Array:
    """Solves a SDE at given time stamps.

    Args:
      dynamics: The SDE dynamics object that evaluates the drift and diffusion
        coefficients.
      x0: The initial condition.
      tspan: The sequence of time points for which to solve the SDE. The first
        entry is assumed to correspond to the time for x0.
      rng: The root Jax random key used to draw realizations of the Wiener
        processes for the SDE.
      params: The parameters for the dynamics; must contain both a "drift" and a
        "diffusion" field.

    Returns:
      Integrated SDE trajectory (initial condition included at time position 0).
    """
    ...


@flax.struct.dataclass
class ScanSdeSolver:
  """A SDE solver based on `jax.lax.scan`.

  Attributes:
    time_axis_pos: move the time axis to the specified position in the output
      tensor (by default it is at the 0th position).
  """

  time_axis_pos: int = 0

  def step(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      t0: Array,
      dt: Array,
      rng: jax.random.KeyArray,
      params: SdeParams,
  ) -> Array:
    """Advances the current state one step forward in time."""
    raise NotImplementedError

  def __call__(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      tspan: Array,
      rng: jax.random.KeyArray,
      params: SdeParams,
  ) -> Array:
    """Solves a SDE by integrating the step function with `jax.lax.scan`."""

    def scan_fun(
        state: tuple[Array, Array],
        ext: tuple[Array, jax.random.KeyArray],
    ) -> tuple[tuple[Array, Array], Array]:
      x0, t0 = state
      t_next, step_rng = ext
      dt = t_next - t0
      x_next = self.step(dynamics, x0, t0, dt, step_rng, params)
      return (x_next, t_next), x_next

    step_rngs = jax.random.split(rng, len(tspan))
    _, out = jax.lax.scan(scan_fun, (x0, tspan[0]), (tspan[1:], step_rngs[:-1]))
    out = jnp.concatenate([x0[None], out], axis=0)
    if self.time_axis_pos:
      out = jnp.moveaxis(out, 0, self.time_axis_pos)
    return out


class EulerMaruyama(ScanSdeSolver):
  """The Euler-Maruyama scheme for integrating the Ito SDE."""

  def step(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      t0: Array,
      dt: Array,
      rng: jax.random.KeyArray,
      params: SdeParams,
  ) -> Array:
    """Makes one Euler-Maruyama integration step in time."""
    _check_sde_params_fields(params)
    drift_coeffs = dynamics.drift(x0, t0, params["drift"])
    diffusion_coeffs = dynamics.diffusion(x0, t0, params["diffusion"])
    noise = jax.random.normal(rng, x0.shape, x0.dtype)
    return (
        x0
        + dt * drift_coeffs
        # `abs` to enable integration backward in time
        + diffusion_coeffs * noise * jnp.sqrt(jnp.abs(dt))
    )
