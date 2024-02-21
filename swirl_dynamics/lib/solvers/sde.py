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

"""Solvers for stochastic differential equations (SDEs)."""

from collections.abc import Mapping
import functools
from typing import Any, ClassVar, Literal, NamedTuple, Protocol
import warnings

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
  """The drift and diffusion functions that represent the SDE dynamics."""

  drift: SdeCoefficientFn
  diffusion: SdeCoefficientFn


def _check_sde_params_fields(params: SdeParams) -> None:
  if not ("drift" in params.keys() and "diffusion" in params.keys()):
    raise ValueError(
        "`params` must contain both `drift` and `diffusion` fields."
    )


class SdeSolver(Protocol):
  """A callable type implementing a SDE solver.

  Attributes:
    terminal_only: If `True`, the solver only returns the terminal state, i.e.
      corresponding to the last time stamp in `tspan`. If `False`, returns the
      full path containing all steps.
  """

  terminal_only: bool

  def __call__(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      tspan: Array,
      rng: Array,
      params: PyTree,
  ) -> Array:
    """Solves a SDE at given time stamps.

    Args:
      dynamics: The SDE dynamics that evaluates the drift and diffusion
        coefficients.
      x0: Initial condition.
      tspan: The sequence of time points on which the approximate solution of
        the SDE are evaluated. The first entry corresponds to the time for x0.
      rng: The root Jax random key used to draw realizations of the Wiener
        processes for the SDE.
      params: Parameters for the dynamics. Must contain both a "drift" and a
        "diffusion" field.

    Returns:
      Integrated SDE trajectory (initial condition included at time position 0).
    """
    ...


@flax.struct.dataclass
class ScanSdeSolver:
  """A SDE solver based on `jax.lax.scan`.

  Attributes:
    time_axis_pos: The index where the time axis should be placed. Defaults to
      the lead axis (index 0).
  """

  time_axis_pos: int = 0
  terminal_only: ClassVar[bool] = False

  def step(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      t0: Array,
      dt: Array,
      rng: Array,
      params: SdeParams,
  ) -> Array:
    """Advances the current state one step forward in time."""
    raise NotImplementedError

  def __call__(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      tspan: Array,
      rng: Array,
      params: SdeParams,
  ) -> Array:
    """Solves a SDE by integrating the step function with `jax.lax.scan`."""

    def scan_fun(
        state: tuple[Array, Array],
        ext: tuple[Array, Array],
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


class LoopSdeSolver:
  """A SDE solver based on `jax.lax.while_loop`.

  Compared to the `scan` based version, the while-loop-based solver is more
  memory friendly. As a tradeoff, this implementation is not reverse-mode
  differentiable.
  """

  terminal_only: bool = True

  def step(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      t0: Array,
      dt: Array,
      rng: Array,
      params: SdeParams,
  ) -> Array:
    """Advances the current state one step forward in time."""
    raise NotImplementedError

  def __call__(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      tspan: Array,
      rng: Array,
      params: SdeParams,
  ) -> Array:
    """Solves a SDE by integrating the step function with `lax.while_loop`."""
    # Rng splits based on the length of `tspan` so that the results are
    # identical to that of the scan verson.
    rngs = jax.random.split(rng, num=len(tspan))
    step_fn = functools.partial(self.step, dynamics)

    def cond_fn(loop_state: tuple[int, Array]) -> bool:
      return loop_state[0] < len(tspan) - 1

    def body_fn(loop_state: tuple[int, Array]) -> tuple[int, Array]:
      i, xi = loop_state
      dt = tspan[i + 1] - tspan[i]
      x_next = step_fn(x0=xi, t0=tspan[i], dt=dt, rng=rngs[i], params=params)
      return i + 1, x_next

    _, out = jax.lax.while_loop(cond_fn, body_fn, (0, x0))
    return out


class EulerMaruyamaStep:
  """The Euler-Maruyama scheme for integrating the Ito SDE."""

  def step(
      self,
      dynamics: SdeDynamics,
      x0: Array,
      t0: Array,
      dt: Array,
      rng: Array,
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


class ScanBasedEulerMaruyama(EulerMaruyamaStep, ScanSdeSolver):
  ...


class LoopBasedEulerMaruyama(EulerMaruyamaStep, LoopSdeSolver):
  ...


def EulerMaruyama(  # pylint: disable=invalid-name
    iter_type: Literal["scan", "loop"] = "scan",
    time_axis_pos: int | None = None,
) -> SdeSolver:
  match iter_type:
    case "scan":
      time_axis_pos = time_axis_pos or 0
      return ScanBasedEulerMaruyama(time_axis_pos=time_axis_pos)
    case "loop":
      if time_axis_pos is not None:
        warnings.warn(
            "`time_axis_pos` is set but does not apply to loop-based solver."
        )
      return LoopBasedEulerMaruyama()
