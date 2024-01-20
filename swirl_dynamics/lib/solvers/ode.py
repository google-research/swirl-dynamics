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

"""Solvers for ordinary differential equations (ODEs)."""

from typing import Any, Protocol

import flax
import flax.linen as nn
import jax
from jax.experimental import checkify
from jax.experimental import ode
import jax.numpy as jnp
import numpy as np

Array = jax.Array
PyTree = Any


class OdeDynamics(Protocol):
  """Dynamics interface."""

  def __call__(self, x: Array, t: Array, params: PyTree) -> Array:
    """Evaluate the instantaneous dynamics."""
    ...


def nn_module_to_dynamics(
    module: nn.Module, autonomous: bool = True, **static_kwargs
) -> OdeDynamics:
  """Generates an `OdeDynamics` callable from a flax.nn module."""

  def _dynamics_func(x: Array, t: Array, params: PyTree) -> Array:
    args = (x,) if autonomous else (x, t)
    # NOTE: `params` here is the whole of model variable, not just the `params`
    # key
    variables = params
    return module.apply(variables, *args, **static_kwargs)

  return _dynamics_func


class OdeSolver(Protocol):
  """Solver interface."""

  def __call__(
      self, func: OdeDynamics, x0: Array, tspan: Array, params: PyTree
  ) -> Array:
    """Solves an ordinary differential equation."""
    ...


@flax.struct.dataclass
class ScanOdeSolver:
  """ODE solver based on `jax.lax.scan`.

  Attributes:
    time_axis_pos: move the time axis to the specified position in the output
      tensor (by default it is at the 0th position).
  """

  time_axis_pos: int = 0

  def step(
      self, func: OdeDynamics, x0: Array, t0: Array, dt: Array, params: PyTree
  ) -> Array:
    """Advances the current state one step forward in time."""
    raise NotImplementedError("Scan solver must implement `step` method.")

  def __call__(
      self, func: OdeDynamics, x0: Array, tspan: Array, params: PyTree
  ) -> Array:
    """Solves an ODE at given time stamps."""

    def scan_fun(
        state: tuple[Array, Array], t_next: Array
    ) -> tuple[tuple[Array, Array], Array]:
      x0, t0 = state
      dt = t_next - t0
      x_next = self.step(func, x0, t0, dt, params)
      return (x_next, t_next), x_next

    _, out = jax.lax.scan(scan_fun, (x0, tspan[0]), tspan[1:])
    out = jnp.concatenate([x0[None], out], axis=0)
    if self.time_axis_pos:
      out = jnp.moveaxis(out, 0, self.time_axis_pos)
    return out


class ExplicitEuler(ScanOdeSolver):
  """1st order Explicit Euler scheme."""

  def step(
      self, func: OdeDynamics, x0: Array, t0: Array, dt: Array, params: PyTree
  ) -> Array:
    return x0 + dt * func(x0, t0, params)


class HeunsMethod(ScanOdeSolver):
  """2nd order Heun's method."""

  def step(
      self, func: OdeDynamics, x0: Array, t0: Array, dt: Array, params: PyTree
  ) -> Array:
    k1 = func(x0, t0, params)
    k2 = func(x0 + dt * k1, t0 + dt, params)
    return x0 + dt * (k1 + k2) / 2


class RungeKutta4(ScanOdeSolver):
  """4th order Runge-Kutta scheme."""

  def step(
      self, func: OdeDynamics, x0: Array, t0: Array, dt: Array, params: PyTree
  ) -> Array:
    k1 = func(x0, t0, params)
    k2 = func(x0 + dt * k1 / 2, t0 + dt / 2, params)
    k3 = func(x0 + dt * k2 / 2, t0 + dt / 2, params)
    k4 = func(x0 + dt * k3, t0 + dt, params)
    return x0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class OneStepDirect(ScanOdeSolver):
  """Solver that directly returns function output as next time step."""

  def step(
      self, func: OdeDynamics, x0: Array, t0: Array, dt: Array, params: PyTree
  ) -> Array:
    """Performs a single prediction step: `x_{n+1} = f(x_n, t_n)`."""
    return func(x0, t0, params)


@flax.struct.dataclass
class DoPri45:
  """Adaptive-step Dormand-Prince solver."""

  rtol: float = 1.4e-8
  atol: float = 1.4e-8
  max_steps: int = jnp.iinfo(jnp.int32).max
  max_dt: float = jnp.inf

  def __call__(
      self, func: OdeDynamics, x0: Array, tspan: Array, params: PyTree
  ) -> Array:
    checkify.check(jnp.all(jnp.diff(tspan) > 0), "time needs to be increasing!")
    return ode.odeint(
        func,
        x0,
        tspan,
        params,
        rtol=self.rtol,
        atol=self.atol,
        mxstep=self.max_steps,
        hmax=self.max_dt,
    )


@flax.struct.dataclass
class MultiStepScanOdeSolver:
  """ODE solver based on `jax.lax.scan` that uses one than one time step.

  Rather than x_{n+1} = f(x_n, t_n), we have
  x_{n+1} = f(x_{n-k}, x_{n-k+1}, ..., x_{n-1}, x_n, t_{n-k}, ..., t_{n-1}, t_n)
  for some 'num_lookback_steps' window k.

  Attributes:
    time_axis_pos: move the time axis to the specified position in the output
      tensor (by default it is at the 0th position). This attribute is used to
      both indicate the temporal axis position of the input and where to place
      the temporal axis of the output.
  """

  time_axis_pos: int = 0

  def stack_timesteps_along_channel_dim(self, x: Array) -> Array:
    """Helper method to package batches for multi-step solvers.

    Args:
      x: Array of containing axes for batch_size (potentially), lookback_steps
        (e.g., temporal axis), spatial_dims, and channels, where spatial_dims
        can have ndim >= 1

    Returns:
      Array where each time step in the temporal dim is concatenated along
      the channel axis
    """
    x = jnp.moveaxis(x, self.time_axis_pos, -2)
    return jnp.reshape(x, x.shape[:-2] + (-1,))

  def step(
      self, func: OdeDynamics, x0: Array, t0: Array, dt: Array, params: PyTree
  ) -> Array:
    """Advances the current state one step forward in time."""
    raise NotImplementedError("Scan solver must implement `step` method.")

  def __call__(
      self,
      func: OdeDynamics,
      x0: Array,
      tspan: Array,
      params: PyTree,
  ) -> Array:
    """Solves an ODE at given time stamps by using k previous steps."""

    def scan_fun(
        state: tuple[Array, Array], t_next: Array
    ) -> tuple[tuple[Array, Array], Array]:
      # Expected dimension for x0 is either:
      # - (t, ...), if time_axis_pos == 0
      # - (batch_size, ..., t, ...), if time_axis_pos > 0
      x0, t0 = state
      x0_stack = self.stack_timesteps_along_channel_dim(x0)
      dt = t_next - t0
      x_next = self.step(func, x0_stack, t0, dt, params)
      x_next = jnp.expand_dims(x_next, axis=self.time_axis_pos)
      x_prev = jnp.take(
          x0,
          np.arange(1, x0.shape[self.time_axis_pos]),
          axis=self.time_axis_pos,
      )
      x_carry = jnp.concatenate([x_prev, x_next], axis=self.time_axis_pos)
      return (x_carry, t_next), x_next.squeeze(axis=self.time_axis_pos)

    _, out = jax.lax.scan(scan_fun, (x0, tspan[0]), tspan[1:])
    return jnp.concatenate(
        [x0, jnp.moveaxis(out, 0, self.time_axis_pos)], axis=self.time_axis_pos
    )


class MultiStepDirect(MultiStepScanOdeSolver):
  """Solver that directly returns function output as next time step."""

  def step(
      self, func: OdeDynamics, x0: Array, t0: Array, dt: Array, params: PyTree
  ) -> Array:
    """Performs a single prediction step."""
    return func(x0, t0, params)
