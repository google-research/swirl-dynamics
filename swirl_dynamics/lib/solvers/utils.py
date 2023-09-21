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

"""Utility functions."""

from typing import Any

import flax.linen as nn
import jax
from swirl_dynamics.lib.solvers import ode

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
PyTree = Any


def nn_module_to_ode_dynamics(
    module: nn.Module, autonomous: bool = True, **static_kwargs
) -> ode.OdeDynamics:
  """Generates an `OdeDynamics` callable from a flax.nn module."""

  def _dynamics_func(x: ArrayLike, t: ArrayLike, params: PyTree) -> Array:
    args = (x,) if autonomous else (x, t)
    # NOTE: `params` here is the whole of model variable, not just the `params`
    # key
    variables = params
    return module.apply(variables, *args, **static_kwargs)

  return _dynamics_func
