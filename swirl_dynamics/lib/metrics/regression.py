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

"""Commonly-used metric functions for regression tasks."""

from collections.abc import Sequence
import jax
import jax.numpy as jnp

Array = jax.Array
Axis = int | Sequence[int] | None


def l2_norm(x: Array, axis: Axis = None) -> Array:
  return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis))


def l2_dist(x1: Array, x2: Array, axis: Axis = None) -> Array:
  return jnp.linalg.norm(x1 - x2, axis=axis)


def l2_err(
    *, pred: Array, true: Array, norm_axis: Axis, relative: bool
) -> Array:
  """L2 absolute or relative error along selected dimensions."""
  err = l2_dist(pred, true, axis=norm_axis)
  if relative:
    err = err / l2_norm(true, axis=norm_axis)
  return err
