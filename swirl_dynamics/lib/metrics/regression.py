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

"""Commonly-used metrics for regression tasks."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp


def mean_squared_error(
    pred: jax.Array,
    true: jax.Array,
    *,
    sum_axes: Sequence[int] = (),
    mean_axes: Sequence[int] | None = None,
    relative: bool = False,
    squared: bool = True,
) -> jax.Array:
  """Computes the mean squared error (MSE).

  The squared errors are first summed over a specified set of axes and then
  averaged over another set of axes.

  Args:
    pred: The array representing the predictions.
    true: The array representing the ground truths.
    sum_axes: The axes over which the squared errors will be summed before
      taking the average.
    mean_axes: The axes over which the average will be taken. If `None`, average
      is taken over all axes. If some elements are common between `sum_axes` and
      `mean_axes`, the the former takes priority.
    relative: Whether to compute the relative MSE. If `True`, the errors are
      normalized by the squared norm of `true`.
    squared: Whether to keep the returned error squared. If `False`, returns the
      square-rooted error, i.e. the root mean squared error.

  Returns:
    The computed MSE array.
  """
  if pred.shape != true.shape:
    raise ValueError(
        f"`pred` {pred.shape} and `true` {true.shape} must have the same shape."
    )

  if mean_axes is not None:
    mean_axes = tuple(sum_axes) + tuple(mean_axes)

  squared_errors = jnp.sum(
      jnp.square(pred - true), axis=sum_axes, keepdims=True
  )
  if relative:
    squared_errors = squared_errors / jnp.sum(
        jnp.square(true), axis=sum_axes, keepdims=True
    )

  output_errors = jnp.mean(squared_errors, axis=mean_axes)
  if not squared:
    output_errors = jnp.sqrt(output_errors)

  return output_errors


def mean_absolute_error(
    pred: jax.Array,
    true: jax.Array,
    *,
    sum_axes: Sequence[int] = (),
    mean_axes: Sequence[int] | None = None,
    relative: bool = True,
) -> jax.Array:
  """Computes the mean absolute error (MAE).

  The absolute errors are first summed over a specified set of axes and then
  averaged over another set of axes.

  Args:
    pred: The array representing the predictions.
    true: The array representing the ground truths.
    sum_axes: The axes over which the absolute errors will be summed before
      taking the average.
    mean_axes: The axes over which the average will be taken. If `None`, average
      is taken over all axes. If some elements are common between `sum_axes` and
      `mean_axes`, the the former takes priority.
    relative: Whether to compute the relative MSE. If `True`, the errors are
      normalized by the L1 norm of `true`.

  Returns:
    The computed MAE array.
  """
  if pred.shape != true.shape:
    raise ValueError(
        f"`pred` {pred.shape} and `true` {true.shape} must have the same shape."
    )

  if mean_axes is not None:
    mean_axes = tuple(sum_axes) + tuple(mean_axes)

  absolute_errors = jnp.sum(jnp.abs(pred - true), axis=sum_axes, keepdims=True)
  if relative:
    absolute_errors = absolute_errors / jnp.sum(
        jnp.abs(true), axis=sum_axes, keepdims=True
    )

  output_errors = jnp.mean(absolute_errors, axis=mean_axes)
  return output_errors
