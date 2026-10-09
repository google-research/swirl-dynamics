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

"""Commonly-used metrics for regression tasks."""

from collections.abc import Sequence
import operator

import jax
import jax.numpy as jnp


def _union_axes(
    sum_axes: Sequence[int], mean_axes: Sequence[int], ndim: int
) -> tuple[int, ...]:
  """Returns distinct reduction axes without hiding out-of-bounds axes."""
  axes = []
  for axis in (*sum_axes, *mean_axes):
    axis = operator.index(axis)
    if not -ndim <= axis < ndim:
      raise ValueError(
          f"axis {axis} is out of bounds for array of dimension {ndim}"
      )
    axes.append(axis % ndim)
  return tuple(dict.fromkeys(axes))


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
  averaged over another set of axes. Inputs are promoted to at least float32
  before error arithmetic, and the result uses that promoted dtype.

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

  # Convert before subtraction, squaring and summation to avoid integer
  # wraparound and low-precision overflow before the mean is computed.
  dtype = jnp.result_type(pred, true, jnp.float32)
  pred = jnp.asarray(pred, dtype=dtype)
  true = jnp.asarray(true, dtype=dtype)

  if mean_axes is not None:
    mean_axes = _union_axes(sum_axes, mean_axes, pred.ndim)

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
  averaged over another set of axes. Inputs are promoted to at least float32
  before error arithmetic, and the result uses that promoted dtype.

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

  # Convert before subtraction, squaring and summation to avoid integer
  # wraparound and low-precision overflow before the mean is computed.
  dtype = jnp.result_type(pred, true, jnp.float32)
  pred = jnp.asarray(pred, dtype=dtype)
  true = jnp.asarray(true, dtype=dtype)

  if mean_axes is not None:
    mean_axes = _union_axes(sum_axes, mean_axes, pred.ndim)

  absolute_errors = jnp.sum(jnp.abs(pred - true), axis=sum_axes, keepdims=True)
  if relative:
    absolute_errors = absolute_errors / jnp.sum(
        jnp.abs(true), axis=sum_axes, keepdims=True
    )

  output_errors = jnp.mean(absolute_errors, axis=mean_axes)
  return output_errors
