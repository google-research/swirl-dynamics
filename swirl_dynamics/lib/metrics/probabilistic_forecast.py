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

"""Commonly-used metrics for probabilistic forecasting tasks."""

import jax
import jax.numpy as jnp

Array = jax.Array


def _process_forecasts(
    forecasts: Array, observations: Array, ensemble_axis: int
) -> Array:
  """Processes the forecast array and validates its shape."""
  # move ensemble axis to last so that shape checking and broadcasting is easier
  if ensemble_axis != -1:
    forecasts = jnp.moveaxis(forecasts, ensemble_axis, -1)

  if forecasts.shape[:-1] != observations.shape:
    raise ValueError(
        "`forecasts` and `observations` must have matching shapes except along"
        f" the ensemble axis: {forecasts.shape[:-1]} vs {observations.shape}."
    )
  return forecasts


def _mean_abs_diff_with_broadcast(forecasts: Array) -> Array:
  """Computes the mean absolute difference of ensembles with broadcasting."""
  return jnp.mean(
      jnp.abs(jnp.expand_dims(forecasts, -1) - jnp.expand_dims(forecasts, -2)),
      axis=(-2, -1),
  )


def _mean_abs_diff_with_loop(forecasts: Array) -> Array:
  """Computes the mean absolute difference of ensembles with `lax.fori_loop`."""

  def _sum_abs_diff(i, abs_diff):
    return abs_diff + jnp.abs(forecasts[..., i, None] - forecasts).mean(axis=-1)

  mean_abs_diff = jax.lax.fori_loop(
      0, forecasts.shape[-1], _sum_abs_diff, jnp.zeros(forecasts.shape[:-1])
  )
  return mean_abs_diff / forecasts.shape[-1]


def crps(
    forecasts: Array,
    observations: Array,
    *,
    ensemble_axis: int = 1,
    direct_broadcast: bool = True,
) -> Array:
  """Calculates the continuous ranked probability score (CRPS).

  Formula:

    CRPS = E |x - y| - 0.5 * E |x - x'|

  where E denotes the expected value, y denotes the observations, x denotes a
  random member drawn from the ensemble forecasts, and x' is another sample
  indepedent from x.

  Reference:
    https://www.stat.washington.edu/research/reports/2004/tr463R.pdf

  Args:
    forecasts: Ensemble forecast members.
    observations: Observations based on which to score the ensemble forecasts,
      with the same shape of the forecasts except for missing the ensemble axis.
    ensemble_axis: The axis of forecasts corresponding to the ensemble members.
    direct_broadcast: Whether to directly broadcast when computing the mean abs
      difference part of the score, which is efficient but memory intensive. If
      `False`, the mean abs diff is instead computed with a loop, which requires
      less memory but also slower.

  Returns:
    The CRPS with same shape as the observations.
  """
  forecasts = _process_forecasts(forecasts, observations, ensemble_axis)
  mae = jnp.mean(jnp.abs(forecasts - observations[..., None]), axis=-1)

  if direct_broadcast:
    mean_abs_diff = _mean_abs_diff_with_broadcast(forecasts)
  else:
    mean_abs_diff = _mean_abs_diff_with_loop(forecasts)

  return mae - 0.5 * mean_abs_diff


def threshold_brier_score(
    forecasts: Array,
    observations: Array,
    thresholds: Array | int | float,
    *,
    ensemble_axis: int = 1,
):
  """Calculates the Brier scores of an ensemble exceeding given thresholds.

  The Brier score (BS) scores binary forecasts k = 0 or 1:

    BS = (p_1 - k)^2,

  where p_1 is the forecast probability of k = 1. The threshold BS simply
  computes p_1 as the fraction of forecasts exceeding a given threshold. The
  same threshold is used to transform the observations into binary values.  Note
  that counting above or below the thresholds correspond to the same scores,
  provided that it is applied consistently between forecasts and observations.

  Args:
    forecasts: Ensemble forecast members.
    observations: Observations based on which to score the ensemble forecasts,
      with the same shape of the forecasts except for missing the ensemble axis.
    thresholds: Threshold values wrt which the Brier scores are calculated. Must
      be a scalar (single threshold) or a 1-dimensional array (multiple
      thresholds at the same time).
    ensemble_axis: The axis of forecasts corresponding to the ensemble members.

  Returns:
    The threshold Brier score with shape of either
      * observations.shape, if thresholds is 0-dimensional
      * observations.shape + thresholds.shape, if thresholds is 1-dimensional
  """
  thresholds = jnp.asarray(thresholds)
  if thresholds.ndim > 1:
    raise ValueError("`thresholds` must be 0- or 1-dimensional.")

  forecasts = _process_forecasts(forecasts, observations, ensemble_axis)
  binary_obs = jnp.greater(
      observations[..., None],
      thresholds.reshape((1,) * observations.ndim + (-1,)),
  )
  binary_obs = jnp.asarray(binary_obs, dtype=jnp.float32)
  prob_forecasts = jnp.greater(
      forecasts[..., None], thresholds.reshape((1,) * forecasts.ndim + (-1,))
  )
  prob_forecasts = jnp.mean(prob_forecasts, axis=-2)
  score = (binary_obs - prob_forecasts) ** 2

  if thresholds.ndim == 0:
    score = jnp.squeeze(score, axis=-1)
  return score
