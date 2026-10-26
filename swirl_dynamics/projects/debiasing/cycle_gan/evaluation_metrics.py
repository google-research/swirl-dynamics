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

"""Distributional and spectral metrics for 1-D debiasing experiments.

All functions are pure JAX and operate on arrays of samples. Notation: the
"PDF" of an array is the normalised histogram of all its entries, `W_1` is the
one-dimensional Wasserstein-1 (earth mover's) distance between the empirical
distributions of all entries, and the energy spectrum is the mean (over
samples and channels) squared modulus of the real FFT along the spatial axis.
"""

import jax
import jax.numpy as jnp

Array = jax.Array


def pdf_l1_error(
    prediction: Array,
    target: Array,
    value_range: tuple[float, float] | None = None,
    bins: int = 201,
) -> Array:
  """L1 distance between the normalised histograms of two arrays.

  This is the "wass1_error" of the rectified-flow package, named for what it
  computes: `sum_k |p_k - q_k| * bin_width` with `p`, `q` the density
  histograms of all entries of `prediction` and `target`. It is bounded by 2
  and is *not* the Wasserstein-1 distance (see `wasserstein1_1d`).

  Args:
    prediction: Predicted samples (any shape).
    target: Target samples (any shape).
    value_range: Range of the histograms; the joint min/max when `None`.
    bins: Number of histogram bins.

  Returns:
    The scalar L1 distance between the two densities.
  """
  prediction = prediction.reshape((-1,))
  target = target.reshape((-1,))
  if value_range is None:
    both = jnp.concatenate([prediction, target])
    histogram_range = (jnp.min(both), jnp.max(both))
  else:
    histogram_range = value_range
  counts_prediction, _ = jnp.histogram(
      prediction, bins=bins, range=histogram_range
  )
  counts_target, _ = jnp.histogram(target, bins=bins, range=histogram_range)
  mass_prediction = counts_prediction / jnp.maximum(prediction.size, 1)
  mass_target = counts_target / jnp.maximum(target.size, 1)
  return jnp.where(
      histogram_range[0] == histogram_range[1],
      jnp.zeros((), dtype=mass_prediction.dtype),
      jnp.sum(jnp.abs(mass_prediction - mass_target)),
  )


def wasserstein1_1d(prediction: Array, target: Array) -> Array:
  """Wasserstein-1 distance between the empirical distributions of all entries.

  Uses the quantile form `W_1 = int_0^1 |F_p^{-1}(q) - F_t^{-1}(q)| dq`, with
  the empirical quantile functions evaluated at the midpoints of
  `max(N_p, N_t)` equal probability bins. This is exact when the smaller
  sample count divides the larger one (in particular for `N_p == N_t`), since
  both step functions are then constant on every bin. Otherwise it is a
  midpoint-rule approximation whose error is at most `range / max(N_p, N_t)`:
  the integrand is 1-Lipschitz in each quantile function, which jumps by at
  most the sample range in total across the bins that straddle a jump.

  Args:
    prediction: Predicted samples (any shape, `N_p` entries).
    target: Target samples (any shape, `N_t` entries).

  Returns:
    The scalar distance (same units as the samples).
  """
  prediction = jnp.sort(prediction.reshape((-1,)))
  target = jnp.sort(target.reshape((-1,)))
  num_quantiles = max(prediction.shape[0], target.shape[0])
  # Midpoints of `num_quantiles` equal probability bins.
  quantiles = (jnp.arange(num_quantiles) + 0.5) / num_quantiles

  def _inverse_cdf(sorted_samples: Array) -> Array:
    index = jnp.floor(quantiles * sorted_samples.shape[0]).astype(jnp.int32)
    return sorted_samples[jnp.clip(index, 0, sorted_samples.shape[0] - 1)]

  return jnp.mean(jnp.abs(_inverse_cdf(prediction) - _inverse_cdf(target)))


def energy_spectrum_1d(samples: Array) -> Array:
  """Mean energy spectrum of 1-D samples along the spatial axis.

  Args:
    samples: Array of shape `(num_samples, length, channels)`.

  Returns:
    The spectrum `E_k = mean |FFT(u)_k|^2` over samples and channels, of shape
    `(length // 2 + 1,)`.

  Raises:
    ValueError: If `samples` is not 3-D.
  """
  if samples.ndim != 3:
    raise ValueError(
        "energy_spectrum_1d expects shape (num_samples, length, channels),"
        f" got {samples.shape}."
    )
  spectrum = jnp.abs(jnp.fft.rfft(samples, axis=1)) ** 2
  return jnp.mean(spectrum, axis=(0, 2))


def spectral_log_ratio(
    prediction: Array, target: Array, floor: float = 1e-30
) -> Array:
  """Energy-weighted mean absolute log10 ratio of two energy spectra.

  The weights are the normalised target spectrum, so the error concentrates on
  the energetic wavenumbers, following the spectral metric of the
  rectified-flow package. The constant mode is excluded.

  Args:
    prediction: Predicted samples of shape `(num_samples, length, channels)`.
    target: Target samples of the same spatial length.
    floor: Lower bound applied to both spectra before taking the logarithm.

  Returns:
    The scalar `sum_k w_k |log10(E^pred_k / E^target_k)|` over `k >= 1`.

  Raises:
    ValueError: If `prediction` and `target` are not 3-D with the same spatial
      length.
  """
  spectrum_prediction = jnp.maximum(energy_spectrum_1d(prediction)[1:], floor)
  spectrum_target = jnp.maximum(energy_spectrum_1d(target)[1:], floor)
  if prediction.shape[1] != target.shape[1]:
    raise ValueError(
        f"Spatial lengths must match, got {prediction.shape[1]} !="
        f" {target.shape[1]}."
    )
  weights = spectrum_target / jnp.sum(spectrum_target)
  return jnp.sum(
      weights * jnp.abs(jnp.log10(spectrum_prediction / spectrum_target))
  )


def cycle_consistency_error(samples: Array, reconstructions: Array) -> Array:
  """Relative L2 error of the cycle reconstructions `F(G(x))` of `x`.

  Args:
    samples: Original samples (any shape).
    reconstructions: Their reconstructions, same shape.

  Returns:
    `||reconstructions - samples||_2 / ||samples||_2` over all entries.

  Raises:
    ValueError: If `samples` and `reconstructions` do not have the same shape.
  """
  if samples.shape != reconstructions.shape:
    raise ValueError(
        "samples and reconstructions must have the same shape, got"
        f" {samples.shape} != {reconstructions.shape}."
    )
  diff_norm = jnp.linalg.norm((reconstructions - samples).reshape((-1,)))
  samples_norm = jnp.linalg.norm(samples.reshape((-1,)))
  safe_samples_norm = jnp.where(samples_norm == 0, 1.0, samples_norm)
  return jnp.where(
      samples_norm == 0,
      jnp.where(diff_norm == 0, jnp.zeros_like(diff_norm), jnp.inf),
      diff_norm / safe_samples_norm,
  )
