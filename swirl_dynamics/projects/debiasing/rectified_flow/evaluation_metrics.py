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

"""Statistical Evaluation metrics."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


# TODO: add test for the functions in this file.
def _energy(x: jax.Array) -> jax.Array:
  """Computes energy of a given array."""
  return jnp.square(jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(x))))


def radial_spectra(
    num_bins: int = 64, lon: int = 240, lat: int = 121
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
  """Computes radial spectra of a given array.

  Args:
    num_bins: Number of bins for the energy spectrum.
    lon: Number of discretization point in the longitudinal direction.
    lat: Number of discretization points in the meridional direction.

  Returns:
    A tuple with the function that computes the radial spectra and the
    frequencies associated to each bin (without the 0-th frequency)
  """
  freqs_lon = np.fft.fftshift(np.fft.fftfreq(lon, d=1 / lon))
  freqs_lat = np.fft.fftshift(np.fft.fftfreq(lat, d=1 / lon))
  kx, ky = np.meshgrid(freqs_lon, freqs_lat, indexing="ij")
  k = np.sqrt(kx**2 + ky**2)
  bins = np.linspace(np.min(k), np.max(k), num=num_bins)
  indices = np.digitize(k, bins)

  def _radial_spectra(x):
    energy = _energy(x)
    energy_k = lambda kk: jnp.sum((indices == kk) * energy)
    rad_energy = jax.vmap(energy_k)(jnp.arange(1, num_bins))
    return rad_energy

  return jax.jit(_radial_spectra), bins[1:]


def build_vectorized_function(
    num_bins: int = 64, lon: int = 240, lat: int = 121
) -> Callable[[jax.Array], jax.Array]:
  """Builds vectorized function radial spectra.

  Args:
    num_bins: Number of bins for the energy spectrum.
    lon: Number of discretization point in the longitudinal direction.
    lat: Number of discretization points in the meridional direction.

  Returns:
    A function that computes the radial spectra of a given array.
  """
  rad_spec_fn, _ = radial_spectra(num_bins, lon, lat)
  vmapped_radspec = jax.vmap(rad_spec_fn)

  def sample_radspec(samples: jax.Array) -> jax.Array:
    spec = vmapped_radspec(samples)
    return np.mean(spec, axis=0)

  return sample_radspec


def mean_log_ratio(
    radspec1: jax.Array, radspec2: jax.Array, range_freq: slice
) -> jax.Array:
  """Computes the mean log ratio of two radial spectra.

  Args:
    radspec1: First radial spectra.
    radspec2: Second radial spectra.
    range_freq: Range of frequencies to compute the error.

  Returns:
    The mean log ratio between the two radial spectra.
  """
  return jnp.mean(
      jnp.abs(jnp.log10(radspec1[range_freq] / radspec2[range_freq]))
  )


def weighted_log_ratio(
    radspec1: jax.Array, radspec2: jax.Array, range_freq: slice
) -> jax.Array:
  """Computes the weighted log ratio of two radial spectra.

  Args:
    radspec1: First radial spectra.
    radspec2: Second radial spectra.
    range_freq: Range of frequencies to compute the error.

  Returns:
    The weighted log ratio between the two radial spectra.
  """
  weight = radspec1[range_freq] / jnp.sum(radspec1[range_freq])
  return jnp.sum(
      weight * jnp.abs(jnp.log10(radspec1[range_freq] / radspec2[range_freq]))
  )


def wass1_error(
    input_array: jax.Array,
    output_array: jax.Array,
    target_array: jax.Array,
    variables: Sequence[str],
    range_hist: tuple[float, float] | None = None,
    bins_hist: int = 201,
) -> dict[str, dict[str, jax.Array]]:
  """Computes the Wasserstein1 error.

  Args:
    input_array: LENS2 data in a collated tensor.
    output_array: Ouput of the Reflow model corresponding to the LENS2
      input_array.
    target_array: Target (by date) ERA5 data.
    variables: List of variables as collated in the channel dimension.
    range_hist: Tuple with the numerical range for this histograms.
    bins_hist: Number of bins in the histogram.

  Returns:
    A dictionary with the Wasserstein-1 error per variable.
  """

  num_fields = len(variables)

  dict_err = {}

  for field_idx in range(num_fields):
    if not range_hist:
      min_u = jnp.min(
          jnp.concat([
              target_array[..., field_idx].reshape((-1,)),
              output_array[..., field_idx].reshape((-1,)),
              input_array[..., field_idx].reshape((-1,)),
          ])
      )
      max_u = jnp.max(
          jnp.concat([
              target_array[..., field_idx].reshape((-1,)),
              output_array[..., field_idx].reshape((-1,)),
              input_array[..., field_idx].reshape((-1,)),
          ])
      )
      range_hist_loc = (min_u, max_u)
    else:
      range_hist_loc = range_hist

    cnts_target, _ = jnp.histogram(
        target_array[..., field_idx].reshape((-1,)),
        bins=bins_hist,
        range=range_hist_loc,
        density=True,
    )

    cnts_output, _ = jnp.histogram(
        output_array[..., field_idx].reshape((-1,)),
        bins=bins_hist,
        range=range_hist_loc,
        density=True,
    )

    cnts_input, bins = jnp.histogram(
        input_array[..., field_idx].reshape((-1,)),
        bins=bins_hist,
        range=range_hist_loc,
        density=True,
    )

    # Computing L^1 errors.
    l1_err = jnp.sum(jnp.abs(cnts_target - cnts_output)) * (bins[1] - bins[0])
    l1_err_ref = jnp.sum(jnp.abs(cnts_input - cnts_target)) * (
        bins[1] - bins[0]
    )

    dict_err[f"{variables[field_idx]}"] = {}
    dict_err[f"{variables[field_idx]}"]["l1_error"] = l1_err
    dict_err[f"{variables[field_idx]}"]["l1_ref_error"] = l1_err_ref

    dict_err[f"{variables[field_idx]}"]["distribution_output"] = cnts_output
    dict_err[f"{variables[field_idx]}"]["distribution_target"] = cnts_target
    dict_err[f"{variables[field_idx]}"]["distribution_input"] = cnts_input
    dict_err[f"{variables[field_idx]}"]["bins"] = bins

    print(
        f"Error for pdf for {variables[field_idx]} is \t {l1_err}, and"
        f" reference error is \t {l1_err_ref}"
    )

  return dict_err


def log_energy_error(
    input_array: jax.Array,
    output: jax.Array,
    target: jax.Array,
    variables: Sequence[str],
    num_bins: int = 64,
    lon: int = 240,
    lat: int = 121,
) -> dict[str, dict[str, jax.Array]]:
  """Computes log energy error.

  Args:
    input_array: LENS2 input data with dimensions (num_samples, lon, lat,
      num_fields).
    output: Reflow debiased data with the same dimensions as the input.
    target: Reference ERA5 data with the same dimensions as the input.
    variables: List of physical variables in the snapshots.
    num_bins: Number of bins for the energy spectrum.
    lon: Number of discretization point in the longitudinal direction.
    lat: Number of discretization points in the meridional direction.

  Returns:
    A dictionary with the errors per field.

  Raises:
    ValueError:
  """
  num_fields = len(variables)
  if num_fields != input_array.shape[-1]:
    raise ValueError(
        f"Number of fields ({num_fields}) does not match the ",
        f"number of channels if the input fields ({input_array.shape[-1]}).",
    )

  sample_radspec = build_vectorized_function(num_bins, lon, lat)

  err_log = {}
  ref_err_log = {}
  err_wlog = {}
  ref_err_wlog = {}

  for field_idx in range(num_fields):
    era4_field = sample_radspec(target[:, :, :, field_idx])
    lens2_field = sample_radspec(input_array[:, :, :, field_idx])
    reflow_field = sample_radspec(output[:, :, :, field_idx])

    # change how to error is stored.
    err_log[variables[field_idx]] = mean_log_ratio(
        era4_field, reflow_field, slice(0, -1)
    )
    err_wlog[variables[field_idx]] = weighted_log_ratio(
        reflow_field, era4_field, slice(0, -1)
    )
    ref_err_log[variables[field_idx]] = mean_log_ratio(
        lens2_field, era4_field, slice(0, -1)
    )
    ref_err_wlog[variables[field_idx]] = weighted_log_ratio(
        lens2_field, era4_field, slice(0, -1)
    )

  dict_metrics = {
      "err_log_ratio": err_log,
      "weighted_err_log_ratio": err_wlog,
      "ref_err_log_ratio": ref_err_log,
      "ref_weighted_err_log_ratio": ref_err_wlog,
  }
  return dict_metrics


def smoothed_average_l1_error(
    input_array: jax.Array,
    output: jax.Array,
    target: jax.Array,
    variables: Sequence[str],
    window_size: int = 365,
) -> dict[str, dict[str, jax.Array]]:
  """Computes the l1 error of between global averages smoothed in time.

  Args:
    input_array: LENS2 input data with dimensions (num_samples, lon, lat,
      num_fields).
    output: Reflow debiased data with the same dimensions as the input.
    target: Reference ERA5 data with the same dimensions as the input.
    variables: List of physical variables in the snapshots.
    window_size: The size of the window for the smoothing in time.

  Returns:
    A dictionary with the errors per field.

  Raises:
    ValueError:
  """
  num_fields = len(variables)
  if num_fields != input_array.shape[-1]:
    raise ValueError(
        f"Number of fields ({num_fields}) does not match the ",
        f"number of channels if the input fields ({input_array.shape[-1]}).",
    )

  err_mean = {}
  ref_err_mean = {}
  diff_mean = {}
  global_mean_lens2 = {}
  global_mean_era5 = {}
  global_mean_reflow = {}

  for field_idx in range(num_fields):
    # Compute global averages per snapshot.
    era5_mean = np.mean(target[:, :, :, field_idx], axis=(-1, -2))
    lens2_mean = np.mean(input_array[:, :, :, field_idx], axis=(-1, -2))
    reflow_mean = np.mean(output[:, :, :, field_idx], axis=(-1, -2))

    # Compute the moving average.
    conv_window = np.ones(window_size)/window_size
    reflow_mean = np.convolve(reflow_mean, conv_window, mode="valid")
    lens2_mean = np.convolve(lens2_mean, conv_window, mode="valid")
    era5_mean = np.convolve(era5_mean, conv_window, mode="valid")

    # Compute the l1 error.
    err_mean[variables[field_idx]] = np.mean(np.abs(reflow_mean - era5_mean))
    diff_mean[variables[field_idx]] = np.mean(np.abs(reflow_mean - lens2_mean))
    ref_err_mean[variables[field_idx]] = np.mean(np.abs(lens2_mean - era5_mean))

    # Save the global means.
    global_mean_lens2[variables[field_idx]] = lens2_mean
    global_mean_era5[variables[field_idx]] = era5_mean
    global_mean_reflow[variables[field_idx]] = reflow_mean

  dict_metrics = {
      "err_mean": err_mean,
      "ref_err_mean": ref_err_mean,
      "diff_mean": diff_mean,
      "global_mean_lens2": global_mean_lens2,
      "global_mean_era5": global_mean_era5,
      "global_mean_reflow": global_mean_reflow,
  }
  return dict_metrics
