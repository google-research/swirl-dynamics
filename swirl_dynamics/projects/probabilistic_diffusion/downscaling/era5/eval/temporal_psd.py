# Copyright 2025 The swirl_dynamics Authors.
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

"""Eval downscaling results for temporal power spectral density."""

from collections.abc import Mapping, Sequence
from typing import Any

from absl import logging
from etils import epath
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate as interp
import seaborn as sns
from swirl_dynamics.data import hdf5_utils as h5u
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.eval import utils as eval_utils
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils as pipeline_utils
import xarray as xr

filesys = epath.backend.tf_backend

sns.set_style("white")
sns.set_palette("colorblind")


def _calculate_melr(
    ref_freq: np.ndarray,
    ref_psd: np.ndarray,
    pred_freq: np.ndarray,
    pred_psd: np.ndarray,
) -> np.ndarray:
  """Calculates the mean energy log ratio."""
  pred_interp = interp.interp1d(
      pred_freq, pred_psd, kind="cubic", bounds_error=False, fill_value=np.nan
  )
  pred_psd = pred_interp(ref_freq)
  return np.nanmean(np.abs(np.log(ref_psd) - np.log(pred_psd)))


def compute_temporal_psd(data_array: xr.DataArray, samples_per_year: int = 100):
  """Computes temporal power spectral density."""
  # `data_array` is a time series for a single pixel consisting of multiple
  # years and members. It is divided into years and the PSD is computed for each
  # year. The PSDs are then averaged over all years and all members.
  all_psd = []
  freq = None
  for year in np.unique(data_array.time.dt.year):
    # Limit the number of members because the computation is expensive to run
    # for all members and PSD typically converges very fast.
    members = np.random.choice(
        range(len(data_array.member)),
        size=min(samples_per_year, len(data_array.member)),
        replace=False,
    )
    for m in members:
      dr_ = data_array.isel(member=m, drop=True).sel(
          time=data_array.time.dt.year == year
      )
      time_series, time = dr_.values, dr_.time.values
      time = (time - time[0]) / np.timedelta64(1, "h")
      dt = time[1] - time[0]

      fft_res = np.fft.rfft(time_series)
      freq = np.fft.rfftfreq(len(time_series), d=dt)
      psd = np.abs(fft_res) ** 2 / len(time_series) * dt

      all_psd.append(psd)

  mean_psd = np.asarray(all_psd).mean(axis=0)
  return freq, mean_psd


def make_temporal_psd_plot(
    predictions: xr.DataArray,
    reference: xr.DataArray,
    comps: Mapping[str, Mapping[str, Any]],
    cities: Sequence[str],
    plot_name: str,
    method: str = "nearest",
    reference_freqs: Sequence[float] | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    save_dir: str | None = None,
) -> None:
  """Plot temporal power spectral density (PSD) comparison for selected cities.

  The input datasets are for a single variable with time, longitude and latitude
  dimensions (and possibly other dimensions such as member). We select a few
  cities (pixels) of interest and compute the PSD on the time series, treating
  each year and members as different realizations.

  Args:
    predictions: The sample dataset to plot.
    reference: The reference dataset to plot.
    comps: (Optional) Other results (previously computed with this function) to
      include in the same plot.
    cities: The cities to plot (coordinates are hardcoded in `eval_utils`).
    plot_name: The (save) name of the plot.
    method: The interpolation method to use (with xarray's `.sel` interface)
      when extracting city pixel distributions.
    reference_freqs: The frequencies to mark on the plot (as vertical dash
      lines).
    xlim: The xlim to set for the plot.
    ylim: The ylim to set for the plot.
    save_dir: The directory to save the plots to.
  """
  nc = len(cities)
  res = eval_utils.nested_defaultdict()

  fig, axes = plt.subplots(
      1, nc, figsize=(3 * nc, 4), tight_layout=True, dpi=200
  )
  for i, city in enumerate(cities):
    lon, lat = eval_utils.CITY_COORDS[city]
    # Process reference.
    ref_data = reference.sel(
        longitude=lon, latitude=lat, method=method, drop=True
    )
    logging.info("Computing PSD for reference, %s.", city)
    ref_freq, ref_psd = compute_temporal_psd(ref_data)
    axes[i].loglog(
        ref_freq, ref_psd, linewidth=1.5, linestyle="solid", label="reference"
    )
    res[f"{city}_ref"] = {"x": ref_freq, "y": ref_psd}

    # Process predictions.
    pred_data = predictions.sel(
        longitude=lon, latitude=lat, method=method, drop=True
    )
    logging.info("Computing PSD for predictions, %s.", city)
    pred_freq, pred_psd = compute_temporal_psd(pred_data)
    pred_err = _calculate_melr(ref_freq, ref_psd, pred_freq, pred_psd)
    axes[i].loglog(
        pred_freq,
        pred_psd,
        linewidth=1.5,
        linestyle="solid",
        label=f"predicted (err={pred_err:.3g})",
    )
    res[city] = {"x": pred_freq, "y": pred_psd}

    # Process previously computed results as baselines (optional).
    for comp, res_dict in comps.items():
      comp_freq, comp_psd = res_dict[city]["x"], res_dict[city]["y"]
      comp_err = _calculate_melr(ref_freq, ref_psd, comp_freq, comp_psd)
      axes[i].loglog(
          comp_freq,
          comp_psd,
          linewidth=1.5,
          linestyle="solid",
          label=f"{comp} (err={comp_err:.3g})",
      )

    axes[i].set_ylabel("")
    axes[i].set_title(city)
    axes[i].legend(fontsize=7, loc="lower left")
    if xlim is not None:
      axes[i].set_xlim(xlim)
    if ylim is not None:
      axes[i].set_ylim(ylim)
    for rfreq in reference_freqs:
      axes[i].axvline(x=rfreq, color="k", linestyle="dashed", linewidth=0.5)

  axes[0].set_ylabel("PSD")

  save_dir = epath.Path(save_dir) / "temporal_psd"
  eval_utils.save_fig(save_dir=save_dir, save_name=f"{plot_name}.png", fig=fig)

  res_save_name = f"{save_dir}/{plot_name}.hdf5"
  h5u.save_array_dict(save_path=res_save_name, data=res)


