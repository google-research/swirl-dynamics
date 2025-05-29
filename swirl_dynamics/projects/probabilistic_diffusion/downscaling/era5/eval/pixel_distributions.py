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

"""Eval downscaling results for pixel distributions."""

from collections.abc import Mapping, Sequence
import functools
from typing import Any, Literal

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

sns.set_palette("colorblind")


def get_pixel_time_series(
    ds: xr.DataArray,
    city: str,
    daily_aggregate_fun: str | None = None,
    method: str = "nearest",
) -> np.ndarray:
  """Gets the time series of a pixel."""
  daily_aggregate_fun = daily_aggregate_fun or ""
  coord = eval_utils.CITY_COORDS[city]
  ds = ds.sel(longitude=coord[0], latitude=coord[1], method=method, drop=True)
  if daily_aggregate_fun:
    times = ds["time"].values
    ds = ds.resample(time="D").map(
        eval_utils.get_agg_function(daily_aggregate_fun)
    )
    ds = ds.where(ds["time"].isin(times), drop=True)
  return ds.to_numpy().ravel()


def plot_density_and_get_data(
    data: np.ndarray, ax: plt.Axes, cumulative: bool
) -> tuple[np.ndarray, np.ndarray]:
  sns.kdeplot(
      data, ax=ax, linewidth=1.5, linestyle="solid", cumulative=cumulative
  )
  line = ax.get_lines()[-1]
  return line.get_xdata(), line.get_ydata()


def compute_err(
    ref_x: np.ndarray,
    ref_px: np.ndarray,
    pred_x: np.ndarray,
    pred_px: np.ndarray,
    cumulative: bool = False,  # Whether the above points represent CDF or PDF.
    num_eval_pts: int = 1000,
) -> np.ndarray:
  """Calculates the mean absolute error of CDFs."""
  eval_pts = np.linspace(np.min(ref_x), np.max(ref_x), num_eval_pts)
  ref_interp = interp.interp1d(
      ref_x, ref_px, kind="linear", fill_value=(0, 1), bounds_error=False
  )(eval_pts)
  pred_interp = interp.interp1d(
      pred_x, pred_px, kind="linear", fill_value=(0, 1), bounds_error=False
  )(eval_pts)
  if not cumulative:
    ref_interp = np.cumsum(ref_interp) / np.sum(ref_interp)
    pred_interp = np.cumsum(pred_interp) / np.sum(pred_interp)
  return np.nanmean(np.abs(ref_interp - pred_interp))


def make_pixel_density_plot(
    predictions: xr.DataArray,
    reference: xr.DataArray,
    comps: Mapping[str, Mapping[str, Any]],
    cities: Sequence[str],
    plot_name: str,
    daily_aggregate_fun: str | None = None,
    method: str = "nearest",
    cumulative: bool = False,
    log_yscale: bool = False,
    save_dir: str | None = None,
) -> None:
  """Makes plots comparing pixelwise distributions.

  The input datasets are for a single variable with (minimally) time, longitude
  and latitude dimensions. We select a few cities (pixels) of interest and
  compare the distributions from the prediction and reference, treating all
  other dimensions except lon/lat as different realizations.

  Args:
    predictions: The prediction dataset.
    reference: The reference dataset.
    comps: (Optional) other results (previously computed with this function) to
      include in the same plot.
    cities: The cities to plot (coordinates are hardcoded in `eval_utils`).
    plot_name: The save name of the plot.
    daily_aggregate_fun: Optional function that aggregate daily statistics (see
      ones supported in `eval_utils.get_agg_function`) for plotting.
    method: The interpolation method to use (with xarray's `.sel` interface)
      when extracting city pixel distributions.
    cumulative: Whether to plot the cumulative distribution or probability
      density function.
    log_yscale: Whether to use log scale for the y-axis.
    save_dir: The directory to save the plots to.
  """
  nc = len(cities)
  plot_type = "cdf" if cumulative else "pdf"
  res = eval_utils.nested_defaultdict()
  fig, axes = plt.subplots(
      1, nc, figsize=(3.5 * nc, 3), tight_layout=True, dpi=200
  )

  for i, city in enumerate(cities):
    # Process reference.
    ref_data = get_pixel_time_series(
        reference, city, daily_aggregate_fun, method
    )
    ref_x, ref_px = plot_density_and_get_data(
        ref_data, axes[i], cumulative=cumulative
    )
    axes[i].get_lines()[-1].set_label("reference")
    res[f"{city}_ref"] = {"x": ref_x, "y": ref_px}

    # Process predictions.
    pred_data = get_pixel_time_series(
        predictions, city, daily_aggregate_fun, method
    )
    pred_x, pred_px = plot_density_and_get_data(
        pred_data, axes[i], cumulative=cumulative
    )
    pred_err = compute_err(
        ref_x, ref_px, pred_x, pred_px, cumulative=cumulative
    )
    axes[i].get_lines()[-1].set_label(f"predicted (err={pred_err:.3g})")
    res[city] = {"x": pred_x, "y": pred_px}

    # Process previously computed results as baselines (optional).
    for comp, res_dict in comps.items():
      comp_x, comp_px = res_dict[city]["x"], res_dict[city]["y"]
      axes[i].plot(comp_x, comp_px, linewidth=1.5, linestyle="solid")
      comp_err = compute_err(
          ref_x, ref_px, comp_x, comp_px, cumulative=cumulative
      )
      axes[i].get_lines()[-1].set_label(f"{comp} (err={comp_err:.3g})")

    axes[i].set_ylabel("")
    if plot_type == "cdf":
      axes[i].set_ylim([0, 1])
    if log_yscale:
      assert not cumulative, "Log scale is not supported for CDF."
      axes[i].set_yscale("log")
      ylim_low = 10 ** np.ceil(np.log10(np.min(ref_px)))
      ylim_hi = 10 ** np.ceil(np.log10(np.max(ref_px)) + 1)
      axes[i].set_ylim([ylim_low, ylim_hi])
    xlim_low = np.min(ref_x) - (np.max(ref_x) - np.min(ref_x)) * 0.2
    xlim_hi = np.max(ref_x) + (np.max(ref_x) - np.min(ref_x)) * 0.2
    axes[i].set_xlim([xlim_low, xlim_hi])
    axes[i].set_title(city)
    axes[i].legend(fontsize=7, loc="upper right")

  ylabel = "Cumulative Probability" if cumulative else "Probability Density"
  axes[0].set_ylabel(ylabel)

  save_dir = epath.Path(save_dir) / plot_type
  eval_utils.save_fig(save_dir=save_dir, save_name=f"{plot_name}.png", fig=fig)

  res_save_name = f"{save_dir}/{plot_name}.hdf5"
  h5u.save_array_dict(save_path=res_save_name, data=res)


def make_pixel_qq_plot(
    predictions: xr.DataArray,
    reference: xr.DataArray,
    comps: Mapping[str, Mapping[str, Any]],
    cities: Sequence[str],
    plot_name: str,
    daily_aggregate_fun: str | None = None,
    method: str = "nearest",
    save_dir: str | None = None,
) -> None:
  """Make quantile-quantile plot for selected cities and compute MAE.

  The input datasets are for a single variable with time, longitude and latitude
  dimensions (and possibly other dimensions such as member). We select a few
  cities (pixels) of interest and compare the distributions from the prediction
  and reference, treating all other dimensions (lon/lat is fixed) as different
  realizations.

  Args:
    predictions: The sample dataset to plot.
    reference: The reference dataset to plot.
    comps: Other (baseline) datasets to include in the same plot.
    cities: The cities to plot (coordinates are hardcoded in `eval_utils`).
    plot_name: The (save) name of the plot.
    daily_aggregate_fun: Optional function that aggregate daily statistics (see
      ones supported in `eval_utils.get_agg_function`) for plotting.
    method: The interpolation method to use (with xarray's `.sel` interface).
    save_dir: The directory to save the plot to.
  """
  nc = len(cities)
  percentiles = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50]
  percentiles += [60, 70, 75, 80, 85, 90, 93, 95, 96, 97, 98, 99]
  res = eval_utils.nested_defaultdict()
  fig, axes = plt.subplots(
      1, nc, figsize=(3 * nc, 3), tight_layout=True, dpi=200
  )
  for i, city in enumerate(cities):
    ref_data = get_pixel_time_series(
        reference, city, daily_aggregate_fun, method
    )
    ref_quantiles = np.percentile(ref_data, percentiles)

    pred_data = get_pixel_time_series(
        predictions, city, daily_aggregate_fun, method
    )
    pred_quantiles = np.percentile(pred_data, percentiles)
    res[city] = pred_quantiles

    sns.scatterplot(
        x=ref_quantiles,
        y=pred_quantiles,
        ax=axes[i],
        s=30,
        edgecolor="black",
        marker="s",
        linewidth=1,
        label="sample",
    )

    for comp, res_dict in comps.items():
      comp_quantiles = res_dict[city]
      sns.scatterplot(
          x=ref_quantiles,
          y=comp_quantiles,
          ax=axes[i],
          s=25,
          edgecolor="black",
          marker="^",
          linewidth=1,
          label=comp,
      )

    m = ref_quantiles[12]  # 50th quantile
    min_val = np.minimum(np.min(pred_quantiles), np.min(ref_quantiles))
    max_val = np.maximum(np.max(pred_quantiles), np.max(ref_quantiles))
    offset = max_val - min_val
    min_val -= 0.20 * offset
    max_val += 0.20 * offset

    axes[i].axline((m, m), slope=1, ls=":", color="black", lw=0.75)
    axes[i].set_xlabel("reference")
    axes[i].set_aspect(aspect="equal", adjustable="datalim")
    axes[i].set_title(city)
    axes[i].set_xlim([min_val, max_val])
    axes[i].set_ylim([min_val, max_val])

  axes[0].set_ylabel("sample")

  save_dir = epath.Path(save_dir) / "qq"
  eval_utils.save_fig(save_dir=save_dir, save_name=f"{plot_name}.png", fig=fig)

  res_save_name = f"{save_dir}/{plot_name}.hdf5"
  h5u.save_array_dict(save_path=res_save_name, data=res)


def add_derived_variables(ds: xr.Dataset) -> xr.Dataset:
  """Adds temperature (F), relative humidity and heat index to dataset."""
  ds = eval_utils.apply_ufunc(
      ds, eval_utils.T_fahrenheit, input_vars=["2mT"], output_var="2mTF"
  )
  ds = eval_utils.apply_ufunc(
      ds,
      eval_utils.relative_humidity,
      input_vars=["2mT", "Q1000", "MSL", "ZS"],
      output_var="RH",
  )
  ds = eval_utils.apply_ufunc(
      ds, eval_utils.heat_index, input_vars=["2mTF", "RH"], output_var="HI"
  )
  return ds


