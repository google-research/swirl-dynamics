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

"""Eval downscaling results for spatial correlation of a pixel with its neighbors."""

from collections.abc import Mapping, Sequence
from typing import Any

from absl import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from etils import epath
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from swirl_dynamics.data import hdf5_utils as h5u
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.eval import utils as eval_utils
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils as pipeline_utils
import xarray as xr

filesys = epath.backend.tf_backend

sns.set_style("white")


def get_pixel_time_series_and_neighbors(
    ds: xr.DataArray,
    city: str,
    daily_aggregate_fun: str,
    srange: float,
    method: str = "nearest",
) -> tuple[xr.DataArray, xr.DataArray]:
  """Extracts the time series for a city and its neighbors."""
  daily_aggregate_fun = daily_aggregate_fun or ""
  lon, lat = eval_utils.CITY_COORDS[city]
  if daily_aggregate_fun:
    times = ds["time"].values
    ds = ds.resample(time="D").map(
        eval_utils.get_agg_function(daily_aggregate_fun)
    )
    ds = ds.where(ds["time"].isin(times), drop=True)

  center = ds.sel(longitude=lon, latitude=lat, method=method)
  neighbors = ds.sel(  # Select neighboring points in range.
      longitude=slice(lon - srange, lon + srange),
      latitude=slice(lat - srange, lat + srange),
  )
  return center, neighbors


def compute_err(corr1: np.ndarray, corr2: np.ndarray) -> np.ndarray:
  """Calculates the MAE of correlation coefficients."""
  return np.nanmean(np.abs(corr1 - corr2))


def spatial_corr(city: xr.DataArray, neighbors: xr.DataArray) -> xr.DataArray:
  """Computes spatial correlation of a city with its surroundings."""
  x = city.to_numpy()  # ~ (time, member)
  y = neighbors.to_numpy()  # ~ (time, member, lon, lat)
  x = x.flatten()
  y = y.reshape(-1, *y.shape[-2:])
  x_centered = x - np.nanmean(x)
  y_centered = y - np.nanmean(y, axis=0)
  cov = np.nanmean(x_centered[:, None, None] * y_centered, axis=0)
  corr = cov / (np.std(x) * np.std(y, axis=0))
  return xr.DataArray(
      corr,
      dims=["longitude", "latitude"],
      coords={"longitude": neighbors.longitude, "latitude": neighbors.latitude},
  )


def make_spatial_corr_plot(
    corr: xr.DataArray,
    city: str,
    fig: plt.Figure,
    ax: plt.Axes,
    err: np.ndarray | None = None,
) -> None:
  """Makes a spatial correlation plot."""
  lons = corr.longitude.to_numpy()
  lats = corr.latitude.to_numpy()
  xx, yy = np.meshgrid(lons, lats, indexing="ij")

  # pytype: disable=attribute-error
  ax.add_feature(cfeature.OCEAN, color="white")
  ax.add_feature(cfeature.LAND, color="lightgray")
  ax.add_feature(cfeature.LAKES, alpha=0.75)
  ax.coastlines(resolution="50m", linewidth=0.4, color="black")
  # pytype: enable=attribute-error

  sc = ax.scatter(
      xx.flatten(),
      yy.flatten(),
      c=corr.data.flatten(),
      s=8,
      transform=ccrs.PlateCarree(),
      cmap=sns.color_palette("Spectral_r", as_cmap=True),
      vmin=0.0,
      vmax=1.0,
  )
  ax.plot(
      eval_utils.CITY_COORDS[city][0],
      eval_utils.CITY_COORDS[city][1],
      marker="*",
      mfc="white",
      mec="k",
      markersize=10,
      transform=ccrs.PlateCarree(),
  )
  fig.colorbar(sc, ax=ax, format="%.2f")
  if err is not None:
    ax.annotate(
        f"MAE: {err:.3f}",
        xy=(0.95, 0.04),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.6),
        fontsize=7,
    )


def make_spatial_corr_plots(
    predictions: xr.DataArray,
    reference: xr.DataArray,
    comps: Mapping[str, Mapping[str, Any]],
    srange: float,
    cities: Sequence[str],
    plot_name: str,
    daily_aggregate_fun: str | None = None,
    method: str = "nearest",
    save_dir: str | None = None,
) -> None:
  """Plot spatial correlation comparison for selected cities and compute MAE.

  The input datasets are for a single variable with time, longitude and latitude
  dimensions (and possibly other dimensions such as member). We select a few
  cities (pixels) of interest and compute the correlation coefficient between
  the pixel and its neighbors within a given spatial range, treating all other
  dimensions (lon/lat is fixed) as different realizations.

  Args:
    predictions: The sample dataset to plot.
    reference: The reference dataset to plot.
    comps: (Optional) Other results (previously computed with this function) to
      include in the same plot.
    srange: The spatial range (in degrees lon/lat) within which to compute the
      correlation coefficients.
    cities: The cities to plot (coordinates are hardcoded in `eval_utils`).
    plot_name: The (save) name of the plot.
    daily_aggregate_fun: (Optional) Function that aggregate daily statistics
      (see ones supported in `eval_utils.get_agg_function`) for plotting.
    method: The interpolation method to use (with xarray's `.sel` interface)
      when extracting city pixel distributions.
    save_dir: The directory to save the plots to.
  """
  nc = len(cities)
  nr = 2 + len(comps)
  res = eval_utils.nested_defaultdict()

  fig, axes = plt.subplots(
      nr,
      nc,
      figsize=(3.5 * nc, 3 * nr),
      subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
      tight_layout=True,
      dpi=200,
  )

  for i, city in enumerate(cities):
    # Process reference.
    logging.info("Computing spatial corr for reference, %s.", city)
    ref_data, ref_neighbors = get_pixel_time_series_and_neighbors(
        reference, city, daily_aggregate_fun, srange, method
    )
    ref_corr = spatial_corr(ref_data, ref_neighbors)
    make_spatial_corr_plot(ref_corr, city, fig, axes[0, i])
    axes[0, i].set_title(f"{city}: reference")
    res[f"{city}_ref"] = {
        "corr": ref_corr.to_numpy(),
        "lon": ref_corr.longitude.to_numpy(),
        "lat": ref_corr.latitude.to_numpy(),
    }

    # Process predictions.
    logging.info("Computing spatial corr for predictions, %s.", city)
    pred_data, pred_neighbors = get_pixel_time_series_and_neighbors(
        predictions, city, daily_aggregate_fun, srange, method
    )
    pred_corr = spatial_corr(pred_data, pred_neighbors)
    pred_err = compute_err(ref_corr.to_numpy(), pred_corr.to_numpy())
    make_spatial_corr_plot(pred_corr, city, fig, axes[1, i], pred_err)
    axes[1, i].set_title(f"{city}: predicted")
    res[city] = {
        "corr": pred_corr.to_numpy(),
        "lon": pred_corr.longitude.to_numpy(),
        "lat": pred_corr.latitude.to_numpy(),
    }

    # Process previously computed results as baselines (optional).
    for j, (comp, res_dict) in enumerate(comps.items()):
      comp_corr = xr.DataArray(
          res_dict[city]["corr"],
          dims=["longitude", "latitude"],
          coords={
              "longitude": res_dict[city]["lon"],
              "latitude": res_dict[city]["lat"],
          },
      )
      comp_err = compute_err(ref_corr.to_numpy(), comp_corr.to_numpy())
      make_spatial_corr_plot(comp_corr, city, fig, axes[j + 2, i], comp_err)
      axes[j + 2, i].set_title(f"{city}: {comp}")

  save_dir = epath.Path(save_dir) / "spatial_corr"
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


