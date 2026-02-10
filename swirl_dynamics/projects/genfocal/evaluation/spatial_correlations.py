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

r"""Script to evaluate spatial correlations of a pixel with its neighbors.

Example usage:

```
SAMPLE_PATH=<sample_zarr_path>
REFERENCE_PATH=<reference_zarr_path>
OUTPUT_DIR=<output_dir>
ZS_PATH=<surface_geopotential_zarr_path>

python swirl_dynamics/projects/genfocal/evaluation/spatial_correlations.py -- \
    --sample_path=${SAMPLE_PATH} \
    --reference_path=${REFERENCE_PATH} \
    --output_dir=${OUTPUT_DIR} \
    --zs_path=${ZS_PATH} \
    --year_start=2010 \
    --year_end=2019 \
    --months=6,7,8 \
    --hour_of_day=18
```
"""

import collections
from collections.abc import Mapping

from absl import app
from absl import flags
from absl import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from etils import epath
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from swirl_dynamics.data import hdf5_utils as h5u
from swirl_dynamics.projects.genfocal.evaluation import utils
from swirl_dynamics.projects.genfocal.super_resolution import data
import xarray as xr

filesys = epath.backend.tf_backend

sns.set_style("white")

# Command line arguments
SAMPLE_PATH = flags.DEFINE_string(
    "sample_path", None, help="Sample dataset Zarr path."
)
REFERENCE_PATH = flags.DEFINE_string(
    "reference_path", None, help="Reference dataset Zarr path."
)
OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, help="Output directory where results will be saved."
)
ZS_PATH = flags.DEFINE_string(
    "zs_path", None, help="Path to the surface geopotential data."
)
SRANGE = flags.DEFINE_float(
    "srange",
    4.0,
    help=(
        "Spatial range (in degrees lon/lat) within which to compute the"
        " correlation coefficients."
    ),
)
YEAR_START = flags.DEFINE_integer(
    "year_start", 2001, help="Starting year for evaluation."
)
YEAR_END = flags.DEFINE_integer(
    "year_end", 2010, help="Ending year for evaluation."
)
MONTHS = flags.DEFINE_list(
    "months", ["6", "7", "8"], help="Months for evaluation."
)
HOUR_OF_DAY = flags.DEFINE_integer(
    "hour_of_day", None, help="Hour of day in UTC time."
)

Variable = data.DatasetVariable
SAMPLE_VARIABLES = [
    Variable("2m_temperature", None, "T2m"),
    Variable("10m_magnitude_of_wind", None, "W10m"),
    Variable("specific_humidity", {"level": [1000]}, "Q1000"),
    Variable("mean_sea_level_pressure", None, "MSL"),
]

CITIES = {
    "San Francisco": (237.58, 37.77),
    "New York": (286, 40.71),
    "Denver": (255, 39.74),
    "Miami": (360 - 80.21, 25.78),
    "Houston": (360 - 95.39, 29.79),
    "Chicago": (360 - 87.68, 41.84),
}


def nested_defaultdict() -> collections.defaultdict:
  return collections.defaultdict(nested_defaultdict)


def get_pixel_time_series_and_neighbors(
    ds: xr.DataArray,
    center_coords: tuple[float, float],  # (lon, lat)
    srange: float,
    method: str = "nearest",
) -> tuple[xr.DataArray, xr.DataArray]:
  """Extracts the time series for a city and its neighbors."""
  lon, lat = center_coords
  center = ds.sel(longitude=lon, latitude=lat, method=method)
  neighbors = ds.sel(  # Select neighboring points in range.
      longitude=slice(lon - srange, lon + srange),
      latitude=slice(lat - srange, lat + srange),
  )
  return center, neighbors


def compute_err(corr1: np.ndarray, corr2: np.ndarray) -> np.ndarray:
  """Calculates the MAE of correlation coefficients."""
  return np.mean(np.abs(corr1 - corr2))


def spatial_corr(city: xr.DataArray, neighbors: xr.DataArray) -> xr.DataArray:
  """Computes spatial correlation of a city with its surroundings."""
  x = city.to_numpy()  # ~ (time, member)
  y = neighbors.to_numpy()  # ~ (time, member, lon, lat)
  x = x.flatten()
  y = y.reshape(-1, *y.shape[-2:])
  x_centered = x - np.mean(x)
  y_centered = y - np.mean(y, axis=0)
  cov = np.mean(x_centered[:, None, None] * y_centered, axis=0)
  corr = cov / (np.std(x) * np.std(y, axis=0))
  return xr.DataArray(
      corr,
      dims=["longitude", "latitude"],
      coords={"longitude": neighbors.longitude, "latitude": neighbors.latitude},
  )


def make_spatial_corr_plot(
    corr: xr.DataArray,
    center_coords: tuple[float, float],  # (lon, lat)
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
      center_coords[0],
      center_coords[1],
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
    srange: float,
    cities: Mapping[str, tuple[float, float]],
    plot_name: str,
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
    srange: The spatial range (in degrees lon/lat) within which to compute the
      correlation coefficients.
    cities: The cities to plot in the format {city: (lon, lat)}.
    plot_name: The (save) name of the plot.
    method: The interpolation method to use (with xarray's `.sel` interface)
      when extracting city pixel distributions.
    save_dir: The directory to save the plots to.
  """
  nc, nr = len(cities), 2
  res = nested_defaultdict()

  fig, axes = plt.subplots(
      nr,
      nc,
      figsize=(3.5 * nc, 3 * nr),
      subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
      tight_layout=True,
      dpi=200,
  )

  for i, (city, coords) in enumerate(cities.items()):
    # Process reference.
    logging.info("Computing spatial corr for reference, %s.", city)
    ref_data, ref_neighbors = get_pixel_time_series_and_neighbors(
        reference, coords, srange, method
    )
    ref_corr = spatial_corr(ref_data, ref_neighbors)
    make_spatial_corr_plot(ref_corr, coords, fig, axes[0, i])
    axes[0, i].set_title(f"{city}: reference")
    res[f"{city}_ref"] = {
        "corr": ref_corr.to_numpy(),
        "lon": ref_corr.longitude.to_numpy(),
        "lat": ref_corr.latitude.to_numpy(),
    }

    # Process predictions.
    logging.info("Computing spatial corr for predictions, %s.", city)
    pred_data, pred_neighbors = get_pixel_time_series_and_neighbors(
        predictions, coords, srange, method
    )
    pred_corr = spatial_corr(pred_data, pred_neighbors)
    pred_err = compute_err(ref_corr.to_numpy(), pred_corr.to_numpy())
    make_spatial_corr_plot(pred_corr, coords, fig, axes[1, i], pred_err)
    axes[1, i].set_title(f"{city}: predicted")
    res[city] = {
        "corr": pred_corr.to_numpy(),
        "lon": pred_corr.longitude.to_numpy(),
        "lat": pred_corr.latitude.to_numpy(),
    }

  save_dir = epath.Path(save_dir) / "spatial_corr"
  fig.savefig(save_dir / f"{plot_name}.png")

  res_save_name = f"{save_dir}/{plot_name}.hdf5"
  h5u.save_array_dict(save_path=res_save_name, data=res)


def add_derived_variables(ds: xr.Dataset, zs_data_path: str) -> xr.Dataset:
  """Adds temperature (F), relative humidity and heat index to dataset.

  Args:
    ds: The dataset to add derived variables to.
    zs_data_path: The path to the surface geopotential data. A Zarr dataset with
      a `geopotential_at_surface` variable and safe to query for the longitude
      and latitude coordinates of `ds`.

  Returns:
    The dataset with derived variables added.
  """
  ds = utils.add_zs(zs_data_path)(ds)

  ds = utils.apply_ufunc(
      ds, utils.T_fahrenheit, input_vars=["T2m"], output_var="T2m_F"
  )
  ds = utils.apply_ufunc(
      ds,
      utils.relative_humidity,
      input_vars=["T2m", "Q1000", "MSL", "ZS"],
      output_var="RH",
  )
  ds = utils.apply_ufunc(
      ds, utils.heat_index, input_vars=["T2m_F", "RH"], output_var="HI"
  )
  return ds


def main(argv: list[str]) -> None:
  del argv

  sample_ds = xr.open_zarr(SAMPLE_PATH.value)
  years = list(range(YEAR_START.value, YEAR_END.value + 1))
  months = [int(m) for m in MONTHS.value]

  sample_ds = add_derived_variables(sample_ds, ZS_PATH.value)
  sample_ds = utils.select_time(sample_ds, years, months)

  ref_ds = utils.get_reference_ds(
      REFERENCE_PATH.value, SAMPLE_VARIABLES, sample_ds
  )
  ref_ds = add_derived_variables(ref_ds, ZS_PATH.value)
  ref_ds = utils.select_time(ref_ds, years, months)

  if HOUR_OF_DAY.value:
    sample_hr_mask = sample_ds.time.dt.hour.isin([HOUR_OF_DAY.value])
    sample_ds = sample_ds.sel(time=sample_hr_mask, drop=True)
    ref_hr_mask = ref_ds.time.dt.hour.isin([HOUR_OF_DAY.value])
    ref_ds = ref_ds.sel(time=ref_hr_mask, drop=True)

  plot_variables = [v.rename for v in SAMPLE_VARIABLES]
  plot_variables += ["RH", "HI"]

  for plot_var in plot_variables:
    logging.info("Generating plot for variable: %s.", plot_var)
    plot_name = f"{plot_var}"
    make_spatial_corr_plots(
        predictions=sample_ds[plot_var],
        reference=ref_ds[plot_var],
        plot_name=plot_name,
        cities=CITIES,
        srange=SRANGE.value,
        save_dir=OUTPUT_DIR.value,
    )


if __name__ == "__main__":
  app.run(main)
