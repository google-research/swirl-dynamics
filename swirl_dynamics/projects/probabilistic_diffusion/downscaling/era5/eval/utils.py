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

"""Eval utils."""

import collections
from collections.abc import Sequence
import logging
from typing import Any

import dask.array as da
from etils import epath
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils as pipeline_utils
import xarray as xr

filesys = epath.backend.tf_backend


CITY_COORDS = {
    # CONUS
    "San Francisco": (237.58, 37.77),
    "Los Angeles": (241.76, 34.05),
    "New York": (286, 40.71),
    "Phoenix": (247.93, 33.45),
    "Denver": (255, 39.74),
    "Seattle": (237.66, 47.61),
    "Miami": (360 - 80.21, 25.78),
    "Houston": (360 - 95.39, 29.79),
    "Chicago": (360 - 87.68, 41.84),
    "Minneapolis": (360 - 93.27, 44.96),
    "Atlanta": (360 - 84.42, 33.76),
    "Boston": (360 - 71.02, 42.33),
    # EU:
    "Berlin": (13.4, 52.5),
    "Istanbul": (28.9, 41.0),
    "Amsterdam": (4.9, 52.3),
    "Paris": (2.4, 48.9),
    "Bucharest": (26.10, 44.43),
    "Belgrade": (20.46, 44.82),
    "Rome": (12.48, 41.90),
    "Athens": (23.73, 37.98),
    "Warsaw": (21.0, 52.2),
    "Copenhagen": (12.55, 55.66),
    # North indian:
    "Mumbai": (72.87, 19.07),
    "Karachi": (67.01, 24.86),
    "Colombo": (79.86, 6.92),
    "Male": (73.51, 4.17),
    "Chittagong": (91.80, 22.35),
    "Yangon": (96.19, 16.86),
}


# **********************
# Dataset utils
# **********************



def get_reference_ds(
    ref_zarr_path: epath.PathLike,
    variables: pipeline_utils.DatasetVariables,
    sample_ds: xr.Dataset,
) -> xr.Dataset:
  """Reads reference dataset corresponding to a given sample dataset."""
  ref_ds = pipeline_utils.select_variables(ref_zarr_path, variables)
  ref_ds = ref_ds.sel(
      time=sample_ds["time"],
      longitude=sample_ds["longitude"],
      latitude=sample_ds["latitude"],
  )
  ref_ds = ref_ds.expand_dims(dim={"member": np.array([0])}, axis=1)
  return ref_ds


def select_time(
    ds: xr.Dataset, years: Sequence[int], months: Sequence[int]
) -> xr.Dataset:
  mask = ds.time.dt.year.isin(years) & ds.time.dt.month.isin(months)
  return ds.sel(time=mask, drop=True)


# **********************
# Derived variables
# **********************


def apply_ufunc(
    ds: xr.Dataset,
    ufunc: Any,
    input_vars: Sequence[str],
    output_var: str,
) -> xr.Dataset:
  inputs = [ds[var] for var in input_vars]
  new_var = xr.apply_ufunc(ufunc, *inputs, dask="parallelized")
  ds[output_var] = new_var
  return ds


def T_fahrenheit(T):  # pylint: disable=invalid-name
  """Converts temperature from Kelvin to Fahrenheit."""
  T = (T - 273.15) * 1.8 + 32
  return T


def relative_humidity(T, q, msl, zs):  # pylint: disable=invalid-name
  """Computes relative humidity from temperature, specific humidity, and pressure."""
  # Barometric formula: https://en.wikipedia.org/wiki/Barometric_formula
  p = msl * np.power(
      1 - 0.0065 * zs / 288.15, (9.81 * 0.0289644) / (8.31447 * 0.0065)
  )  # unit: Pa
  # From ideal gas law and Dalton's law
  e = q * p / (q * 0.378 + 0.622)
  # Magnus formula, unit: Pa
  es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
  return (e / es) * 100


def heat_index(TF, RH):  # pylint: disable=invalid-name
  """Computes heat index (°F) using temperature (°F) and relative humidity."""
  # https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
  c = [
      -42.379,
      2.04901523,
      10.14333127,
      -0.22475541,
      -6.837e-3,
      -5.481717e-2,
      1.22874e-3,
      8.5282e-4,
      -1.99e-6,
  ]
  RH = np.clip(RH, 0, 100)
  hi1 = 0.5 * (TF + 61.0 + (TF - 68.0) * 1.2 + (RH * 0.094))
  hi2 = (
      c[0]
      + c[1] * TF
      + c[2] * RH
      + c[3] * TF * RH
      + c[4] * TF * TF
      + c[5] * RH * RH
      + c[6] * TF * TF * RH
      + c[7] * TF * RH * RH
      + c[8] * TF * TF * RH * RH
  )
  adj_a = ((13 - RH) / 4) * np.sqrt((17 - np.abs(TF - 95)) / 17)
  adj_a = np.nan_to_num(adj_a, nan=0.0)
  hi2 -= np.less(RH, 13) * np.greater(TF, 80) * np.less(TF, 112) * adj_a
  adj_b = ((RH - 85) / 10) * ((87 - TF) / 5)
  hi2 += np.greater(RH, 85) * np.greater(TF, 80) * np.less(TF, 87) * adj_b
  hi = (hi1 > 80) * hi1 + (hi1 <= 80) * hi2
  return hi


def add_zs(surf_geopotential_ds: epath.PathLike):
  """Adds surface elevation to data."""

  ref_ds = xr.open_zarr(surf_geopotential_ds)
  zs = ref_ds["geopotential_at_surface"].load()

  def _add_zs(ds: xr.Dataset) -> xr.Dataset:
    ds["ZS"] = (
        zs.sel(longitude=ds["longitude"], latitude=ds["latitude"]).transpose(
            "longitude", "latitude"
        )
        / 9.8
    )
    return ds

  return _add_zs


# **********************
# Aggregation functions
# **********************


def diurnal_range(x):
  return x.max(dim="time") - x.min(dim="time")


def daily_max(x):
  return x.max(dim="time")


def daily_mean(x):
  return x.mean(dim="time")


def daily_std(x):
  return x.std(dim="time")


def get_agg_function(fun: str):
  match fun:
    case "daily_range":
      return diurnal_range
    case "daily_max":
      return daily_max
    case "daily_mean":
      return daily_mean
    case "daily_std":
      return daily_std
    case _:
      raise ValueError(f"Unsupported aggregation function: {fun}")


# **********************
# Plotting utils
# **********************


def transparent_cmap(
    color: str = "darkred", n_colors: int = 256
) -> mpl.colors.LinearSegmentedColormap:
  palette = sns.light_palette(color, as_cmap=True)
  colors = palette(np.arange(n_colors))
  colors[:, -1] = np.linspace(0, 1, n_colors)
  return mpl.colors.LinearSegmentedColormap.from_list(
      f"transparent_{color}", colors
  )


def save_fig(
    save_dir: epath.PathLike,
    save_name: str,
    fig: plt.Figure,
    close_fig: bool = True,
    **kwargs,
) -> None:
  """Saves a figure to a given directory."""
  save_dir = epath.Path(save_dir)
  if not filesys.exists(save_dir):
    filesys.makedirs(save_dir)

  save_path = save_dir / save_name
  with filesys.open(save_path, "wb") as f:
    fig.savefig(f, **kwargs)

  logging.info("Saved figure to %s.", save_path)

  if close_fig:
    plt.close(fig)


def nested_defaultdict() -> collections.defaultdict:
  return collections.defaultdict(nested_defaultdict)
