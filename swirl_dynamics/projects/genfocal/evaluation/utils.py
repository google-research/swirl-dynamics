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

"""Utility functions for evaluation."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from swirl_dynamics.projects.genfocal.super_resolution import data
import xarray as xr

# **********************
# Physical constants
# **********************

# Duarte et al (2014), https://doi.org/10.1175/MWR-D-13-00368.1, page 4275.
_R_D = 287.058  # J / kg / K
_R_V = 461.85  # J / kg / K
_EPS = _R_D / _R_V  # Ratio of dry air to water vapor gas constants.
_C_VV = 1424.0  # J / kg / K
_C_VL = 4186.0  # J / kg / K
_L_V0 = 2.5e6  # J / kg
_T_REF = 273.15  # Zero degree Celsius in Kelvin, also Triple point temp.
_P_TRIPLE = 611.0  # Triple-point pressure of water, in Pa

_ISA_LAPSE_RATE = 0.0065  # [K/m]
_G = 9.81  # Gravity in [m/s^2]
_M_AIR = 0.0289644  # Air molar mass in [kg/mol]
_R = 8.31447  # Gas constant in [J / kg / K]
_ISA_REF_T = 288.15  # [K]

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
  T = (T - _T_REF) * 1.8 + 32
  return T


def sat_vapor_pressure(t):
  """Computes the saturation vapor pressure with respect to liquid water.

  The expression used here follows Duarte et al (2014),
  https://doi.org/10.1175/MWR-D-13-00368.1. All constants are defined in the
  same paper, page 4275.

  Args:
    t: Temperature in [K].

  Returns:
    The saturation vapor pressure in [Pa].
  """
  cpv = _C_VV + _R_V
  e0v = _L_V0 - _R_V * _T_REF
  alpha = (cpv - _C_VL) / _R_V
  beta = (e0v - (_C_VV - _C_VL) * _T_REF) / _R_V
  return _P_TRIPLE * (t / _T_REF) ** alpha * np.exp(beta * (1 / _T_REF - 1 / t))


def pressure_from_msl_and_height(msl, h):
  """Computes pressure from mean sea level pressure and height.

  This diagnosis assumes that pressure changes with height according to the
  barometric formula and the International Standard Atmosphere,
  https://en.wikipedia.org/wiki/Barometric_formula.

  Args:
    msl: Mean sea level pressure, in [Pa].
    h: Height at which pressure is evaluated with respect to sea level, in [m].

  Returns:
    The pressure at height h, in [Pa].
  """
  return msl * np.power(
      1 - _ISA_LAPSE_RATE * h / _ISA_REF_T,
      (_G * _M_AIR) / (_R * _ISA_LAPSE_RATE),
  )


def relative_humidity(t, q, msl, zs):
  """Computes the relative humidity with respect to liquid water.

  Args:
    t: Temperature in [K].
    q: Specific humidity in [kg/kg].
    msl: Mean sea level pressure in [Pa].
    zs: The height of the (t,q) measurements with respect to sea level, in [m].

  Returns:
    The relative humidity, as a percentage, at the same height `zs` as `q` and
    `t`.
  """
  p = pressure_from_msl_and_height(msl, zs)
  e = p * q / (_EPS + (1 - _EPS) * q)
  return 100 * e / sat_vapor_pressure(t)


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
  adj_a = ((13.0 - RH) / 4.0) * np.sqrt((17.0 - np.abs(TF - 95.0)) / 17.0)
  adj_a = np.nan_to_num(adj_a, nan=0.0)
  hi2 -= np.less(RH, 13.0) * np.greater(TF, 80.0) * np.less(TF, 112.0) * adj_a
  adj_b = ((RH - 85.0) / 10.0) * ((87.0 - TF) / 5.0)
  hi2 += np.greater(RH, 85.0) * np.greater(TF, 80.0) * np.less(TF, 87.0) * adj_b
  hi = (hi2 > 80.0) * hi2 + (hi2 <= 80.0) * hi1
  hi = (hi - 32.0) * 5.0 / 9.0 + 273.15  # Convert to Kelvin
  return hi


def add_zs(surf_geopotential_data_path: str):
  """Adds surface elevation to data."""

  ref_ds = xr.open_zarr(surf_geopotential_data_path)
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
# Dataset utils
# **********************


def get_reference_ds(
    ref_zarr_path: str, variables: data.DatasetVariables, sample_ds: xr.Dataset
) -> xr.Dataset:
  """Reads reference dataset corresponding to a given sample dataset."""
  ref_ds = data.select_variables(ref_zarr_path, variables)
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
