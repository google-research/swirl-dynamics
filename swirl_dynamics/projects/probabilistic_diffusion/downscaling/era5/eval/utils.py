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
from scipy import optimize
import seaborn as sns
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils as pipeline_utils
import xarray as xr

filesys = epath.backend.tf_backend


# Duarte et al (2014), https://doi.org/10.1175/MWR-D-13-00368.1, page 4275.
_R_D = 287.058  # J / kg / K
_R_V = 461.85  # J / kg / K
_EPS = _R_D / _R_V  # Ratio of dry air to water vapor gas constants.
_C_VV = 1424.0  # J / kg / K
_C_VL = 4186.0  # J / kg / K
_L_V0 = 2.5e6  # J / kg
_T_REF = 273.15  # Zero degree Celsius in Kelvin, also Triple point temp.
_P_TRIPLE = 611.0  # Triple-point pressure of water, in Pa
_RAINWATER_DENSITY = 1000.0  # Rainwater density in [kg/m3]

_ISA_LAPSE_RATE = 0.0065  # [K/m]
_G = 9.81  # Gravity in [m/s^2]
_M_AIR = 0.0289644  # Air molar mass in [kg/mol]
_R = 8.31447  # Gas constant in [J / kg / K]
_ISA_REF_T = 288.15  # [K]

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
  return (T - _T_REF) * 1.8 + 32


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


def specific_humidity_from_dewpoint(td, msl, zs):
  """Computes specific humidity from dewpoint, sea level pressure and height.

  This diagnosis leverages the (exact) fact that the vapor pressure of a gas is
  equal to the saturation vapor pressure of its dewpoint temperature.

  Args:
    td: Dewpoint temperature at height zs, in [K].
    msl: Mean sea level pressure, in [Pa].
    zs: The surface height with respect to sea level.

  Returns:
    The specific humidity at height zs.
  """
  e = sat_vapor_pressure(td)
  p = pressure_from_msl_and_height(msl, zs)
  q = _EPS * e / (p - (1 - _EPS) * e)
  return q


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


# pylint: disable=invalid-name
def _find_eqvar(Ta: float, RH: float, **solve_kwargs) -> list[Any]:
  """Returns the equivalent variables for heat index calculation.

  Args:
    Ta: Temperature, in [K].
    RH: Relative humidity, between 0 and 1 (not enforced).
    **solve_kwargs: Keyword arguments to pass to `optimize.brentq`.

  Returns: [eqvar_name, phi, Rf, Rs, dTcdt]
    A list of equivalent variables, formed by the name of the main equivalent
    variable, and the values of all equivalent variables.
  """

  # Thermodynamic parameters
  E0v = 2.3740e6  # J/kg
  cva = 719.0  # J/kg/K
  cvv = 1418.0  # J/kg/K
  cvl = 4119.0  # J/kg/K
  cpa = cva + _R_D

  # The latent heat of vaporization of water
  Le = lambda T: E0v + (cvv - cvl) * (T - _T_REF) + _R_V * T

  # Thermoregulatory parameters.
  sigma = 5.67e-8  # Stefan-Boltzmann constant [W/m^2/K^4]
  epsilon = 0.97  # Emissivity of surface, steadman1979
  M = 83.6  # Mass of average US adults [kg], fryar2018
  H = 1.69  # Height of average US adults [m], fryar2018
  A = 0.202 * (M**0.425) * (H**0.725)  # DuBois formula coefficient, parson2014
  cpc = 3492.0  # Specific heat capacity of core [J/kg/K], gagge1972
  C = M * cpc / A  # Heat capacity of core
  r = 124.0  # Zf/Rf [Pa/K], steadman1979
  Q = 180.0  # Metabolic rate per skin area [W/m^2], steadman1979
  phi_salt = 0.9  # Saturation vapor pressure of saline solution, steadman1979
  Tc = 310.0  # Core temperature [K], steadman1979
  Pc = phi_salt * sat_vapor_pressure(Tc)  # Core vapor pressure
  L = Le(310.0)  # Latent heat of vaporization at 310 K
  p = 1.013e5  # Surface atmospheric pressure
  eta = 1.43e-6  # inhaled mass / metabolic rate [kg/J], steadman1979
  # Mass transfer resistance through air, exposed part of skin
  Za = 60.6 / 17.4  # Pa m^2/W
  # Mass transfer resistance through air, clothed part of skin
  Za_bar = 60.6 / 11.6  # Pa m^2/W
  # Mass transfer resistance through air, when being naked
  Za_un = 60.6 / 12.3  # Pa m^2/W

  # Thermoregulatory functions
  def Qv(Ta: float, Pa: float) -> float:
    """Returns the respiratory heat loss, in [W/m^2]."""
    return eta * Q * (cpa * (Tc - Ta) + L * _R_D / (p * _R_V) * (Pc - Pa))

  def Zs(Rs: float) -> float:
    """Returns the mass transfer resistance through skin, in [Pa m^2/W]."""
    return 52.1 if Rs == 0.0387 else 6.0e8 * Rs**5

  def Ra(Ts: float, Ta: float) -> float:
    """Returns the heat transfer resistance through air of exposed skin."""
    hc = 17.4
    phi_rad = 0.85
    hr = epsilon * phi_rad * sigma * (Ts**2 + Ta**2) * (Ts + Ta)
    return 1.0 / (hc + hr)

  def Ra_bar(Tf: float, Ta: float) -> float:
    """Returns the heat transfer resistance through air of clothed skin."""
    hc = 11.6
    phi_rad = 0.79
    hr = epsilon * phi_rad * sigma * (Tf**2 + Ta**2) * (Tf + Ta)
    return 1.0 / (hc + hr)

  def Ra_un(Ts: float, Ta: float) -> float:
    """Returns the heat transfer resistance through air of naked skin."""
    hc = 12.3
    phi_rad = 0.80
    hr = epsilon * phi_rad * sigma * (Ts**2 + Ta**2) * (Ts + Ta)
    return 1.0 / (hc + hr)

  Pa = RH * sat_vapor_pressure(Ta)  # air vapor pressure

  # Initialize equivalent variables.
  Rs = 0.0387  # Heat transfer resistance through skin [m^2K/W]
  phi = 0.84  # Covering fraction
  dTcdt = 0.0  # Rate of change in Tc [K/s]
  m = (Pc - Pa) / (Zs(Rs) + Za)
  m_bar = (Pc - Pa) / (Zs(Rs) + Za_bar)
  Ts = optimize.brentq(
      lambda Ts: (Ts - Ta) / Ra(Ts, Ta)
      + (Pc - Pa) / (Zs(Rs) + Za)
      - (Tc - Ts) / Rs,
      max(1e-2, min(Tc, Ta) - Rs * abs(m)),
      max(Tc, Ta) + Rs * abs(m),
      **solve_kwargs,
  )

  flux1 = (
      Q - Qv(Ta, Pa) - (1.0 - phi) * (Tc - Ts) / Rs
  )  # C*dTc/dt when Rf=Zf=\inf
  # Region I
  if flux1 <= 0.0:
    eqvar_name = "phi"
    phi = 1.0 - (Q - Qv(Ta, Pa)) * Rs / (Tc - Ts)
    Rf = float("inf")

  else:
    Tf = optimize.brentq(
        lambda Tf: (Tf - Ta) / Ra_bar(Tf, Ta)
        + (Pc - Pa) / (Zs(Rs) + Za_bar)
        - (Tc - Tf) / Rs,
        max(1e-2, min(Tc, Ta) - Rs * abs(m_bar)),
        max(Tc, Ta) + Rs * abs(m_bar),
        **solve_kwargs,
    )
    flux2 = (
        Q - Qv(Ta, Pa) - (1.0 - phi) * (Tc - Ts) / Rs - phi * (Tc - Tf) / Rs
    )  # C*dTc/dt when Rf=Zf=0

    # Regions II & III
    if flux2 <= 0.0:
      eqvar_name = "Rf"
      Ts_bar = Tc - (Q - Qv(Ta, Pa)) * Rs / phi + (1.0 / phi - 1.0) * (Tc - Ts)
      Tf = optimize.brentq(
          lambda Tf: (Tf - Ta) / Ra_bar(Tf, Ta)
          + (Pc - Pa)
          * (Tf - Ta)
          / ((Zs(Rs) + Za_bar) * (Tf - Ta) + r * Ra_bar(Tf, Ta) * (Ts_bar - Tf))
          - (Tc - Ts_bar) / Rs,
          Ta,
          Ts_bar,
          **solve_kwargs,
      )
      Rf = Ra_bar(Tf, Ta) * (Ts_bar - Tf) / (Tf - Ta)

    else:  # region IV,V,VI
      Rf = 0.0
      flux3 = Q - Qv(Ta, Pa) - (Tc - Ta) / Ra_un(Tc, Ta) - (Pc - Pa) / Za_un
      # Regions IV & V
      if flux3 < 0.0:
        Ts = optimize.brentq(
            lambda Ts: (Ts - Ta) / Ra_un(Ts, Ta)
            + (Pc - Pa) / (Zs((Tc - Ts) / (Q - Qv(Ta, Pa))) + Za_un)
            - (Q - Qv(Ta, Pa)),
            1e-2,
            Tc,
            **solve_kwargs,
        )
        Rs = (Tc - Ts) / (Q - Qv(Ta, Pa))
        eqvar_name = "Rs"
        Ps = Pc - (Pc - Pa) * Zs(Rs) / (Zs(Rs) + Za_un)
        if Ps > phi_salt * sat_vapor_pressure(Ts):  # region V
          Ts = optimize.brentq(
              lambda Ts: (Ts - Ta) / Ra_un(Ts, Ta)
              + (phi_salt * sat_vapor_pressure(Ts) - Pa) / Za_un
              - (Q - Qv(Ta, Pa)),
              1e-2,
              Tc,
              **solve_kwargs,
          )
          Rs = (Tc - Ts) / (Q - Qv(Ta, Pa))
          eqvar_name = "Rs*"

      # Region VI
      else:
        Rs = 0.0
        eqvar_name = "dTcdt"
        dTcdt = (1.0 / C) * flux3

  return [eqvar_name, phi, Rf, Rs, dTcdt]


def _heat_index_from_eqvar(
    eqvar_name: str,
    eqvar: float,
    hi_min: float = 0.0,
    hi_max: float = 1000.0,
    **solve_kwargs,
) -> float:
  """Returns the heat index from the equivalent variables.

  Args:
    eqvar_name: Name of the main equivalent variable for the region.
    eqvar: Value of the equivalent variable.
    hi_min: Minimum value of the heat index considered, must be non-negative.
    hi_max: Maximum value of the heat index considered, must be lower than 1000.
    **solve_kwargs: Keyword arguments to pass to `optimize.brentq`.

  Returns:
    The heat index, in [K].
  """
  Pa0 = 1.6e3  # Reference pressure in regions III, IV, V, VI [Pa], steadman1979
  # region I
  if eqvar_name == "phi":
    T = optimize.brentq(
        lambda T: _find_eqvar(T, 1.0)[1] - eqvar, hi_min, 240.0, **solve_kwargs
    )
  # regions II and III
  elif eqvar_name == "Rf":
    T = optimize.brentq(
        lambda T: _find_eqvar(T, min(1.0, Pa0 / sat_vapor_pressure(T)))[2]
        - eqvar,
        230.0,
        300.0,
        **solve_kwargs,
    )
  # regions IV and V
  elif eqvar_name == "Rs" or eqvar_name == "Rs*":
    T = optimize.brentq(
        lambda T: _find_eqvar(T, Pa0 / sat_vapor_pressure(T))[3] - eqvar,
        295.0,
        350.0,
        **solve_kwargs,
    )
  # region VI
  else:
    T = optimize.brentq(
        lambda T: _find_eqvar(T, Pa0 / sat_vapor_pressure(T))[4] - eqvar,
        340.0,
        hi_max,
        **solve_kwargs,
    )
  return T


def heat_index(
    Ta: float,
    RH: float,
    eqvar_tol: float = 1e-5,
    t_tol: float = 1e-2,
    hi_min: float = 173.15,
    hi_max: float = 400.0,
) -> float:
  """Returns the heat index from the temperature and relative humidity.

  This function is not vectorized, so it should be vectorized using
  `numpy.vectorize`, or xarray's `xr.apply_ufunc` with `vectorize=True`.
  The returned heat index follows the definition of Lu and Romps (2022).
  This implementation is a faster version of the original implementation:
  https://github.com/ankurmahesh/earth2mip-fork/blob/HENS/scripts/Calculate_Heat_Index_Lookup.ipynb.

  Args:
    Ta: Temperature, in [K].
    RH: Relative humidity, between 0 and 1.
    eqvar_tol: Absolute tolerance of root solves in `_find_eqvar`.
    t_tol: Absolute tolerance of heat index.
    hi_min: Minimum value of the heat index considered. Increase to accelerate
      root find.
    hi_max: Maximum value of the heat index considered. Reduce to accelerate
      root find.

  Returns:
    The heat index, in [K].
  """
  dic = {"phi": 1, "Rf": 2, "Rs": 3, "Rs*": 3, "dTcdt": 4}
  try:
    eqvars = _find_eqvar(Ta, RH, xtol=eqvar_tol)
    T = _heat_index_from_eqvar(
        eqvars[0],
        eqvars[dic[eqvars[0]]],
        xtol=t_tol,
        hi_min=hi_min,
        hi_max=hi_max,
    )
  except ValueError as e:
    logging.warning("Failed to compute heat index: %s", e)
    return np.nan
  return T


# pylint: enable=invalid-name


def noaa_heat_index(TF, RH):  # pylint: disable=invalid-name
  """Computes heat index according to NOAA's definition.

  NOAA's definition can be found at
  https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml. It follows the
  regression analysis of Rothfusz, and therefore it is not applicable in
  conditions of extreme temperature or relative humidity. Therefore, this
  is definition should not be used to analyze heat extremes.

  Args:
    TF: Temperature in degrees Fahrenheit.
    RH: Relative humidity, as a percentage.

  Returns:
    The heat index, in Kelvin.
  """
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
  hi = (hi - 32.0) * 5.0 / 9.0 + _T_REF  # Convert to Kelvin
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
