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

r"""Configuration for evaluation of experiments LENS2 to ERA5 data.

Here we focus on the evaluation of the multiple ensemble members of the LENS2
dataset. Each one of the members is evaluated against the ERA5 dataset.

"""

import ml_collections

# Variables in the weather model to be used for debiasing.
_ERA5_VARIABLES = {
    "10m_magnitude_of_wind": None,
    "2m_temperature": None,
    "geopotential": {"level": [200, 500]},
    "mean_sea_level_pressure": None,
    "specific_humidity": {"level": 1000},
    "u_component_of_wind": {"level": [200, 850]},
    "v_component_of_wind": {"level": [200, 850]},
}

# Variables in the LENS2 model to be used for debiasing.
_LENS2_VARIABLE_NAMES = (
    "WSPDSRFAV",
    "TREFHT",
    "Z200",
    "Z500",
    "PSL",
    "QREFHT",
    "U200",
    "U850",
    "V200",
    "V850",
)

# Indices for the ensemble members (each index is a LENS2 member).
_LENS2_MEMBER_INDEXER = (
    {"member": "cmip6_1001_001"},
    {"member": "cmip6_1251_001"},
    {"member": "cmip6_1301_010"},
    {"member": "smbb_1301_020"},
)

_LENS2_VARIABLES = {v: _LENS2_MEMBER_INDEXER[0] for v in _LENS2_VARIABLE_NAMES}

# pylint: enable=line-too-long
_ERA5_DATASET_PATH = "data/era5/era5_240x121_lonlat_1980-2020_10_vars.zarr"
_ERA5_STATS_PATH = "data/era5/1p5deg_11vars_windspeed_1961-2000_daily_v2.zarr"

# Interpolated dataset to match the resolution of the ERA5 data set.
_LENS2_DATASET_PATH = "data/lens2/lens2_240x121_lonlat_1960-2020_10_vars_4_train_members.zarr"

# Statistics for the LENS2 dataset.
_LENS2_STATS_PATH = "data/lens2/lens2_240x121_10_vars_4_members_lonlat_clim_daily_1961_to_2000_31_dw.zarr"

# Mean and STD of the statistics for the LENS2 dataset.
_LENS2_MEAN_CLIMATOLOGY_PATH = (
    "data/lens2/mean_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr"
)
_LENS2_STD_STATS_PATH = (
    "data/lens2/std_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr"
)

# pylint: disable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = "validation_reflow_lens2_to_era5"
  config.model_dir = "experiment/"
  config.date_range = ("2000", "2010")
  config.batch_size_eval = 16
  config.weighted_norm = True
  config.variables = (
      "wind_speed",
      "temperature",
      "geopotential_200",
      "geopotential_500",
      "mean_sea_level_pressure",
      "specific_humidity",
      "u_component_of_wind_200",
      "u_component_of_wind_850",
      "v_component_of_wind_200",
      "v_component_of_wind_850",
  )
  config.lens2_member_indexer = _LENS2_MEMBER_INDEXER
  config.lens2_variables = _LENS2_VARIABLES
  config.era5_variables = _ERA5_VARIABLES
  config.input_dataset_path = _LENS2_DATASET_PATH
  config.input_climatology = _LENS2_STATS_PATH
  config.input_mean_stats_path = _LENS2_MEAN_CLIMATOLOGY_PATH
  config.input_std_stats_path = _LENS2_STD_CLIMATOLOGY_PATH
  config.output_dataset_path = _ERA5_DATASET_PATH
  config.output_climatology = _ERA5_STATS_PATH
  return config


def sweep(add):
  """Define param sweep."""
  # We run the evaluation on a subset of the LENS2 members. Here we run 14
  # member in parallel.
  for lens2_member_indexer in (
      ("cmip6_1001_001",),
      ("cmip6_1041_003",),
      ("cmip6_1081_005",),
      ("cmip6_1121_007",),
      ("cmip6_1231_001",),
      ("cmip6_1231_003",),
      ("cmip6_1231_005",),
      ("cmip6_1231_007",),
      ("smbb_1011_001",),
      ("smbb_1301_011",),
      ("cmip6_1281_001",),
      ("cmip6_1301_003",),
      ("smbb_1251_013",),
      ("smbb_1301_020",),
  ):
    add(lens2_member_indexer=lens2_member_indexer)
