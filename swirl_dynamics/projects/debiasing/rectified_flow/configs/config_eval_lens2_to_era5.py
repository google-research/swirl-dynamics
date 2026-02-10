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
    "2m_temperature": None,
    "specific_humidity": {"level": 1000},
    "mean_sea_level_pressure": None,
    "10m_magnitude_of_wind": None,
}

# pylint: disable=line-too-long
_ERA5_DATASET_PATH = "/lzepedanunez/data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = "/lzepedanunez/data/era5/stats/daily_mean_240x121_all_variables_and_wind_speed_1961-2000.zarr"

# Variables in the LENS2 model to be used for debiasing.
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV")

# Indices for the ensemble members (each index is a LENS2 member).
_LENS2_MEMBER_INDEXER = (
    "cmip6_1001_001",
    "cmip6_1041_003",
    "cmip6_1081_005",
    "cmip6_1121_007",
    "cmip6_1231_001",
    "cmip6_1231_003",
    "cmip6_1231_005",
    "cmip6_1231_007",
    "smbb_1011_001",
    "smbb_1301_011",
    "cmip6_1281_001",
    "cmip6_1301_003",
    "smbb_1251_013",
    "smbb_1301_020",
)

_LENS2_VARIABLES = {v: _LENS2_MEMBER_INDEXER[0] for v in _LENS2_VARIABLE_NAMES}

# Interpolated dataset to match the resolution of the ERA5 data set.
_LENS2_DATASET_PATH = (
    "/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
)

# Statistics for the LENS2 dataset.
_LENS2_STATS_PATH = "/lzepedanunez/data/lens2/stats/all_variables_240x121_lonlat_1961-2000.zarr"

# Mean and STD of the statistics for the LENS2 dataset.
_LENS2_MEAN_STATS_PATH = "/lzepedanunez/data/lens2/stats/lens2_mean_stats_all_variables_240x121_lonlat_1961-2000.zarr"
_LENS2_STD_STATS_PATH = "/lzepedanunez/data/lens2/stats/lens2_std_stats_all_variables_240x121_lonlat_1961-2000.zarr"
# pylint: enable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = "evaluation_reflow_lens2_to_era5"
  config.model_dir = "/lzepedanunez/xm/debiasing/reflow_lens2_to_era5_4_member_ens_all_surface_variables_batch_OT_sweep_vlp_long_training_large_channel/114628134"
  config.date_range = ("2000", "2010")
  config.batch_size_eval = 16
  config.weighted_norm = True
  config.variables = (
      "temperature",
      "specific_humidity",
      "mean_sea_level_pressure",
      "wind_speed",
  )
  config.lens2_member_indexer = _LENS2_MEMBER_INDEXER
  config.lens2_variables = None
  config.era5_variables = None

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
