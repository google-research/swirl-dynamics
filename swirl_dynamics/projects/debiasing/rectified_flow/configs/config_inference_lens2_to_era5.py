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

r"""Configuration for inference of experiments LENS2 to ERA5 data.

We bias correct selected variables from LENS2 to ERA5. from 1960 to 2099.

"""

import ml_collections

# Variables in the weather model to be used for debiasing.
_ERA5_VARIABLES = {
    "2m_temperature": None,
    "specific_humidity": {"level": 1000},
    "mean_sea_level_pressure": None,
    "10m_magnitude_of_wind": None,
}

_ERA5_WIND_COMPONENTS = {}

# pylint: disable=line-too-long
_ERA5_DATASET_PATH = "/lzepedanunez/data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = "/lzepedanunez/data/era5/stats/daily_mean_240x121_all_variables_and_wind_speed_1961-2000.zarr"

# Variables in the LENS2 model to be used for debiasing.
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV")
_LENS2_MEMBER_INDEXER = (
    "cmip6_1001_001",
)

_LENS2_VARIABLES = {v: _LENS2_MEMBER_INDEXER[0] for v in _LENS2_VARIABLE_NAMES}

# Interpolated dataset to match the resolution of the ERA5 data set.
_LENS2_DATASET_PATH = (
    "/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
)
_LENS2_STATS_PATH = "/lzepedanunez/data/lens2/stats/all_variables_240x121_lonlat_1961-2000.zarr"

_LENS2_MEAN_STATS_PATH = "/lzepedanunez/data/lens2/stats/lens2_mean_stats_all_variables_240x121_lonlat_1961-2000.zarr"
_LENS2_STD_STATS_PATH = "/lzepedanunez/data/lens2/stats/lens2_std_stats_all_variables_240x121_lonlat_1961-2000.zarr"
# pylint: enable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = "evaluation_reflow_lens2_to_era5"
  config.model_dir = "/lzepedanunez/xm/debiasing/reflow_lens2_to_era5_ens_all_surface_variables_batch_OT_sweep_vlp_num_chunks/112720508/11"
  config.date_range = ("1960", "2100")
  config.batch_size_eval = 16
  config.variables = (
      "temperature",
      "specific_humidity",
      "mean_sea_level_pressure",
      "wind_speed",
  )
  config.lens2_variables = None
  config.lens2_variables_names = _LENS2_VARIABLE_NAMES
  config.lens2_member_indexer = _LENS2_MEMBER_INDEXER
  config.era5_variables = None

  return config


def sweep(add):
  """Define param sweep."""
  # We run the for the full LENS2 ensemble.
  for lens2_member_indexer in (
      ("cmip6_1001_001",), ("cmip6_1021_002",),
      ("cmip6_1041_003",), ("cmip6_1061_004",),
      ("cmip6_1081_005",), ("cmip6_1101_006",),
      ("cmip6_1121_007",), ("cmip6_1141_008",),
      ("cmip6_1161_009",), ("cmip6_1181_010",),
      ("cmip6_1231_001",), ("cmip6_1231_002",),
      ("cmip6_1231_003",), ("cmip6_1231_004",),
      ("cmip6_1231_005",), ("cmip6_1231_006",),
      ("cmip6_1231_007",), ("cmip6_1231_008",),
      ("cmip6_1231_009",), ("cmip6_1231_010",),
      ("cmip6_1251_001",), ("cmip6_1251_002",),
      ("cmip6_1251_003",), ("cmip6_1251_004",),
      ("cmip6_1251_005",), ("cmip6_1251_006",),
      ("cmip6_1251_007",), ("cmip6_1251_008",),
      ("cmip6_1251_009",), ("cmip6_1251_010",),
      ("cmip6_1281_001",), ("cmip6_1281_002",),
      ("cmip6_1281_003",), ("cmip6_1281_004",),
      ("cmip6_1281_005",), ("cmip6_1281_006",),
      ("cmip6_1281_007",), ("cmip6_1281_008",),
      ("cmip6_1281_009",), ("cmip6_1281_010",),
      ("cmip6_1301_001",), ("cmip6_1301_002",),
      ("cmip6_1301_003",), ("cmip6_1301_004",),
      ("cmip6_1301_005",), ("cmip6_1301_006",),
      ("cmip6_1301_007",), ("cmip6_1301_008",),
      ("cmip6_1301_009",), ("cmip6_1301_010",),
      ("smbb_1011_001",), ("smbb_1031_002",),
      ("smbb_1051_003",), ("smbb_1071_004",),
      ("smbb_1091_005",), ("smbb_1111_006",),
      ("smbb_1131_007",), ("smbb_1151_008",),
      ("smbb_1171_009",), ("smbb_1191_010",),
      ("smbb_1231_011",), ("smbb_1231_012",),
      ("smbb_1231_013",), ("smbb_1231_014",),
      ("smbb_1231_015",), ("smbb_1231_016",),
      ("smbb_1231_017",), ("smbb_1231_018",),
      ("smbb_1231_019",), ("smbb_1231_020",),
      ("smbb_1251_011",), ("smbb_1251_012",),
      ("smbb_1251_013",), ("smbb_1251_014",),
      ("smbb_1251_015",), ("smbb_1251_016",),
      ("smbb_1251_017",), ("smbb_1251_018",),
      ("smbb_1251_019",), ("smbb_1251_020",),
      ("smbb_1281_011",), ("smbb_1281_012",),
      ("smbb_1281_013",), ("smbb_1281_014",),
      ("smbb_1281_015",), ("smbb_1281_016",),
      ("smbb_1281_017",), ("smbb_1281_018",),
      ("smbb_1281_019",), ("smbb_1281_020",),
      ("smbb_1301_011",), ("smbb_1301_012",),
      ("smbb_1301_013",), ("smbb_1301_014",),
      ("smbb_1301_015",), ("smbb_1301_016",),
      ("smbb_1301_017",), ("smbb_1301_018",),
      ("smbb_1301_019",), ("smbb_1301_020",),):
    add(lens2_member_indexer=lens2_member_indexer,)
