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

r"""Configuration for inference of experiments LENS2 to ERA5 data.

We bias correct selected variables from LENS2 to ERA5. from 1960 to 2099.

"""


import ml_collections

_ERA5_VARIABLES = {
    "10m_magnitude_of_wind": None,
    "2m_temperature": None,
    "geopotential": {"level": [200, 500]},
    "mean_sea_level_pressure": None,
    "specific_humidity": {"level": 1000},
    "u_component_of_wind": {"level": [200, 850]},
    "v_component_of_wind": {"level": [200, 850]},
}

# Cell where the data is located.
cell_data = "eb"  # "eb" for VF and "oa" for A100 and ta for H100.

_ERA5_DATASET_PATH = f"/cns/{cell_data}-d/home/attractor/lzepedanunez/data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = f"/cns/{cell_data}-d/home/attractor/lzepedanunez/data/era5/climat/1p5deg_11vars_windspeed_1961-2000_daily_v2.zarr"

_LENS2_DATASET_PATH = (
    f"/cns/{cell_data}-d/home/attractor/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
)
_LENS2_STATS_PATH = f"/cns/{cell_data}-d/home/attractor/lzepedanunez/data/lens2/climat/lens2_240x121_lonlat_clim_daily_1961_to_2000_31_dw.zarr"
_LENS2_MEAN_CLIMATOLOGY_PATH = f"/cns/{cell_data}-d/home/attractor/lzepedanunez/data/lens2/climat/mean_lens2_240x121_lonlat_clim_daily_1961_to_2000.zarr"
_LENS2_STD_CLIMATOLOGY_PATH = f"/cns/{cell_data}-d/home/attractor/lzepedanunez/data/lens2/climat/std_lens2_240x121_lonlat_clim_daily_1961_to_2000.zarr"

_LENS2_MEMBER_INDEXER = ("cmip6_1001_001",)

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
_LENS2_VARIABLES = {v: _LENS2_MEMBER_INDEXER[0] for v in _LENS2_VARIABLE_NAMES}
# pylint: enable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = "inference_reflow_lens2_to_era5_with_clim_and_time_cons"
  config.model_dir = "/lzepedanunez/xm/debiasing/reflow_lens2_to_era5_4_new_member_ens_all_surface_climatology_time_coherent_tpu/148227746/1"
  config.date_range = ("1960", "2100")
  config.batch_size_eval = 32  # check if this is fine with VFs.
  config.variables = (
      "wind_speed",
      "temperature",
      "geopotential_200",
      "geopotential_200",
      "mean_sea_level_pressure",
      "specific_humidity",
      "u_component_of_wind_200",
      "u_component_of_wind_850",
      "v_component_of_wind_200",
      "v_component_of_wind_850",
  )

  config.lens2_variable_names = _LENS2_VARIABLE_NAMES
  config.lens2_member_indexer = _LENS2_MEMBER_INDEXER
  # avoid nested containers. It doesn't work very well with the configdict.
  config.era5_variables = _ERA5_VARIABLES
  config.lens2_variables = _LENS2_VARIABLES

  config.input_dataset_path = _LENS2_DATASET_PATH
  config.input_climatology = _LENS2_STATS_PATH
  config.input_mean_stats_path = _LENS2_MEAN_CLIMATOLOGY_PATH
  config.input_std_stats_path = _LENS2_STD_CLIMATOLOGY_PATH
  config.output_dataset_path = _ERA5_DATASET_PATH
  config.output_climatology = _ERA5_STATS_PATH
  config.output_variables = _ERA5_VARIABLES
  config.num_sampling_steps = 100

  return config


def sweep(add):
  """Define param sweep."""
  for lens2_member_indexer in (
      ("cmip6_1001_001",),
      ("cmip6_1021_002",),
      ("cmip6_1041_003",),
      ("cmip6_1061_004",),
      ("cmip6_1081_005",),
      ("cmip6_1101_006",),
      ("cmip6_1121_007",),
      ("cmip6_1141_008",),
      ("cmip6_1161_009",),
      ("cmip6_1181_010",),
      ("cmip6_1231_001",),
      ("cmip6_1231_002",),
      ("cmip6_1231_003",),
      ("cmip6_1231_004",),
      ("cmip6_1231_005",),
      ("cmip6_1231_006",),
      ("cmip6_1231_007",),
      ("cmip6_1231_008",),
      ("cmip6_1231_009",),
      ("cmip6_1231_010",),
      ("cmip6_1251_001",),
      ("cmip6_1251_002",),
      ("cmip6_1251_003",),
      ("cmip6_1251_004",),
      ("cmip6_1251_005",),
      ("cmip6_1251_006",),
      ("cmip6_1251_007",),
      ("cmip6_1251_008",),
      ("cmip6_1251_009",),
      ("cmip6_1251_010",),
      ("cmip6_1281_001",),
      ("cmip6_1281_002",),
      ("cmip6_1281_003",),
      ("cmip6_1281_004",),
      ("cmip6_1281_005",),
      ("cmip6_1281_006",),
      ("cmip6_1281_007",),
      ("cmip6_1281_008",),
      ("cmip6_1281_009",),
      ("cmip6_1281_010",),
      ("cmip6_1301_001",),
      ("cmip6_1301_002",),
      ("cmip6_1301_003",),
      ("cmip6_1301_004",),
      ("cmip6_1301_005",),
      ("cmip6_1301_006",),
      ("cmip6_1301_007",),
      ("cmip6_1301_008",),
      ("cmip6_1301_009",),
      ("cmip6_1301_010",),
      ("smbb_1011_001",),
      ("smbb_1031_002",),
      ("smbb_1051_003",),
      ("smbb_1071_004",),
      ("smbb_1091_005",),
      ("smbb_1111_006",),
      ("smbb_1131_007",),
      ("smbb_1151_008",),
      ("smbb_1171_009",),
      ("smbb_1191_010",),
      ("smbb_1231_011",),
      ("smbb_1231_012",),
      ("smbb_1231_013",),
      ("smbb_1231_014",),
      ("smbb_1231_015",),
      ("smbb_1231_016",),
      ("smbb_1231_017",),
      ("smbb_1231_018",),
      ("smbb_1231_019",),
      ("smbb_1231_020",),
      ("smbb_1251_011",),
      ("smbb_1251_012",),
      ("smbb_1251_013",),
      ("smbb_1251_014",),
      ("smbb_1251_015",),
      ("smbb_1251_016",),
      ("smbb_1251_017",),
      ("smbb_1251_018",),
      ("smbb_1251_019",),
      ("smbb_1251_020",),
      ("smbb_1281_011",),
      ("smbb_1281_012",),
      ("smbb_1281_013",),
      ("smbb_1281_014",),
      ("smbb_1281_015",),
      ("smbb_1281_016",),
      ("smbb_1281_017",),
      ("smbb_1281_018",),
      ("smbb_1281_019",),
      ("smbb_1281_020",),
      ("smbb_1301_011",),
      ("smbb_1301_012",),
      ("smbb_1301_013",),
      ("smbb_1301_014",),
      ("smbb_1301_015",),
      ("smbb_1301_016",),
      ("smbb_1301_017",),
      ("smbb_1301_018",),
      ("smbb_1301_019",),
      ("smbb_1301_020",),
  ):
    for model_dir in [
        "/lzepedanunez/xm/debiasing/reflow_lens2_to_era5_4_new_member_clim_3d_10_vars_8_time_6_lvls_tpu_large_3_tmp_attn_bs_16_1980_2000/151818801/1",
    ]:
      add(
          model_dir=model_dir,
          lens2_member_indexer=lens2_member_indexer,
      )
