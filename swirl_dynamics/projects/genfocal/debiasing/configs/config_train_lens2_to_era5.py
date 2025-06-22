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

r"""Default Hyperparameter configuration for LENS2 to ERA5 data.

This is the config file for the training of the models.

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

# Variables in the climate model to be used for debiasing.
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

# Tuple of dictionaries (due to xarray indexing interface) containing the
# members to be used for training.
_LENS2_MEMBER_INDEXER = (
    {"member": "cmip6_1001_001"},
    {"member": "cmip6_1251_001"},
    {"member": "cmip6_1301_010"},
    {"member": "smbb_1301_020"},
)

# pylint: enable=line-too-long
# Interpolated dataset to match the resolution of the ERA5 data set.
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
_LENS2_STD_CLIMATOLOGY_PATH = (
    "data/lens2/std_lens2_240x121_10_vars_lonlat_clim_daily_1961_to_2000.zarr"
)

# pylint: disable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = "genfocal_lens2_to_era5"

  config.era5_variables = _ERA5_VARIABLES
  config.era5_dataset_path = _ERA5_DATASET_PATH
  config.era5_stats_path = _ERA5_STATS_PATH
  config.lens2_member_indexer = _LENS2_MEMBER_INDEXER
  config.lens2_variable_names = _LENS2_VARIABLE_NAMES
  config.lens2_dataset_path = _LENS2_DATASET_PATH
  config.lens2_stats_path = _LENS2_STATS_PATH
  config.lens2_mean_stats_path = _LENS2_MEAN_CLIMATOLOGY_PATH
  config.lens2_std_stats_path = _LENS2_STD_CLIMATOLOGY_PATH

  config.distributed = True

  # Train params.
  config.num_train_steps = 300_000
  config.seed = 666
  config.metric_aggregation_steps = 1_000
  config.save_interval_steps = 10_000
  config.max_checkpoints_to_keep = 10
  config.learning_rate = 0.00005
  config.transition_steps = 50_000
  config.max_norm = 0.6
  config.eval_every_steps = 2_000
  config.decay_rate = 0.98
  config.staircase = True
  config.num_batches_per_eval = 4
  config.time_sampler = "uniform"

  # Optimizer params.
  config.initial_lr = 1e-7
  config.peak_lr = 1e-4
  config.warmup_steps = 50_000
  config.decay_steps = 500
  config.end_lr = 1e-7
  config.beta1 = 0.9

  # Date aligned with ensemble loader.
  config.time_to_channel = False  # creates 3D tensors.

  config.date_range_train = ("1980", "2000")
  config.date_range_eval = ("2005", "2010")
  config.shuffle = True
  config.batch_size = 32  # effective batch size is batch_size/time_batch_size.
  config.chunk_size = 32  # We only call one chunk per device.
  config.time_batch_size = 8
  config.batch_size_eval = 32
  # So far it doesn't work with more than 0 workers.
  config.num_workers = 0

  # Model params
  config.out_channels = 10
  config.num_channels = (128, 128, 128, 256, 256, 256)
  config.downsample_ratio = (2, 2, 2, 2, 2, 2)

  config.bfloat16 = False

  config.use_3d_model = True
  config.resize_to_shape = (256, 128)
  config.num_blocks = 6
  config.dropout_rate = 0.5
  config.noise_embed_dim = 256
  config.padding = "LONLAT"
  config.use_spatial_attention = (False, False, False, False, True, True)
  config.use_temporal_attention = (False, False, False, False, True, True)
  config.use_position_encoding = True
  config.num_heads = 128
  config.cond_embed_dim = 128
  config.ema_decay = 0.99
  config.final_act_fun = lambda x: x
  config.use_skips = True
  config.use_weight_global_skip = True
  config.input_shapes = ((1, 8, 240, 121, 10), (1, 8, 240, 121, 10))
  config.same_dimension = True
  config.min_time = 1e-4
  config.max_time = 1 - 1e-4
  config.normalize_qk = True

  return config
