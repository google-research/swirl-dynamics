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

r"""Default Hyperparameter configuration for LENS2 to ERA5 data.

This is the config file for the training of the models.

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
_ERA5_DATASET_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/era5/daily_mean_1959-2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr"
_ERA5_STATS_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/era5/stats/daily_mean_240x121_all_variables_and_wind_speed_1961-2000.zarr"

# Variables in the climate model to be used for debiasing.
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV")

# Tuple of dictionaries (due to xarray indexing interface) containing the
# members to be used for training.
_LENS2_MEMBER_INDEXER = (
    {"member": "cmip6_1001_001"},
    {"member": "cmip6_1021_002"},
    {"member": "cmip6_1041_003"},
    {"member": "cmip6_1061_004"},
)

# Interpolated dataset to match the resolution of the ERA5 data set.
_LENS2_DATASET_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/lens2/lens2_240x121_lonlat.zarr"
_LENS2_STATS_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/lens2/stats/all_variables_240x121_lonlat_1961-2000.zarr"

_LENS2_MEAN_STATS_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/lens2/stats/lens2_mean_stats_all_variables_240x121_lonlat_1961-2000.zarr"
_LENS2_STD_STATS_PATH = "/cns/uy-d/home/sim-research/lzepedanunez/data/lens2/stats/lens2_std_stats_all_variables_240x121_lonlat_1961-2000.zarr"
# pylint: enable=line-too-long


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.experiment = "reflow_lens2_to_era5_ens_ot"

  config.era5_dataset_path = _ERA5_DATASET_PATH
  config.era5_stats_path = _ERA5_STATS_PATH
  config.lens2_member_indexer = _LENS2_MEMBER_INDEXER
  config.lens2_variable_names = _LENS2_VARIABLE_NAMES
  config.lens2_dataset_path = _LENS2_DATASET_PATH
  config.lens2_stats_path = _LENS2_STATS_PATH
  config.lens2_mean_stats_path = _LENS2_MEAN_STATS_PATH
  config.lens2_std_stats_path = _LENS2_STD_STATS_PATH

  # Wheter to use distributed training or not.
  config.distributed = True

  # Parameters for training.
  config.num_train_steps = 2_000_000
  config.seed = 777
  config.metric_aggregation_steps = 500
  config.save_interval_steps = 6_000
  config.max_checkpoints_to_keep = 10
  config.learning_rate = 0.00005
  config.transition_steps = 50_000
  config.max_norm = 0.6
  config.eval_every_steps = 2_000
  config.decay_rate = 0.98
  config.staircase = True
  config.num_batches_per_eval = 4
  config.time_sampler = "uniform"

  # Optimizer parameters.
  config.initial_lr = 1e-7
  config.peak_lr = 1e-4
  config.warmup_steps = 50_000
  config.decay_steps = 500
  config.end_lr = 1e-7
  config.beta1 = 0.9

  # Different options for the data loader.
  config.ens_chunked_aligned_loader = True
  config.random_local_shuffle = False
  config.batch_ot_shuffle = False  # To make the training faster.
  config.norm_stats_loader = True

  # Parameters for the data loader.
  config.data_range_train = ("1990", "2000")  # Reduced train period.
  config.data_range_eval = ("2001", "2010")
  config.normalize = True
  config.shuffle = True

  # This is supposed to be run on 4 replicas, each replica containing 8 chips.
  config.batch_size = 8  # one sample per chip.
  config.batch_size_eval = 8
  # Optimization flags for the data loader.
  config.num_chunks = 1
  config.num_workers = 0

  # Model params
  config.out_channels = 4
  config.num_channels = (128, 256, 512, 1024)
  config.downsample_ratio = (2, 2, 2, 2)

  config.bfloat16 = False

  config.resize_to_shape = (240, 128)
  config.num_blocks = 6
  config.dropout_rate = 0.5
  config.noise_embed_dim = 256
  config.padding = "LONLAT"
  config.use_attention = True
  config.use_position_encoding = True
  config.num_heads = 128
  config.ema_decay = 0.99
  config.final_act_fun = lambda x: x
  config.use_skips = True
  config.use_weight_global_skip = False
  config.use_local = False
  config.input_shapes = ((1, 240, 121, 4), (1, 240, 121, 4))
  config.same_dimension = True
  config.min_time = 1e-4
  config.max_time = 1 - 1e-4
  config.normalize_qk = True

  return config
