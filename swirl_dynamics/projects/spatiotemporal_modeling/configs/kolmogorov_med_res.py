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

r"""Config file for ViViT Denoiser.


"""

import ml_collections
# pylint: disable=line-too-long
DATA_PATH = '/datasets/hdf5/pde/2d/ns/attractor_spectral_grid_256_spatial_downsample_4_dt_0.001_v0_3_warmup_40.0_t_final_200.0_nu_0.001_n_samples_2000_ntraj_train_256_ntraj_eval_32_ntraj_test_32_drag_0.1_wave_number_4_random_seeds_combined_4.hdf5'
# pylint: enable=line-too-long


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()

  # Model.
  # TODO: Undo all the nested dictionaries.
  config.model_name = 'ViViT Denoiser'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = 576
  config.spatial_downsample_factor = 1

  config.model.num_heads = 18
  config.model.mlp_dim = 512
  config.model.num_layers = 6
  config.model.dropout_rate = 0.3
  config.model_dtype_str = 'float32'
  config.model.noise_embed_dim = 256
  config.model.diffusion_scheme = 'variance_exploding'

  config.save_interval_steps = 1000
  config.max_checkpoints_to_keep = 10

  # TODO: create custom data structures.
  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  # pylint: disable=line-too-long
  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'
  # pylint: enable=line-too-long
  config.model.positional_embedding = 'sinusoidal_3d'  # 'sinusoidal_3d'

  # TODO: patches doesn't need to be a dictionary.
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = (4, 4, 4)  # (time, height, width)

  config.model.attention_config = ml_collections.ConfigDict()
  # config.model.attention_config.type = 'factorized_encoder'
  config.model.attention_config.type = 'factorized_self_attention_block'
  config.model.attention_config.attention_order = 'time_space'
  config.model.attention_config.attention_kernel_init_method = 'xavier'

  config.data = ml_collections.ConfigDict()
  config.data.file_path_data = DATA_PATH
  config.data.num_time_steps = 32
  config.data.time_stride = 2
  config.data.batch_size = 8
  config.data.normalize = True
  config.data.random_seed = 1
  config.data.tf_lookup_batch_size = 32
  config.data.std = 1.0
  config.data.space_shape = (64, 64, 1)

  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.num_train_steps = 1000000
  config.optimizer.initial_lr = 0.0
  config.optimizer.peak_lr = 3e-4
  config.optimizer.warmup_steps = 50000
  config.optimizer.end_lr = 1e-6
  config.optimizer.ema_decay = 0.999
  config.optimizer.ckpt_interval = 1000
  config.optimizer.max_ckpt_to_keep = 5
  config.optimizer.clip_min = 1e-4
  config.optimizer.metric_aggreration_steps = 50
  config.optimizer.eval_every_steps = 1000
  config.optimizer.num_batches_per_eval = 8
  config.optimizer.clip = 1.
  config.optimizer.beta1 = 0.99

  return config

