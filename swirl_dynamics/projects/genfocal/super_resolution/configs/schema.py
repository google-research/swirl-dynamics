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

"""Config schemas."""

import dataclasses
from typing import Any, Literal


@dataclasses.dataclass(frozen=True, slots=True)
class RegionConfig:
  """Region config.

  Attributes:
    longitude_start: The starting longitude coordinate (inclusive, in [0, 360]).
    longitude_end: The end longitude coordinate (exclusive, in [0, 360]).
    latitude_start: The start latitude coordinate (inclusive, in [-90, 90]).
    latitude_end: The end latitude coordinate (exclusive, in [-90, 90]).
    input_resolution_degrees: The input spatial resolution in degrees.
    output_resolution_degrees: The output spatial resolution in degrees.
  """

  longitude_start: float
  longitude_end: float
  latitude_start: float
  latitude_end: float
  input_resolution_degrees: float
  output_resolution_degrees: float


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingDataConfig:
  """Data config for training.

  Attributes:
    hourly_dataset_path: See `data.create_dataloader`.
    hourly_variables: See `data.create_dataloader`.
    hourly_downsample: See `data.create_dataloader`.
    hourly_data_std: The standard deviation of the hourly dataset. This is a
      required for the diffusion scheme to scale the noise level and
      preconditioning coeffcients.
    daily_dataset_path: See `data.create_dataloader`.
    daily_variables: See `data.create_dataloader`.
    daily_stats_path: See `data.create_dataloader`.
    num_days_per_example: See `data.create_dataloader`.
    date_range: See `data.create_dataloader`.
    batch_size: See `data.create_dataloader`.
    shuffle: See `data.create_dataloader`.
    worker_count: See `data.create_dataloader`.
    rng_seed: See `data.create_dataloader`.
    cond_maskout_prob: See `data.create_dataloader`.
  """

  hourly_dataset_path: str
  hourly_variables: list[tuple[str, dict[str, Any] | None, str]]
  hourly_downsample: int
  hourly_data_std: float
  daily_dataset_path: str
  daily_variables: list[tuple[str, dict[str, Any] | None, str]]
  daily_stats_path: str
  num_days_per_example: int
  date_range: tuple[str, str]
  batch_size: int
  shuffle: bool
  worker_count: int
  rng_seed: int
  cond_maskout_prob: float


@dataclasses.dataclass(frozen=True, slots=True)
class ModelConfig:
  """Super-resolution (denoising) model config.

  Attributes:
    num_days: The number of days modeled (i.e. single segment).
    hour_interval: The number of hours between two consecutive time slices in
      the sample.
    skip_interp_method: The method for interpolating the low-resolution input.
      The model learns to sample the residual between the target output and the
      interpolated low-resolution input.
    num_output_channels: The number of output channels, equal to the number of
      output variables.
    sample_resize: The (optional) shape to resize the samples to, such that they
      can be conveniently downsampled at integer-valued factors.
    num_latent_channels: The number of latent channels for each downsample
      block.
    downsample_ratio: The downsample ratio for each downsample block.
    num_blocks: The number of downsample/upsample blocks in the UNet backbone.
    noise_embedding_dim: The embedding dimension for the diffusion noise level.
    num_input_projection_channels: The number of input projection channels.
    num_output_projection_channels: The number of output projection channels.
    padding: The padding method for the convolution layers.
    use_spatial_attention: Whether to apply attention along spatial axes for
      each downsample block.
    use_temporal_attention: Whether to apply attention along the time axis for
      each downsample block.
    use_positional_encoding: Whether to use positional encoding for all
      attention layers.
    num_attention_heads: The number of heads in the multi-head attention layers.
    normalize_qk: Whether to normalize query and key vectors in the attention
      layers.
    cond_resize_method: The method for resizing the condition inputs. They are
      resized to match the sample spatial dimensions, projected, and then
      appended in the channel dimension to the noisy sample.
    cond_embedding_dim: The dimension to project the resized condition inputs.
  """

  num_days: int
  hour_interval: int
  skip_interp_method: str
  num_output_channels: int
  sample_resize: tuple[int, int] | None
  num_latent_channels: list[int]
  downsample_ratio: list[int]
  num_blocks: int
  noise_embedding_dim: int
  num_input_projection_channels: int
  num_output_projection_channels: int
  padding: str
  use_spatial_attention: list[bool]
  use_temporal_attention: list[bool]
  use_positional_encoding: bool
  num_attention_heads: int
  normalize_qk: bool
  cond_resize_method: str
  cond_embedding_dim: int


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingConfig:
  """Training parameters config.

  Attributes:
    work_dir: The work directory for training where training progress (metrics,
      checkpoints, etc.) will be saved.
    total_train_steps: The total number of training steps.
    checkpoint_every_n_steps: The number of steps between checkpoints.
    checkpoint_max_to_keep: The maximum number of checkpoints to keep.
    trainer_rng_seed: The random seed for the trainer.
    lr_initial: The initial learning rate (for
      `optax.warmup_cosine_decay_schedule`).
    lr_peak: The peak learning rate (for `optax.warmup_cosine_decay_schedule`).
    lr_end: The end learning rate (for `optax.warmup_cosine_decay_schedule`).
    lr_warmup_steps: The number of warmup steps (for
      `optax.warmup_cosine_decay_schedule`).
    clip_grad_norm: The gradient clipping norm (for
      `optax.clip_by_global_norm`).
    train_max_noise_level: The maximum noise level during training.
    ema_decay: The rate of exponential moving average decay for model
      parameters.
    eval_every_n_steps: The number of steps between evaluations.
    eval_num_batches_per_step: The number of batches to evaluate per step.
    eval_num_sde_steps: The number of SDE steps for sampling during evaluation.
    eval_num_samples_per_condition: The number of samples per condition to
      generate for evaluation.
    eval_num_ode_steps: The number of ODE steps for sampling during evaluation.
    eval_cfg_strength: The classifier-free guidance strength during evaluation.
    eval_num_likelihood_probes: The number of likelihood probes during
      evaluation.
  """

  work_dir: str
  total_train_steps: int
  checkpoint_every_n_steps: int
  checkpoint_max_to_keep: int
  trainer_rng_seed: int
  lr_initial: float = 0.0
  lr_peak: float = 1e-4
  lr_end: float = 1e-7
  lr_warmup_steps: int = 1000
  clip_grad_norm: float = 0.6
  train_max_noise_level: float = 80.0
  ema_decay: float = 0.9999
  eval_every_n_steps: int = 2000
  eval_num_batches_per_step: int = 1
  eval_num_sde_steps: int = 128
  eval_num_samples_per_condition: int = 1
  eval_num_ode_steps: int = 64
  eval_cfg_strength: float = 0.2
  eval_num_likelihood_probes: int = 1


@dataclasses.dataclass(frozen=True, slots=True)
class SamplerConfig:
  """Diffusion sampler config.

  Attributes:
    num_overlap_days: The number of overlap days between contiguous samples. For
      example, if there are 4 segments (each sampled by a different device) they
      will cover days 1-3, 3-5, 5-7 and 7-9 (inclusively) respectively, where
      days 3, 5 and 7 are overlapped.
    num_samples: The total number of samples to generate.
    start_noise_level: The starting noise level (i.e. maximum sigma/standard
      deviation).
    num_diffusion_steps: The number of diffusion steps when solving the ODE or
      the SDE sampler.
    cfg_strength: The classifier-free guidance strength.
    sampler_type: The diffusion sampler type ('ode' or 'sde').
    batch_size: The batch size at which the samples are generated.
    seed: A random seed for sampler initialization and stochastic processes.
  """

  num_overlap_days: int
  num_samples: int
  start_noise_level: float
  num_diffusion_steps: int
  cfg_strength: float
  sampler_type: Literal["sde", "ode"]
  batch_size: int
  seed: int


@dataclasses.dataclass(frozen=True, slots=True)
class InferenceConfig:
  """Inference config.

  Attributes:
    start_date: The start date of the trajectory to be sampled, in format
      'YYYY-MM-DD'.
    total_sample_days: The total number of days to sample. This currently has a
      deterministic relationship with the number of devices used: `num_devices *
      (num_model_days - num_overlap_days) + num_overlap_days`.
    input_variables: Input variable names.
    output_variables: Output variable names.
    input_file_path: The input data file path (.zarr).
    input_stats_path: The input stats file path (.zarr).
    output_zarr_path: The output file path (.zarr).
    output_stats_path: The output stats file path (.zarr).
    model_checkpoint_path: The checkpoint path for the trained denoiser
      backbone.
  """

  start_date: str
  total_sample_days: int
  input_variables: list[str]
  output_variables: list[str]
  input_file_path: str
  input_stats_path: str
  output_zarr_path: str
  output_stats_path: str
  model_checkpoint_path: str
