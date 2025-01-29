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

r"""Script to run inference.

Run with:
```
python3 run_inference.py \
    --config=yaml_configs/inference_conus_9days.yaml
```
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal

from absl import app
from absl import flags
import numpy as np
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects import probabilistic_diffusion as pdfn
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5 import sampling_parallel as sampling
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import inference as inference_pipeline
from swirl_dynamics.projects.probabilistic_diffusion.downscaling.era5.input_pipelines import utils
import xarray as xr
import yaml

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", None, "Path to the YAML config file containing inference args."
)
flags.mark_flags_as_required("config")


@dataclasses.dataclass(frozen=True)
class DownscaleConfig:
  """Downscale config.

  Attributes:
    start_date: The start date of the trajectory to be sampled, in format
      'YYYY-MM-DD'.
    total_sample_days: The total number of days to sample. This currently has a
      deterministic relationship with the number of devices used: `num_devices *
      (num_model_days - num_overlap_days) + num_overlap_days`.
    input_variables: Input variable names.
    output_variables: Output variable names.
    longitude_start: The starting longitude coordinate (inclusive, in [0, 360]).
    longitude_end: The end longitude coordinate (exclusive, in [0, 360]).
    latitude_start: The start latitude coordinate (inclusive, in [-90, 90]).
    latitude_end: The end latitude coordinate (exclusive, in [-90, 90]).
    input_file: The input file path (.zarr).
    input_resolution_degrees: The input resolution in degrees.
    input_stats_file: The input stats file path (.zarr).
    output_zarr_path: The output file path (.zarr).
    output_resolution_degrees: The output resolution in degrees.
    output_stats_file: The output stats file path (.zarr).
  """

  start_date: str
  total_sample_days: int
  input_variables: list[str]
  output_variables: list[str]
  longitude_start: float
  longitude_end: float
  latitude_start: float
  latitude_end: float
  input_file: str
  input_resolution_degrees: float
  input_stats_file: str
  output_zarr_path: str
  output_resolution_degrees: float
  output_stats_file: str


@dataclasses.dataclass(frozen=True)
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
    checkpoint: The checkpoint path for the trained denoiser backbone.
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
  checkpoint: str


@dataclasses.dataclass(frozen=True)
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
    seed: The random seed.
  """

  num_overlap_days: int
  num_samples: int
  start_noise_level: float
  num_diffusion_steps: int
  cfg_strength: float
  sampler_type: Literal["sde", "ode"]
  batch_size: int
  seed: int


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
  """Inference config."""

  downscale: DownscaleConfig
  model: ModelConfig
  sampler: SamplerConfig


def _to_dataset_variables(
    variables: Sequence[str],
) -> list[utils.DatasetVariable]:
  return [
      utils.DatasetVariable(name=var, indexers=None, rename=var)
      for var in variables
  ]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config_path = FLAGS.config
  with open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

  config = InferenceConfig(
      downscale=DownscaleConfig(**config_data["downscale"]),
      model=ModelConfig(**config_data["model"]),
      sampler=SamplerConfig(**config_data["sampler"]),
  )

  print("Downscale config:", config.downscale)
  lon_coords = np.arange(
      config.downscale.longitude_start,
      config.downscale.longitude_end,
      config.downscale.output_resolution_degrees,
  )
  lat_coords = np.arange(
      config.downscale.latitude_start,
      config.downscale.latitude_end,
      config.downscale.output_resolution_degrees,
  )
  end_date = np.datetime64(config.downscale.start_date) + np.timedelta64(
      config.downscale.total_sample_days, "D"
  )
  cond_loader = inference_pipeline.DefaultCondLoader(
      date_range=(config.downscale.start_date, end_date),
      dataset=config.downscale.input_file,
      dataset_variables=_to_dataset_variables(config.downscale.input_variables),
      stats=config.downscale.input_stats_file,
      dims_order=("time", "longitude", "latitude"),
  )

  print("Model config:", config.model)
  backbone = dfn_lib.PreconditionedDenoiserUNet3d(
      out_channels=len(config.downscale.output_variables),
      resize_to_shape=config.model.sample_resize,
      num_channels=config.model.num_latent_channels,
      downsample_ratio=config.model.downsample_ratio,
      num_blocks=config.model.num_blocks,
      noise_embed_dim=config.model.noise_embedding_dim,
      input_proj_channels=config.model.num_input_projection_channels,
      output_proj_channels=config.model.num_output_projection_channels,
      padding=config.model.padding,
      use_spatial_attention=config.model.use_spatial_attention,
      use_temporal_attention=config.model.use_temporal_attention,
      use_position_encoding=config.model.use_positional_encoding,
      num_heads=config.model.num_attention_heads,
      normalize_qk=config.model.normalize_qk,
      cond_resize_method=config.model.cond_resize_method,
      cond_embed_dim=config.model.cond_embedding_dim,
  )
  denoise_fn = pdfn.DenoisingTrainer.inference_fn_from_state_dict(
      state=pdfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
          config.model.checkpoint
      ),
      denoiser=backbone,
      use_ema=True,
  )

  print("Sampler config:", config.sampler)
  scheme = dfn_lib.create_variance_exploding_scheme(
      sigma=dfn_lib.tangent_noise_schedule(
          clip_max=config.sampler.start_noise_level, start=-1.5, end=1.5
      ),
      data_std=1.0,  # Samples are assumed to be normalized.
  )
  tspan = dfn_lib.edm_noise_decay(
      scheme=scheme, num_steps=config.sampler.num_diffusion_steps
  )
  sampler = sampling.TrajectorySamplerParallel(
      start_date=config.downscale.start_date,
      total_sample_days=config.downscale.total_sample_days,
      variables=config.downscale.output_variables,
      lon_coords=lon_coords,
      lat_coords=lat_coords,
      scheme=scheme,
      denoise_fn=denoise_fn,
      tspan=tspan,
      cond_loader=cond_loader,
      lowres_ds=xr.open_zarr(config.downscale.input_file),
      stats_ds=xr.open_zarr(config.downscale.output_stats_file),
      store_path=config.downscale.output_zarr_path,
      hour_interval=config.model.hour_interval,
      batch_size=config.sampler.batch_size,
      num_model_days=config.model.num_days,
      num_overlap_days=config.sampler.num_overlap_days,
      interp_method=config.model.skip_interp_method,
      sampling_type=config.sampler.sampler_type,
  )

  print("Running inference...")
  sampler.generate_and_save(config.sampler.seed, config.sampler.num_samples)

  print("Inference done. Results saved to: ", config.downscale.output_zarr_path)


if __name__ == "__main__":
  app.run(main)
