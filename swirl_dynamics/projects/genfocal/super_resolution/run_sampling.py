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

r"""Script to run inference.

Run with:
```
python run_sampling.py \
    --config=configs/conus_sample.yaml
```
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
import numpy as np
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects.genfocal.super_resolution import data
from swirl_dynamics.projects.genfocal.super_resolution import sampling
from swirl_dynamics.projects.genfocal.super_resolution import training
from swirl_dynamics.projects.genfocal.super_resolution.configs import schema as cfg
import xarray as xr
import yaml

filesys = epath.backend.tf_backend

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", None, "Path to the YAML config file containing inference args."
)
flags.mark_flags_as_required("config")


def _to_dataset_variables(
    variables: Sequence[str],
) -> list[data.DatasetVariable]:
  return [
      data.DatasetVariable(name=var, indexers=None, rename=var)
      for var in variables
  ]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config_path = FLAGS.config
  with filesys.open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

  region_cfg = cfg.RegionConfig(**config_data["region"])
  model_cfg = cfg.ModelConfig(**config_data["model"])
  sampler_cfg = cfg.SamplerConfig(**config_data["sampler"])
  inference_cfg = cfg.InferenceConfig(**config_data["inference"])

  logging.info("Region config: %s", region_cfg)
  logging.info("Inference config: %s", inference_cfg)
  lon_coords = np.arange(
      region_cfg.longitude_start,
      region_cfg.longitude_end,
      region_cfg.output_resolution_degrees,
  )
  lat_coords = np.arange(
      region_cfg.latitude_start,
      region_cfg.latitude_end,
      region_cfg.output_resolution_degrees,
  )
  end_date = np.datetime64(inference_cfg.start_date) + np.timedelta64(
      inference_cfg.total_sample_days, "D"
  )
  cond_loader = data.InputLoader(
      date_range=(inference_cfg.start_date, end_date),
      daily_dataset_path=inference_cfg.input_file_path,
      daily_variables=_to_dataset_variables(inference_cfg.input_variables),
      daily_stats_path=inference_cfg.input_stats_path,
      dims_order=("time", "longitude", "latitude"),
  )

  logging.info("Model config: %s", model_cfg)
  backbone = dfn_lib.PreconditionedDenoiserUNet3d(
      out_channels=len(inference_cfg.output_variables),
      resize_to_shape=model_cfg.sample_resize,
      num_channels=model_cfg.num_latent_channels,
      downsample_ratio=model_cfg.downsample_ratio,
      num_blocks=model_cfg.num_blocks,
      noise_embed_dim=model_cfg.noise_embedding_dim,
      input_proj_channels=model_cfg.num_input_projection_channels,
      output_proj_channels=model_cfg.num_output_projection_channels,
      padding=model_cfg.padding,
      use_spatial_attention=model_cfg.use_spatial_attention,
      use_temporal_attention=model_cfg.use_temporal_attention,
      use_position_encoding=model_cfg.use_positional_encoding,
      num_heads=model_cfg.num_attention_heads,
      normalize_qk=model_cfg.normalize_qk,
      cond_resize_method=model_cfg.cond_resize_method,
      cond_embed_dim=model_cfg.cond_embedding_dim,
  )
  denoise_fn = training.DenoisingTrainer.inference_fn_from_state_dict(
      state=training.DenoisingModelTrainState.restore_from_orbax_ckpt(
          inference_cfg.model_checkpoint_path
      ),
      denoiser=backbone,
      use_ema=True,
  )

  logging.info("Sampler config: %s", sampler_cfg)
  scheme = dfn_lib.create_variance_exploding_scheme(
      sigma=dfn_lib.tangent_noise_schedule(
          clip_max=sampler_cfg.start_noise_level, start=-1.5, end=1.5
      ),
      data_std=1.0,  # Samples are assumed to be normalized.
  )
  tspan = dfn_lib.edm_noise_decay(
      scheme=scheme, num_steps=sampler_cfg.num_diffusion_steps
  )
  sampler = sampling.TrajectorySamplerParallel(
      start_date=inference_cfg.start_date,
      total_sample_days=inference_cfg.total_sample_days,
      variables=inference_cfg.output_variables,
      lon_coords=lon_coords,
      lat_coords=lat_coords,
      scheme=scheme,
      denoise_fn=denoise_fn,
      tspan=tspan,
      cond_loader=cond_loader,
      lowres_ds=xr.open_zarr(inference_cfg.input_file_path),
      stats_ds=xr.open_zarr(inference_cfg.output_stats_path),
      store_path=inference_cfg.output_zarr_path,
      hour_interval=model_cfg.hour_interval,
      batch_size=sampler_cfg.batch_size,
      num_model_days=model_cfg.num_days,
      num_overlap_days=sampler_cfg.num_overlap_days,
      interp_method=model_cfg.skip_interp_method,
      sampling_type=sampler_cfg.sampler_type,
  )

  logging.info("Running inference.")
  sampler.generate_and_save(sampler_cfg.seed, sampler_cfg.num_samples)

  logging.info(
      "Inference done. Results saved to: %s", inference_cfg.output_zarr_path
  )


if __name__ == "__main__":
  app.run(main)
