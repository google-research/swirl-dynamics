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

r"""Script to run training.

Run with:
```
python run_training.py \
    --config=configs/conus_train.yaml
```
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from etils import epath
import jax
import optax
from orbax import checkpoint
from swirl_dynamics import templates
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects.genfocal.super_resolution import data
from swirl_dynamics.projects.genfocal.super_resolution import training
from swirl_dynamics.projects.genfocal.super_resolution.configs import schema as cfg
import yaml

filesys = epath.backend.tf_backend

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", None, "Path to the YAML config file containing training args."
)
flags.mark_flags_as_required("config")


def regionalize_variables(
    variables: data.DatasetVariables,
    region_cfg: cfg.RegionConfig,
    epsilon: float = 5e-2,
) -> data.DatasetVariables:
  """Adds regional indexers to variables."""
  region_indexers = {
      "longitude": slice(
          region_cfg.longitude_start - epsilon,
          region_cfg.longitude_end - epsilon,
      ),
      "latitude": slice(
          region_cfg.latitude_start - epsilon,
          region_cfg.latitude_end - epsilon,
      ),
  }
  return data.add_indexers(region_indexers, variables)


def build_dataloader(
    data_cfg: cfg.TrainingDataConfig, region_cfg: cfg.RegionConfig
) -> data.pygrain.DataLoader:
  """Builds a dataloader for training or evaluation."""
  hourly_variables = regionalize_variables(
      [
          data.DatasetVariable(name, indexer, rename)
          for name, indexer, rename in data_cfg.hourly_variables
      ],
      region_cfg=region_cfg,
  )
  daily_variables = regionalize_variables(
      [
          data.DatasetVariable(name, indexer, rename)
          for name, indexer, rename in data_cfg.daily_variables
      ],
      region_cfg=region_cfg,
  )
  return data.create_dataloader(
      date_range=data_cfg.date_range,
      hourly_dataset_path=data_cfg.hourly_dataset_path,
      hourly_variables=hourly_variables,
      hourly_downsample=data_cfg.hourly_downsample,
      num_days_per_example=data_cfg.num_days_per_example,
      daily_dataset_path=data_cfg.daily_dataset_path,
      daily_variables=daily_variables,
      daily_stats_path=data_cfg.daily_stats_path,
      shuffle=data_cfg.shuffle,
      seed=data_cfg.rng_seed,
      batch_size=data_cfg.batch_size,
      worker_count=data_cfg.worker_count,
      cond_maskout_prob=data_cfg.cond_maskout_prob,
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config_path = FLAGS.config
  with filesys.open(config_path, "r") as f:
    config_data = yaml.safe_load(f)

  region_cfg = cfg.RegionConfig(**config_data["region"])
  train_data_cfg = cfg.TrainingDataConfig(**config_data["train_data"])
  eval_data_cfg = cfg.TrainingDataConfig(**config_data["eval_data"])
  model_cfg = cfg.ModelConfig(**config_data["model"])
  training_cfg = cfg.TrainingConfig(**config_data["training"])

  logging.info("Training data config: %s", train_data_cfg)
  train_dataloader = build_dataloader(train_data_cfg, region_cfg)

  logging.info("Eval data config: %s", eval_data_cfg)
  eval_dataloader = build_dataloader(eval_data_cfg, region_cfg)

  logging.info("Model config: %s", model_cfg)
  backbone = dfn_lib.PreconditionedDenoiserUNet3d(
      out_channels=len(train_data_cfg.hourly_variables),
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
  scheme = dfn_lib.create_variance_exploding_scheme(
      sigma=dfn_lib.tangent_noise_schedule(
          clip_max=training_cfg.train_max_noise_level, start=-1.5, end=1.5
      ),
      data_std=train_data_cfg.hourly_data_std,
  )
  nt = (
      train_data_cfg.num_days_per_example
      * 24
      // train_data_cfg.hourly_downsample
  )
  lon_span = region_cfg.longitude_end - region_cfg.longitude_start
  lat_span = region_cfg.latitude_end - region_cfg.latitude_start
  model = training.DenoisingModel(
      sample_shape=(
          nt,
          int(lat_span / region_cfg.output_resolution_degrees),
          int(lon_span / region_cfg.output_resolution_degrees),
          len(train_data_cfg.hourly_variables),
      ),
      denoiser=backbone,
      noise_sampling=dfn_lib.log_uniform_sampling(
          scheme=scheme, clip_min=1e-4, uniform_grid=True
      ),
      noise_weighting=dfn_lib.edm_weighting(
          data_std=train_data_cfg.hourly_data_std
      ),
      cond_shape={
          "channel:daily_mean": (
              nt,
              int(lat_span / region_cfg.input_resolution_degrees),
              int(lon_span / region_cfg.input_resolution_degrees),
              len(train_data_cfg.daily_variables),
          )
      },
      diffusion_scheme=scheme,
      cfg_strength=training_cfg.eval_cfg_strength,
      num_sde_steps=training_cfg.eval_num_sde_steps,
      num_samples_per_condition=training_cfg.eval_num_samples_per_condition,
      num_ode_steps=training_cfg.eval_num_ode_steps,
      num_likelihood_probes=training_cfg.eval_num_likelihood_probes,
  )

  logging.info("Training config: %s", training_cfg)
  trainer = training.DenoisingTrainer(
      rng=jax.random.key(training_cfg.trainer_rng_seed),
      model=model,
      optimizer=optax.chain(
          optax.clip_by_global_norm(max_norm=training_cfg.clip_grad_norm),
          optax.adam(
              learning_rate=optax.warmup_cosine_decay_schedule(
                  init_value=training_cfg.lr_initial,
                  peak_value=training_cfg.lr_peak,
                  warmup_steps=training_cfg.lr_warmup_steps,
                  decay_steps=training_cfg.total_train_steps,
                  end_value=training_cfg.lr_end,
              ),
          ),
      ),
      ema_decay=training_cfg.ema_decay,
  )

  templates.run_train(
      train_dataloader=train_dataloader,
      trainer=trainer,
      workdir=training_cfg.work_dir,
      total_train_steps=training_cfg.total_train_steps,
      metric_aggregation_steps=50,
      eval_dataloader=eval_dataloader,
      eval_every_steps=training_cfg.eval_every_n_steps,
      num_batches_per_eval=training_cfg.eval_num_batches_per_step,
      callbacks=[
          templates.ProgressReport(
              num_train_steps=training_cfg.total_train_steps
          ),
          templates.TrainStateCheckpoint(
              base_dir=training_cfg.work_dir,
              options=checkpoint.CheckpointManagerOptions(
                  best_fn=lambda metrics: metrics.get("eval_mean_crps", 1e8),
                  best_mode="min",
                  save_interval_steps=training_cfg.checkpoint_every_n_steps,
                  max_to_keep=training_cfg.checkpoint_max_to_keep,
              ),
          ),
      ],
  )


if __name__ == "__main__":
  app.run(main)
