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

r"""The main entry point for running training loops."""

import json
from os import path as osp

from absl import app
from absl import flags
import jax
from ml_collections import config_flags
import optax
from orbax import checkpoint
from swirl_dynamics.lib.diffusion import diffusion
from swirl_dynamics.lib.diffusion import vivit_diffusion
from swirl_dynamics.projects.probabilistic_diffusion import models
from swirl_dynamics.projects.probabilistic_diffusion import trainers
from swirl_dynamics.projects.spatiotemporal_modeling import data_utils
from swirl_dynamics.templates import callbacks
from swirl_dynamics.templates import train
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  config = FLAGS.config
  # Dump config as json to workdir.
  workdir = FLAGS.workdir
  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)
  # Only 0-th process should write the json file to disk, in order to avoid
  # race conditions.
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(name=osp.join(workdir,
                                         "config.json"), mode="w") as f:
      conf_json = config.to_json_best_effort()
      if isinstance(conf_json, str):  # Sometimes `.to_json()` returns string
        conf_json = json.loads(conf_json)
      json.dump(conf_json, f)
  tf.config.experimental.set_visible_devices([], "GPU")

  # Defining experiments through the config file.
  schedule = optax.warmup_cosine_decay_schedule(
      init_value=config.optimizer.initial_lr,
      peak_value=config.optimizer.peak_lr,
      warmup_steps=config.optimizer.warmup_steps,
      decay_steps=config.optimizer.num_train_steps,
      end_value=config.optimizer.end_lr,
  )

  optimizer = optax.chain(
      optax.clip(config.optimizer.clip),
      optax.adam(
          learning_rate=schedule,
          b1=config.optimizer.beta1,
      ),
  )

  train_dataloader, stats = data_utils.create_loader_from_hdf5(
      num_time_steps=config.data.num_time_steps,
      time_stride=config.data.time_stride,
      batch_size=config.data.batch_size,
      spatial_downsample_factor=config.spatial_downsample_factor,
      dataset_path=config.data.file_path_data,
      seed=config.data.random_seed,
      tf_lookup_batch_size=config.data.tf_lookup_batch_size,
      split="train",
      normalize=config.data.normalize,
  )

  eval_dataloader, _ = data_utils.create_loader_from_hdf5(
      num_time_steps=config.data.num_time_steps,
      time_stride=config.data.time_stride,
      batch_size=config.data.batch_size,
      normalize_stats=stats,
      spatial_downsample_factor=config.spatial_downsample_factor,
      dataset_path=config.data.file_path_data,
      tf_lookup_batch_size=config.data.tf_lookup_batch_size,
      seed=config.data.random_seed,
      split="eval",
      normalize=config.data.normalize,
  )

  # Setting up the denoiser neural network.
  denoiser_model = vivit_diffusion.PreconditionedDenoiser(
      mlp_dim=config.model.mlp_dim,
      num_layers=config.model.num_layers,
      num_heads=config.model.num_heads,
      output_features=1,
      noise_embed_dim=config.model.noise_embed_dim,
      patches=config.model.patches,
      hidden_size=config.model.hidden_size,
      temporal_encoding_config=config.model.temporal_encoding_config,
      attention_config=config.model.attention_config,
      positional_embedding=config.model.positional_embedding,
      sigma_data=1.0,  # standard deviation of the entire dataset.
  )

  if config.model.diffusion_scheme == "variance_exploding":
    diffusion_scheme = diffusion.Diffusion.create_variance_exploding(
        sigma=diffusion.tangent_noise_schedule(
            clip_max=80.0, start=-1.5, end=1.5
        ),
        data_std=config.data.std,
    )
  elif config.model.diffusion_scheme == "variance_preserving":
    diffusion_scheme = diffusion.Diffusion.create_variance_preserving(
        sigma=diffusion.tangent_noise_schedule(),
        data_std=config.data.std,
    )
  else:
    raise ValueError(
        f"Unknown diffusion scheme: {config.model.diffusion_scheme}"
    )

  model = models.DenoisingModel(
      input_shape=(
          config.data.num_time_steps,
          config.data.space_shape[0] // config.spatial_downsample_factor,
          config.data.space_shape[1] // config.spatial_downsample_factor,
          config.data.space_shape[2],
      ),  # This must agree with the expected sample shape.
      denoiser=denoiser_model,
      noise_sampling=diffusion.log_uniform_sampling(
          diffusion_scheme,  # pylint: disable=undefined-variable
          clip_min=config.optimizer.clip_min,
          uniform_grid=True,
      ),
      noise_weighting=diffusion.edm_weighting(data_std=1.0),
  )

  # Defining the trainer.
  trainer = trainers.DenoisingTrainer(
      model=model,
      rng=jax.random.PRNGKey(888),
      optimizer=optimizer,
      # This option is to minimize the colorshift.
      ema_decay=config.optimizer.ema_decay,
  )

  # Setting up checkpointing.
  ckpt_options = checkpoint.CheckpointManagerOptions(
      save_interval_steps=config.save_interval_steps,
      max_to_keep=config.max_checkpoints_to_keep,
  )

  # Run training loop.
  train.run(
      train_dataloader=train_dataloader,
      trainer=trainer,
      workdir=workdir,
      total_train_steps=config.optimizer.num_train_steps,
      metric_aggregation_steps=config.optimizer.metric_aggregation_steps,  # 30
      eval_dataloader=eval_dataloader,
      eval_every_steps=config.optimizer.eval_every_steps,
      num_batches_per_eval=config.optimizer.num_batches_per_eval,
      callbacks=(
          # This callback saves model checkpoint periodically.
          callbacks.TrainStateCheckpoint(
              base_dir=workdir,
              options=ckpt_options,
          ),
          # TODO add a plot callback.
      ),
  )


if __name__ == "__main__":
  app.run(main)
