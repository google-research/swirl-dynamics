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
from swirl_dynamics.projects.debiasing.rectified_flow import data_utils
from swirl_dynamics.projects.debiasing.rectified_flow import models
from swirl_dynamics.projects.debiasing.rectified_flow import trainers
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
      init_value=config.initial_lr,
      peak_value=config.peak_lr,
      warmup_steps=config.warmup_steps,
      decay_steps=config.num_train_steps,
      end_value=config.end_lr,
  )

  optimizer = optax.chain(
      # optax.clip(config.clip),
      optax.clip_by_global_norm(config.max_norm),
      optax.adam(
          learning_rate=schedule,
          b1=config.beta1,
      ),
  )

  assert (
      config.input_shapes[0][-1] == config.input_shapes[1][-1]
      and config.input_shapes[0][-1] == config.out_channels
  )

  if config.tf_grain_hdf5:

    train_dataloader = data_utils.UnpairedDataLoader(
        batch_size=config.batch_size,
        dataset_path_a=config.dataset_path_u_lf,
        dataset_path_b=config.dataset_path_u_hf,
        seed=config.seed,
        split="train",
        spatial_downsample_factor_a=config.spatial_downsample_factor[0],
        spatial_downsample_factor_b=config.spatial_downsample_factor[1],
        normalize=config.normalize,
        tf_lookup_batch_size=config.tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=config.tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=config.tf_interleaved_shuffle,
    )

    eval_dataloader = data_utils.UnpairedDataLoader(
        batch_size=config.batch_size,
        dataset_path_a=config.dataset_path_u_lf,
        dataset_path_b=config.dataset_path_u_hf,
        seed=config.seed,
        split="eval",
        spatial_downsample_factor_a=config.spatial_downsample_factor[0],
        spatial_downsample_factor_b=config.spatial_downsample_factor[1],
        normalize=config.normalize,
        tf_lookup_batch_size=config.tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=config.tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=config.tf_interleaved_shuffle,
    )
  elif config.pygrain_zarr:

    era5_loader_train = data_utils.create_era5_loader(
        date_range=config.data_range_train,
        shuffle=config.shuffle,
        seed=config.seed,
        batch_size=config.batch_size,
        filter_extremes=config.filter_extremes,
        extreme_norm=config.extreme_norm,
        drop_remainder=True,
        worker_count=0,)

    lens2_loader_train = data_utils.create_lens2_loader(
        date_range=config.data_range_train,
        shuffle=config.shuffle,
        seed=config.seed,
        batch_size=config.batch_size,
        drop_remainder=True,
        interp_shapes=config.interp_shapes,
        worker_count=0,)

    train_dataloader = data_utils.DualLens2Era5Dataset(era5_loader_train,
                                                       lens2_loader_train)

    era5_loader_eval = data_utils.create_era5_loader(
        date_range=config.data_range_eval,
        shuffle=config.shuffle,
        seed=config.seed,
        batch_size=config.batch_size_eval,
        drop_remainder=True,
        worker_count=0,)

    lens2_loader_eval = data_utils.create_lens2_loader(
        date_range=config.data_range_eval,
        shuffle=config.shuffle,
        seed=config.seed,
        batch_size=config.batch_size_eval,
        drop_remainder=True,
        interp_shapes=config.interp_shapes,
        worker_count=0,)

    eval_dataloader = data_utils.DualLens2Era5Dataset(era5_loader_eval,
                                                      lens2_loader_eval)
  else:  # to avoid the linter to complain.
    train_dataloader = None
    eval_dataloader = None

  # Setting up the neural network for the flow model.
  flow_model = models.RescaledUnet(
      out_channels=config.out_channels,
      num_channels=config.num_channels,
      downsample_ratio=config.downsample_ratio,
      num_blocks=config.num_blocks,
      noise_embed_dim=config.noise_embed_dim,
      padding=config.padding,
      dropout_rate=config.dropout_rate,
      use_attention=config.use_attention,
      resize_to_shape=config.resize_to_shape,
      use_position_encoding=config.use_position_encoding,
      num_heads=config.num_heads,
  )

  model = models.ReFlowModel(
      # TODO: clean this part.
      input_shape=(
          config.input_shapes[0][1],
          config.input_shapes[0][2],
          config.input_shapes[0][3],
      ),  # This must agree with the expected sample shape.
      flow_model=flow_model,
      min_eval_time_lvl=config.min_time,  # This should be close to 0.
      max_eval_time_lvl=config.max_time  # It should be close to 1.
  )

  # Defining the trainer.
  if config.distributed:
    trainer = trainers.DistributedReFlowTrainer(
        model=model,
        rng=jax.random.PRNGKey(config.seed),
        optimizer=optimizer,
        ema_decay=config.ema_decay,
    )
  else:
    trainer = trainers.ReFlowTrainer(
        model=model,
        rng=jax.random.PRNGKey(config.seed),
        optimizer=optimizer,
        ema_decay=config.ema_decay,
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
      total_train_steps=config.num_train_steps,
      metric_aggregation_steps=config.metric_aggregation_steps,  # 30
      eval_dataloader=eval_dataloader,
      eval_every_steps=config.eval_every_steps,
      num_batches_per_eval=config.num_batches_per_eval,
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
