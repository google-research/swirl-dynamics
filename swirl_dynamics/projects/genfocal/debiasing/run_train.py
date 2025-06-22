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

r"""The main entry point for running training loops.

This script is used the train the models with different ensemble members, and
using a time-coherent chunked dataloader, which normalizes the samples using
the computing climatology, and it then normalizes the climatology using the
empiral mean and std. This normalized climatology is used as a conditioning
signal in the model.
"""

import json
from os import path as osp

from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
import optax
from orbax import checkpoint
from swirl_dynamics.projects.genfocal.debiasing import trainers
from swirl_dynamics.projects.genfocal.debiasing import utils
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
  # Flags --jax_backend_target and --jax_xla_backend are available through JAX.
  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax_xla_backend = (
        "None" if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
    )
    logging.info("Using JAX XLA backend %s", jax_xla_backend)
  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())

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
    with tf.io.gfile.GFile(
        name=osp.join(workdir, "config.json"), mode="w"
    ) as f:
      conf_json = config.to_json_best_effort()
      if isinstance(conf_json, str):  # Sometimes `.to_json()` returns string
        conf_json = json.loads(conf_json)
      json.dump(conf_json, f)
  tf.config.experimental.set_visible_devices([], "GPU")

  # Defining the optimizer.
  schedule = optax.warmup_cosine_decay_schedule(
      init_value=config.initial_lr,
      peak_value=config.peak_lr,
      warmup_steps=config.warmup_steps,
      decay_steps=config.num_train_steps,
      end_value=config.end_lr,
  )

  optimizer = optax.chain(
      optax.clip_by_global_norm(config.max_norm),
      optax.adam(
          learning_rate=schedule,
          b1=config.beta1,
      ),
  )

  # Checks the shapes of the inputs and the channels.
  utils.checks_input_shapes(config.input_shapes, config.out_channels)

  # Instantiates the dataloaders for training and evaluation in two steps:
  # 1. Get the dataloader configuration from the config file.
  # 2. Build the dataloader from the dataloader configuration.
  train_dataloader_config = utils.get_dataloader_config(config, "train")
  train_dataloader = utils.build_dataloader_from_config(train_dataloader_config)
  eval_dataloader_config = utils.get_dataloader_config(config, "eval")
  eval_dataloader = utils.build_dataloader_from_config(eval_dataloader_config)

  # Instantiates the model, in two steps:
  # 1. Get the model configuration from the config file.
  # 2. Build the model from the model configuration.
  model_config = utils.get_model_config(config)
  model = utils.build_model_from_config(model_config)

  # Defines the trainer depending if the training is distributed or not.
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

  # Loads the parameters from the checkpoint of an already trained model.
  if (trained_state_dir := config.get("trained_state_dir", None)) is not None:
    # Loads the parameters from the checkpoint of an already trained model.
    logging.info("Loading trained state from %s", trained_state_dir)
    trained_state = trainers.TrainState.restore_from_orbax_ckpt(
        f"{trained_state_dir}/checkpoints",
        step=None,
        ref_state=trainer.train_state,
    )

    # Modifies the train_state with parameters from a checkpoint.
    trainer.train_state = trainer.train_state.replace(
        params=trained_state.params,
        flax_mutables=trained_state.flax_mutables,
    )
    # Avoid having more than one checkpoint.
    del trained_state

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
          # Callback to add the number of iterations/second.
          callbacks.ProgressReport(
              num_train_steps=config.num_train_steps,
          ),
      ),
  )


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  handler = app.run
  handler(main)
