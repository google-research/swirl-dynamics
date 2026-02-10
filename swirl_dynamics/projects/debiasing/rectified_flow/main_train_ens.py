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

r"""The main entry point for running training loops.

This script is used the train the models with different ensemble members.
"""

import functools
import json
from os import path as osp

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import optax
from orbax import checkpoint
from swirl_dynamics.lib.diffusion import unets
from swirl_dynamics.projects.debiasing.rectified_flow import data_utils
from swirl_dynamics.projects.debiasing.rectified_flow import dataloaders
from swirl_dynamics.projects.debiasing.rectified_flow import models
from swirl_dynamics.projects.debiasing.rectified_flow import trainers
from swirl_dynamics.templates import callbacks
from swirl_dynamics.templates import train
import tensorflow as tf


_ERA5_VARIABLES = {
    "2m_temperature": None,
    "specific_humidity": {"level": 1000},
    "mean_sea_level_pressure": None,
    "10m_magnitude_of_wind": None,
}

# For the training of ensemble data, the indexer is a tuple of dictionaries.
# For evaluation and inference is a tuple of strings.
_LENS2_MEMBER_INDEXER = (
    {"member": "cmip6_1001_001"},
    {"member": "cmip6_1021_002"},
    {"member": "cmip6_1041_003"},
    {"member": "cmip6_1061_004"},
    {"member": "cmip6_1081_005"},
    {"member": "cmip6_1101_006"},
    {"member": "cmip6_1121_007"},
    {"member": "cmip6_1231_001"},
    {"member": "cmip6_1231_002"},
    {"member": "cmip6_1231_003"},
    {"member": "cmip6_1231_004"},
    {"member": "cmip6_1231_005"},
    {"member": "cmip6_1231_006"},
    {"member": "cmip6_1231_007"},
    {"member": "cmip6_1231_008"},
)
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "PSL", "WSPDSRFAV")

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

  # Defining experiments through the config file.
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

  if (
      config.input_shapes[0][-1] != config.input_shapes[1][-1]
      or config.input_shapes[0][-1] != config.out_channels
  ):
    raise ValueError(
        "The number of channels in the input and output must be the same."
    )

  # This is to avoid the default behavior of the ConfigDict, which converts to
  # ConfigDict any nested dictionaries.
  era5_variables = config.get("era5_variables", default=_ERA5_VARIABLES)
  if isinstance(era5_variables, ml_collections.ConfigDict):
    era5_variables = era5_variables.to_dict()

  lens2_member_indexer = config.get(
      "lens2_member_indexer", default=_LENS2_MEMBER_INDEXER
  )
  if isinstance(lens2_member_indexer, ml_collections.ConfigDict):
    lens2_member_indexer = lens2_member_indexer.to_dict()

  lens2_variable_names = config.get(
      "lens2_variable_names", default=_LENS2_VARIABLE_NAMES
  )

  if jax.process_index() == 0:
    print("ERA5 variables", flush=True)
    print(era5_variables, flush=True)
    print("LENS2 variable names", flush=True)
    print(lens2_variable_names, flush=True)

  if config.get("climatological_data_loader", default=False):

    logging.info("Using climatological data loader")
    train_dataloader = (
        dataloaders.create_ensemble_lens2_era5_loader_with_climatology(
            date_range=config.data_range_train,
            batch_size=config.batch_size,
            shuffle=True,
            worker_count=config.num_workers,
            input_dataset_path=config.lens2_dataset_path,
            input_climatology=config.lens2_stats_path,
            input_mean_stats_path=config.lens2_mean_stats_path,
            input_std_stats_path=config.lens2_std_stats_path,
            input_variable_names=lens2_variable_names,
            input_member_indexer=lens2_member_indexer,
            output_variables=era5_variables,
            output_dataset_path=config.era5_dataset_path,
            output_climatology=config.era5_stats_path,
        )
    )
    eval_dataloader = (
        dataloaders.create_ensemble_lens2_era5_loader_with_climatology(
            date_range=config.data_range_eval,
            batch_size=config.batch_size_eval,
            shuffle=True,
            worker_count=config.num_workers,
            input_dataset_path=config.lens2_dataset_path,
            input_climatology=config.lens2_stats_path,
            input_mean_stats_path=config.lens2_mean_stats_path,
            input_std_stats_path=config.lens2_std_stats_path,
            input_variable_names=lens2_variable_names,
            input_member_indexer=lens2_member_indexer,
            output_variables=era5_variables,
            output_dataset_path=config.era5_dataset_path,
            output_climatology=config.era5_stats_path,
        )
    )

  elif config.get("climatological_chunked_data_loader", default=False):
    print("Using climatological chunked data loader", flush=True)

    # Default behavior is non-time-coherent.
    if not config.get("time_coherent", default=False):
      logging.info("Using non-time-coherent data loader")
      # Defines the dataloaders directly.
      train_dataloader = dataloaders.create_ensemble_lens2_era5_chunked_loader_with_climatology(
          date_range=config.data_range_train,
          batch_size=config.batch_size,
          chunk_size=config.chunk_size,
          shuffle=True,
          worker_count=config.num_workers,
          input_dataset_path=config.lens2_dataset_path,
          input_climatology=config.lens2_stats_path,
          input_mean_stats_path=config.lens2_mean_stats_path,
          input_std_stats_path=config.lens2_std_stats_path,
          input_variable_names=lens2_variable_names,
          input_member_indexer=lens2_member_indexer,
          output_variables=era5_variables,
          output_dataset_path=config.era5_dataset_path,
          output_climatology=config.era5_stats_path,
      )
      eval_dataloader = dataloaders.create_ensemble_lens2_era5_chunked_loader_with_climatology(
          date_range=config.data_range_eval,
          batch_size=config.batch_size_eval,
          chunk_size=config.chunk_size,
          shuffle=True,
          worker_count=config.num_workers,
          input_dataset_path=config.lens2_dataset_path,
          input_climatology=config.lens2_stats_path,
          input_mean_stats_path=config.lens2_mean_stats_path,
          input_std_stats_path=config.lens2_std_stats_path,
          input_variable_names=lens2_variable_names,
          input_member_indexer=lens2_member_indexer,
          output_variables=era5_variables,
          output_dataset_path=config.era5_dataset_path,
          output_climatology=config.era5_stats_path,
      )

    else:
      logging.info("Using time-coherent data loader.")
      if config.get("use_3d_model", default=False):
        logging.info("Using 3D dataloader.")
        time_to_channel = False
      else:
        logging.info("Using 2D daloader.")
        time_to_channel = True

      train_dataloader = dataloaders.create_ensemble_lens2_era5_time_chunked_loader_with_climatology(
          date_range=config.data_range_train,
          batch_size=config.batch_size,
          chunk_size=config.chunk_size,
          shuffle=True,
          worker_count=config.num_workers,
          input_dataset_path=config.lens2_dataset_path,
          input_climatology=config.lens2_stats_path,
          input_mean_stats_path=config.lens2_mean_stats_path,
          input_std_stats_path=config.lens2_std_stats_path,
          input_variable_names=lens2_variable_names,
          input_member_indexer=lens2_member_indexer,
          output_variables=era5_variables,
          time_batch_size=config.time_batch_size,
          output_dataset_path=config.era5_dataset_path,
          output_climatology=config.era5_stats_path,
          time_to_channel=time_to_channel,
      )
      eval_dataloader = dataloaders.create_ensemble_lens2_era5_time_chunked_loader_with_climatology(
          date_range=config.data_range_eval,
          batch_size=config.batch_size_eval,
          chunk_size=config.chunk_size,
          shuffle=True,
          worker_count=config.num_workers,
          input_dataset_path=config.lens2_dataset_path,
          input_climatology=config.lens2_stats_path,
          input_mean_stats_path=config.lens2_mean_stats_path,
          input_std_stats_path=config.lens2_std_stats_path,
          input_variable_names=lens2_variable_names,
          input_member_indexer=lens2_member_indexer,
          output_variables=era5_variables,
          time_batch_size=config.time_batch_size,
          output_dataset_path=config.era5_dataset_path,
          output_climatology=config.era5_stats_path,
          time_to_channel=time_to_channel,
      )

  else:
    norm_stats_loader = config.get("norm_stats_loader", default=True)
    logging.info("Using normalized stats")
    train_dataloader = data_utils.create_ensemble_lens2_era5_loader_chunked_with_normalized_stats(
        date_range=config.data_range_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_chunks=config.num_chunks,
        worker_count=config.num_workers,
        random_local_shuffle=config.random_local_shuffle,
        batch_ot_shuffle=config.batch_ot_shuffle,
        input_dataset_path=config.lens2_dataset_path,
        input_stats_path=config.lens2_stats_path,
        input_mean_stats_path=config.lens2_mean_stats_path,
        input_std_stats_path=config.lens2_std_stats_path,
        input_variable_names=lens2_variable_names,
        input_member_indexer=lens2_member_indexer,
        output_variables=era5_variables,
        output_dataset_path=config.era5_dataset_path,
        output_stats_path=config.era5_stats_path,
        normalize_stats=norm_stats_loader,
        overlapping_chunks=config.overlapping_chunks,
    )
    eval_dataloader = data_utils.create_ensemble_lens2_era5_loader_chunked_with_normalized_stats(
        date_range=config.data_range_eval,
        batch_size=config.batch_size_eval,
        shuffle=True,
        worker_count=config.num_workers,
        random_local_shuffle=config.random_local_shuffle,
        batch_ot_shuffle=config.batch_ot_shuffle,
        input_dataset_path=config.lens2_dataset_path,
        input_stats_path=config.lens2_stats_path,
        input_mean_stats_path=config.lens2_mean_stats_path,
        input_std_stats_path=config.lens2_std_stats_path,
        input_variable_names=lens2_variable_names,
        input_member_indexer=lens2_member_indexer,
        output_variables=era5_variables,
        output_dataset_path=config.era5_dataset_path,
        output_stats_path=config.era5_stats_path,
        normalize_stats=norm_stats_loader,
        overlapping_chunks=config.overlapping_chunks,
    )

  if "bfloat16" in config and config.bfloat16:
    dtype = jax.numpy.bfloat16
    param_dtype = jax.numpy.bfloat16
  else:
    dtype = jax.numpy.float32
    param_dtype = jax.numpy.float32

  # Adds the conditional embedding for the FILM layer.
  if config.get("conditional_embedding", default=False):
    logging.info("Using conditional embedding")
    cond_embed_fn = unets.EmbConvMerge
  else:
    cond_embed_fn = None

  if config.get("use_3d_model", default=False):
    logging.info("Using 3D model")
    if config.get("use_subtracted_model", default=False):
      logging.info("Using subtracted 3D model")
      flow_model = models.RescaledSubstractedUnet3d(
          out_channels=config.out_channels,
          num_channels=config.num_channels,
          downsample_ratio=config.downsample_ratio,
          num_blocks=config.num_blocks,
          noise_embed_dim=config.noise_embed_dim,
          padding=config.padding,
          dropout_rate=config.dropout_rate,
          use_spatial_attention=config.use_spatial_attention,
          use_temporal_attention=config.use_temporal_attention,
          resize_to_shape=config.resize_to_shape,
          use_position_encoding=config.use_position_encoding,
          num_heads=config.num_heads,
          normalize_qk=config.normalize_qk,
          ffn_type=config.get("ffn_type", default="dense"),
      )
    else:
      logging.info("Using vanilla 3D model")
      flow_model = models.RescaledUnet3d(
          out_channels=config.out_channels,
          num_channels=config.num_channels,
          downsample_ratio=config.downsample_ratio,
          num_blocks=config.num_blocks,
          noise_embed_dim=config.noise_embed_dim,
          padding=config.padding,
          dropout_rate=config.dropout_rate,
          use_spatial_attention=config.use_spatial_attention,
          use_temporal_attention=config.use_temporal_attention,
          resize_to_shape=config.resize_to_shape,
          use_position_encoding=config.use_position_encoding,
          num_heads=config.num_heads,
          normalize_qk=config.normalize_qk,
          ffn_type=config.get("ffn_type", default="dense"),
      )

  else:
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
        normalize_qk=config.normalize_qk,
        cond_embed_fn=cond_embed_fn,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  if (sampler_type := config.get("time_sampler", None)) == "lognorm":
    time_sampler = models.lognormal_sampler()
  elif sampler_type == "uniform":
    time_sampler = functools.partial(
        jax.random.uniform, dtype=jax.numpy.float32
    )
  else:
    raise ValueError(f"Unknown time sampler: {sampler_type}")

  # Adds a latitude weighted norm for the loss function plus a regularization
  # factor to account for the difference in the representation geometry.
  if config.get("weighted_norm", default=False):
    lat = jnp.linspace(-90.0, 90.0, config.input_shapes[0][2])
    reg_factor = config.get("reg_factor", default=0.05)
    # Reshapes to the correct broadcast shape.
    weighted_norm = (
        jnp.cos(jnp.deg2rad(lat)).reshape((1, 1, -1, 1)) + reg_factor
    )
    weighted_norm = jnp.broadcast_to(
        weighted_norm, (1,) + config.input_shapes[0][1:]
    )
  else:
    weighted_norm = None

  input_shape = config.input_shapes[0][1:]
  cond_shape = {
      "channel:mean": config.input_shapes[0][1:],
      "channel:std": config.input_shapes[0][1:],
  }

  # Checks the shapes of the input and conditioning.
  if config.get("use_3d_model", default=False):
    if len(input_shape) != 4:
      raise ValueError("Input shape must be 4D for 3D model.")
    if len(cond_shape["channel:mean"]) != 4:
      raise ValueError("Conditional shape of the mean must be 4D for 3D model.")
    if len(cond_shape["channel:std"]) != 4:
      raise ValueError("Conditional shape of the std must be 4D for 3D model.")
  else:
    if len(input_shape) != 3:
      raise ValueError("Input shape must be 3D for 2D model.")
    if len(cond_shape["channel:mean"]) != 3:
      raise ValueError("Conditional shape of the mean must be 3D for 2D model.")
    if len(cond_shape["channel:std"]) != 3:
      raise ValueError("Conditional shape of the std must be 3D for 2D model.")

  model = models.ConditionalReFlowModel(
      input_shape=input_shape,  # This must agree with the sample shape.
      cond_shape=cond_shape,
      flow_model=flow_model,
      time_sampling=time_sampler,
      min_train_time=config.min_time,  # It should be close to 0.
      max_train_time=config.max_time,  # It should be close to 1.
      weighted_norm=weighted_norm,
  )

  # Defines the trainer.
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
