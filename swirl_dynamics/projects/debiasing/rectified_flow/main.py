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

import functools
import itertools
import json
from os import path as osp

from absl import app
from absl import flags
from absl import logging
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


_ERA5_VARIABLES = {
    "2m_temperature": None,
    "specific_humidity": {"level": 1000},
    "geopotential": {"level": [200, 500]},
    "mean_sea_level_pressure": None,
    "10m_magnitude_of_wind": None,
}
_ERA5_WIND_COMPONENTS = {}

_LENS2_MEMBER_INDEXER = {"member": "cmip6_1001_001"}
_LENS2_VARIABLE_NAMES = ("TREFHT", "QREFHT", "Z200", "Z500", "PSL", "WSPDSRFAV")

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

  assert (
      config.input_shapes[0][-1] == config.input_shapes[1][-1]
      and config.input_shapes[0][-1] == config.out_channels
  )

  # TODO: add an utility function to encapsulate all this code.

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

    if "era5_variables" in config:
      era5_variables = config.era5_variables.to_dict()
    else:
      era5_variables = _ERA5_VARIABLES

    if "era5_wind_components" in config:
      era5_wind_components = config.era5_wind_components.to_dict()
    else:
      era5_wind_components = _ERA5_WIND_COMPONENTS

    if "lens2_member_indexer" in config:
      if "ens_chunked_aligned_loader" and config.ens_chunked_aligned_loader:
        # This will be a tuple of dictionaries.
        lens2_member_indexer = config.lens2_member_indexer
      else:
        lens2_member_indexer = config.lens2_member_indexer.to_dict()
    else:
      lens2_member_indexer = _LENS2_MEMBER_INDEXER

    if "lens2_variable_names" in config:
      lens2_variable_names = config.lens2_variable_names
    else:
      lens2_variable_names = _LENS2_VARIABLE_NAMES

    if "default_loaders" in config and config.default_loaders:

      era5_loader_train = data_utils.create_default_era5_loader(
          date_range=config.data_range_train,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      lens2_loader_train = data_utils.create_default_lens2_loader(
          date_range=config.data_range_train,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      era5_loader_eval = data_utils.create_default_era5_loader(
          date_range=config.data_range_eval,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size_eval,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      lens2_loader_eval = data_utils.create_default_lens2_loader(
          date_range=config.data_range_eval,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size_eval,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

    elif "dummy_loaders" in config and config.dummy_loaders:
      # Dummy data.
      fake_batch_lens2 = {
          "x_0": jax.numpy.zeros(
              (config.batch_size,) + config.input_shapes[0][1:]
          )
      }
      fake_batch_era5 = {
          "x_1": jax.numpy.ones(
              (config.batch_size,) + config.input_shapes[1][1:]
          )
      }

      era5_loader_train = era5_loader_eval = itertools.repeat(fake_batch_era5)
      lens2_loader_train = lens2_loader_eval = itertools.repeat(
          fake_batch_lens2
      )

    elif "chunked_loaders" in config and config.chunked_loaders:
      logging.info("Using chunked loaders.")
      era5_loader_train = data_utils.create_chunked_era5_loader(
          date_range=config.data_range_train,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size,
          num_chunks=config.num_chunks,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      lens2_loader_train = data_utils.create_chunked_lens2_loader(
          date_range=config.data_range_train,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size,
          member_indexer=lens2_member_indexer,
          variable_names=lens2_variable_names,
          num_chunks=config.num_chunks,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      era5_loader_eval = data_utils.create_chunked_era5_loader(
          date_range=config.data_range_eval,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size_eval,
          num_chunks=config.num_chunks,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      lens2_loader_eval = data_utils.create_chunked_lens2_loader(
          date_range=config.data_range_eval,
          shuffle=config.shuffle,
          seed=config.seed,
          member_indexer=lens2_member_indexer,
          variable_names=lens2_variable_names,
          batch_size=config.batch_size_eval,
          num_chunks=config.num_chunks,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

    elif "chunked_aligned_loader" and config.chunked_aligned_loader:
      logging.info("Using chunked aligned loaders.")

      lens2_era5_loader_train = data_utils.create_lens2_era5_loader_chunked(
          date_range=config.data_range_train,
          shuffle=config.shuffle,
          input_member_indexer=lens2_member_indexer,
          input_variable_names=lens2_variable_names,
          output_variables=era5_variables,
          output_wind_components=era5_wind_components,
          seed=config.seed,
          batch_size=config.batch_size_eval,
          drop_remainder=True,
          worker_count=config.num_workers,
      )
      lens2_era5_loader_eval = data_utils.create_lens2_era5_loader_chunked(
          date_range=config.data_range_eval,
          shuffle=config.shuffle,
          input_member_indexer=lens2_member_indexer,
          input_variable_names=lens2_variable_names,
          output_variables=era5_variables,
          output_wind_components=era5_wind_components,
          seed=config.seed,
          batch_size=config.batch_size_eval,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

    elif "ens_chunked_aligned_loader" and config.ens_chunked_aligned_loader:
      logging.info("Using ensemble chunked aligned loaders.")

      lens2_era5_loader_train = (
          data_utils.create_ensemble_lens2_era5_loader_chunked(
              date_range=config.data_range_train,
              shuffle=config.shuffle,
              input_member_indexer=lens2_member_indexer,
              input_variable_names=lens2_variable_names,
              output_variables=era5_variables,
              output_wind_components=era5_wind_components,
              seed=config.seed,
              batch_size=config.batch_size_eval,
              drop_remainder=True,
              random_local_shuffle=config.random_local_shuffle,
              batch_ot_shuffle=config.batch_ot_shuffle,
              worker_count=config.num_workers,
              num_chunks=config.num_chunks,
          )
      )
      lens2_era5_loader_eval = (
          data_utils.create_ensemble_lens2_era5_loader_chunked(
              date_range=config.data_range_eval,
              shuffle=config.shuffle,
              input_member_indexer=lens2_member_indexer,
              input_variable_names=lens2_variable_names,
              output_variables=era5_variables,
              output_wind_components=era5_wind_components,
              seed=config.seed,
              batch_size=config.batch_size_eval,
              drop_remainder=True,
              random_local_shuffle=config.random_local_shuffle,
              batch_ot_shuffle=config.batch_ot_shuffle,
              worker_count=config.num_workers,
              num_chunks=config.num_chunks,
          )
      )
    else:
      era5_loader_train = data_utils.create_era5_loader(
          date_range=config.data_range_train,
          shuffle=config.shuffle,
          variables=era5_variables,
          wind_components=era5_wind_components,
          seed=config.seed,
          batch_size=config.batch_size,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      lens2_loader_train = data_utils.create_lens2_loader(
          date_range=config.data_range_train,
          shuffle=config.shuffle,
          seed=config.seed,
          member_indexer=lens2_member_indexer,
          variable_names=lens2_variable_names,
          batch_size=config.batch_size,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      era5_loader_eval = data_utils.create_era5_loader(
          date_range=config.data_range_eval,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size_eval,
          variables=era5_variables,
          wind_components=era5_wind_components,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

      lens2_loader_eval = data_utils.create_lens2_loader(
          date_range=config.data_range_eval,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size_eval,
          member_indexer=lens2_member_indexer,
          variable_names=lens2_variable_names,
          drop_remainder=True,
          worker_count=config.num_workers,
      )

    # Creating the DataLoaders.
    if "chunked_loaders" in config and config.chunked_loaders:

      # Then create the mixed dataloaders here.
      train_dataloader = data_utils.DualChunkedLens2Era5Dataset(
          era5_loader=era5_loader_train, lens2_loader=lens2_loader_train  # pylint: disable=undefined-variable
      )

      eval_dataloader = data_utils.DualChunkedLens2Era5Dataset(
          era5_loader=era5_loader_eval, lens2_loader=lens2_loader_eval  # pylint: disable=undefined-variable
      )

    elif "date_aligned" in config and config.date_aligned:
      logging.info("Using date aligned loaders.")
      train_dataloader = data_utils.create_lens2_era5_loader(
          date_range=config.data_range_train,
          output_variables=era5_variables,
          output_wind_components=era5_wind_components,
          input_member_indexer=lens2_member_indexer,
          input_variable_names=lens2_variable_names,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size,
          drop_remainder=True,
          worker_count=config.num_workers,
      )
      eval_dataloader = data_utils.create_lens2_era5_loader(
          date_range=config.data_range_eval,
          output_variables=era5_variables,
          output_wind_components=era5_wind_components,
          input_member_indexer=lens2_member_indexer,
          input_variable_names=lens2_variable_names,
          shuffle=config.shuffle,
          seed=config.seed,
          batch_size=config.batch_size_eval,
          drop_remainder=True,
          worker_count=config.num_workers,
      )
    elif ("chunked_aligned_loader" and config.chunked_aligned_loader) or (
        "ens_chunked_aligned_loader" and config.ens_chunked_aligned_loader
    ):
      logging.info("Using chunked aligned loaders.")
      train_dataloader = data_utils.AlignedChunkedLens2Era5Dataset(
          loader=lens2_era5_loader_train  # pylint: disable=undefined-variable
      )
      eval_dataloader = data_utils.AlignedChunkedLens2Era5Dataset(
          loader=lens2_era5_loader_eval  # pylint: disable=undefined-variable
      )
    else:
      # Then create the mixed dataloaders here.
      train_dataloader = data_utils.DualLens2Era5Dataset(
          era5_loader=era5_loader_train, lens2_loader=lens2_loader_train  # pylint: disable=undefined-variable
      )

      eval_dataloader = data_utils.DualLens2Era5Dataset(
          era5_loader=era5_loader_eval, lens2_loader=lens2_loader_eval  # pylint: disable=undefined-variable
      )

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
      normalize_qk=config.normalize_qk,
  )

  if "time_sampler" in config and config.time_sampler == "lognorm":
    time_sampler = models.lognormal_sampler()
  else:
    time_sampler = functools.partial(
        jax.random.uniform, dtype=jax.numpy.float32
    )

  model = models.ReFlowModel(
      # TODO: clean this part.
      input_shape=(
          config.input_shapes[0][1],
          config.input_shapes[0][2],
          config.input_shapes[0][3],
      ),  # This must agree with the expected sample shape.
      flow_model=flow_model,
      time_sampling=time_sampler,
      min_train_time=config.min_time,  # It should be close to 0.
      max_train_time=config.max_time,  # It should be close to 1.
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
          # Callback to add the number of iterations/second.
          callbacks.ProgressReport(
              num_train_steps=config.num_train_steps,
          ),
          # TODO add a plot callback.
      ),
  )


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  handler = app.run
  handler(main)
