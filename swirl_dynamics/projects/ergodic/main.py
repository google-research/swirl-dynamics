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
# TODO: Consider enabling float64 for Lorenz63 experiment

import functools
import json
from os import path as osp
from typing import Iterable, Tuple

from absl import app
from absl import flags
import jax
from ml_collections import config_flags
import optax
from orbax import checkpoint
from swirl_dynamics.projects.ergodic import choices
from swirl_dynamics.projects.ergodic import ks_1d
from swirl_dynamics.projects.ergodic import lorenz63
from swirl_dynamics.projects.ergodic import ns_2d
from swirl_dynamics.projects.ergodic import stable_ar
from swirl_dynamics.projects.ergodic import utils
from swirl_dynamics.templates import callbacks
from swirl_dynamics.templates import train
import tensorflow as tf


Array = jax.Array
PipelinePayload = Tuple[Iterable[Array], stable_ar.StableARTrainer]

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
  # Dump config as json to workdir
  workdir = FLAGS.workdir
  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)
  # Only 0-th order process write the json file to disk.
  # This is to avoid race conditions.
  if jax.process_index() == 0:
    with tf.io.gfile.GFile(name=osp.join(workdir,
                                         "config.json"), mode="w") as f:
      conf_json = config.to_json_best_effort()
      if isinstance(conf_json, str):  # Sometimes `.to_json()` returns string
        conf_json = json.loads(conf_json)
      json.dump(conf_json, f)
  tf.config.experimental.set_visible_devices([], "GPU")
  # Setup experiment-specific params
  experiment = choices.Experiment(config.experiment)
  if experiment == choices.Experiment.L63:
    fig_callback_cls = lorenz63.Lorenz63PlotFigures
    state_dims = (3 // config.spatial_downsample_factor,)
  elif experiment == choices.Experiment.KS_1D:
    fig_callback_cls = ks_1d.KS1DPlotFigures
    state_dims = (
        512 // config.spatial_downsample_factor,
        config.num_lookback_steps,
    )

  elif experiment == choices.Experiment.NS_2D:
    fig_callback_cls = ns_2d.NS2dPlotFigures
    # TODO: This state dim is temporary for FNO data, should be 256
    state_dims = (
        64 // config.spatial_downsample_factor,
        64 // config.spatial_downsample_factor,
        config.num_lookback_steps,
    )
  else:
    raise NotImplementedError(f"Unknown experiment: {config.experiment}")

  # Slicing the decay options in the learning rate scheduler.
  if "decay_rate" in config:
    decay_rate = config.decay_rate
  else:
    decay_rate = 0.5

  if "num_steps_for_decrease_lr" in config:
    num_steps_for_decrease_lr = config.num_steps_for_decrease_lr
  else:
    num_steps_for_decrease_lr = config.train_steps_per_cycle

  if config.use_lr_scheduler:
    optimizer = optax.adam(
        learning_rate=optax.exponential_decay(
            init_value=config.lr,
            transition_steps=num_steps_for_decrease_lr,
            decay_rate=decay_rate,
            staircase=True,
        )
    )
  else:
    optimizer = optax.adam(learning_rate=config.lr)

  # Dataloaders
  if "use_tfds" in config and config.use_tfds:
    train_loader, normalize_stats = utils.create_loader_from_tfds(
        num_time_steps=config.num_time_steps,
        time_stride=config.time_stride,
        batch_size=config.batch_size,
        dataset_path=config.dataset_path,
        dataset_name=config.dataset_name,
        seed=config.seed,
        normalize=config.normalize,
        split="train",
        tf_lookup_batch_size=config.tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=config.tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=config.tf_interleaved_shuffle,
    )
    eval_loader, _ = utils.create_loader_from_tfds(
        num_time_steps=config.num_time_steps_eval,
        time_stride=config.time_stride,
        batch_size=config.batch_size_eval,
        seed=config.seed,
        dataset_path=config.dataset_path,
        dataset_name=config.dataset_name,
        normalize=config.normalize,
        split="eval",
        tf_lookup_batch_size=config.tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=config.tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=config.tf_interleaved_shuffle,
    )
  elif "use_hdf5_reshaped" in config and config.use_hdf5_reshaped:
    train_loader, normalize_stats = utils.create_loader_from_hdf5_reshaped(
        num_time_steps=config.num_time_steps,
        time_stride=config.time_stride,
        batch_size=config.batch_size,
        seed=config.seed,
        dataset_path=config.dataset_path,
        split="train",
        normalize=config.normalize,
        normalize_stats=None,
        spatial_downsample_factor=config.spatial_downsample_factor,
        tf_lookup_batch_size=config.tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=config.tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=config.tf_interleaved_shuffle,
    )
    eval_loader, _ = utils.create_loader_from_hdf5_reshaped(
        num_time_steps=-1,
        time_stride=config.time_stride,
        batch_size=-1,
        seed=config.seed,
        dataset_path=config.dataset_path,
        split="eval",
        normalize=config.normalize,
        normalize_stats=normalize_stats,
        spatial_downsample_factor=config.spatial_downsample_factor,
        tf_lookup_batch_size=config.tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=config.tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=config.tf_interleaved_shuffle,
    )

  else:
    train_loader, normalize_stats = utils.create_loader_from_hdf5(
        num_time_steps=config.num_time_steps,
        time_stride=config.time_stride,
        batch_size=config.batch_size,
        seed=config.seed,
        dataset_path=config.dataset_path,
        split="train",
        normalize=config.normalize,
        normalize_stats=None,
        spatial_downsample_factor=config.spatial_downsample_factor,
        tf_lookup_batch_size=config.tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=config.tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=config.tf_interleaved_shuffle,
    )
    eval_loader, _ = utils.create_loader_from_hdf5(
        num_time_steps=-1,
        time_stride=config.time_stride,
        batch_size=-1,
        seed=config.seed,
        dataset_path=config.dataset_path,
        split="eval",
        normalize=config.normalize,
        normalize_stats=normalize_stats,
        spatial_downsample_factor=config.spatial_downsample_factor,
        tf_lookup_batch_size=config.tf_lookup_batch_size,
        tf_lookup_num_parallel_calls=config.tf_lookup_num_parallel_calls,
        tf_interleaved_shuffle=config.tf_interleaved_shuffle,
    )

  # Model
  measure_dist_fn_raw = choices.MeasureDistance(
      config.measure_dist_type
  ).dispatch()

  # We modify the bandwith in this case.
  if config.measure_dist_type == "MMD":
    measure_dist_fn = functools.partial(
        measure_dist_fn_raw, bandwidth=config.mmd_bandwidth
    )
  else:
    measure_dist_fn = measure_dist_fn_raw

  model_config = stable_ar.StableARModelConfig(
      state_dimension=state_dims,
      dynamics_model=choices.Model(config.model).dispatch(config),
      integrator=choices.Integrator(config.integrator),
      measure_dist=measure_dist_fn,
      use_pushfwd=config.use_pushfwd,
      add_noise=config.add_noise,
      noise_level=config.noise_level,
      measure_dist_lambda=config.measure_dist_lambda,
      measure_dist_k_lambda=config.measure_dist_k_lambda,
      num_lookback_steps=config.num_lookback_steps,
      normalize_stats=normalize_stats,
      use_sobolev_norm=config.use_sobolev_norm,
      order_sobolev_norm=config.order_sobolev_norm,
  )
  model = stable_ar.StableARModel(conf=model_config)

  # Trainer
  trainer_config = stable_ar.StableARTrainerConfig(
      rollout_weighting=choices.RolloutWeighting(
          config.rollout_weighting
      ).dispatch(config),
      num_rollout_steps=config.num_rollout_steps,
      num_lookback_steps=config.num_lookback_steps,
      add_noise=config.add_noise,
      use_curriculum=config.use_curriculum,
      use_pushfwd=config.use_pushfwd,
      train_steps_per_cycle=config.train_steps_per_cycle,
      time_steps_increase_per_cycle=config.time_steps_increase_per_cycle,
  )

  # Using the distributed trainer.
  if "use_distributed" in config and config.use_distributed:
    trainer = stable_ar.DistributedStableARTrainer(
        model=model,
        conf=trainer_config,
        rng=jax.random.PRNGKey(config.seed),
        optimizer=optimizer,
    )
  else:  # Otherwise fall back to the regular one.
    trainer = stable_ar.StableARTrainer(
        model=model,
        conf=trainer_config,
        rng=jax.random.PRNGKey(config.seed),
        optimizer=optimizer,
    )

  # Setup checkpointing
  ckpt_options = checkpoint.CheckpointManagerOptions(
      save_interval_steps=config.save_interval_steps,
      max_to_keep=config.max_checkpoints_to_keep,
  )
  # Run train
  train.run(
      train_dataloader=train_loader,
      eval_dataloader=eval_loader,
      eval_every_steps=config.save_interval_steps,
      num_batches_per_eval=1,
      trainer=trainer,
      workdir=workdir,
      total_train_steps=config.train_steps,
      metric_aggregation_steps=config.metric_aggregation_steps,
      callbacks=[
          callbacks.TrainStateCheckpoint(
              base_dir=workdir,
              options=ckpt_options,
          ),
          callbacks.ProgressReport(
              num_train_steps=config.train_steps,
          ),
          callbacks.TqdmProgressBar(
              total_train_steps=config.train_steps,
              train_monitors=["loss"],
              eval_monitors=["sd"],
          ),
          fig_callback_cls(),
      ],
  )


if __name__ == "__main__":
  app.run(main)
