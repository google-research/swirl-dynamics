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

"""Experiment logic shared by the training and evaluation entry points.

Builds data, model and trainer from a config (schema of
`configs/config_ks_1d.py`), runs training with checkpointing and evaluates a
checkpoint on the held-out samples.
"""

from collections.abc import Callable
import dataclasses
import json
import os
from typing import Any

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import orbax.checkpoint as ocp
from swirl_dynamics.lib.networks import cycle_gan
from swirl_dynamics.projects.debiasing.cycle_gan import dataloaders
from swirl_dynamics.projects.debiasing.cycle_gan import evaluation_metrics
from swirl_dynamics.projects.debiasing.cycle_gan import loss_functions
from swirl_dynamics.projects.debiasing.cycle_gan import models
from swirl_dynamics.projects.debiasing.cycle_gan import trainers
from swirl_dynamics.templates import callbacks
from swirl_dynamics.templates import train

Array = jax.Array
_SYNTHETIC_NUM_TRAJECTORIES = 24
_SYNTHETIC_NUM_TIMES = 8


def _identity(x: Array) -> Array:
  """Identity output activation of the generators."""
  return x


@dataclasses.dataclass(frozen=True)
class Datasets:
  """Training and held-out snapshots of the two domains.

  Attributes:
    train_a: Training snapshots of domain A, `(num_samples, length, 1)`.
    train_b: Training snapshots of domain B.
    test_a: Held-out snapshots of domain A.
    test_b: Held-out snapshots of domain B.
    stats_a: Mean/std of `train_a`, or `None` without normalisation.
    stats_b: Mean/std of `train_b`, or `None` without normalisation.
  """

  train_a: np.ndarray
  train_b: np.ndarray
  test_a: np.ndarray
  test_b: np.ndarray
  stats_a: dataloaders.NormalizationStats | None
  stats_b: dataloaders.NormalizationStats | None


def _smooth_fields(
    rng: np.random.Generator, num_samples: int, length: int, width: int
) -> np.ndarray:
  """Random fields smoothed by a circular moving average of `width` points."""
  noise = rng.normal(size=(num_samples, length))
  kernel = np.ones(width) / width
  fields = np.stack([
      np.convolve(np.tile(row, 3), kernel, mode="same")[length : 2 * length]
      for row in noise
  ])
  return fields[..., None].astype(np.float32)


def synthetic_datasets(config: ml_collections.ConfigDict) -> Datasets:
  """Deterministic synthetic stand-in for the KS data (smoke tests only).

  Domain A consists of rough random fields and domain B of smoother, larger
  amplitude ones, so that the two marginals differ in both spectrum and scale.

  Args:
    config: Experiment config (`sample_length`, `seed`, `normalize`).

  Returns:
    The datasets.
  """
  rng = np.random.default_rng(config.seed)
  num_samples = _SYNTHETIC_NUM_TRAJECTORIES * _SYNTHETIC_NUM_TIMES
  rough = _smooth_fields(rng, num_samples, config.sample_length, width=2)
  smooth = 2.0 * _smooth_fields(rng, num_samples, config.sample_length, width=6)
  split = num_samples - _SYNTHETIC_NUM_TIMES * 4
  return _with_stats(
      rough[:split], smooth[:split], rough[split:], smooth[split:], config
  )


def _with_stats(
    train_a: np.ndarray,
    train_b: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
    config: ml_collections.ConfigDict,
) -> Datasets:
  """Bundles the splits with the training statistics (if `config.normalize`)."""
  stats_a = stats_b = None
  if config.normalize:
    stats_a = dataloaders.normalization_stats(train_a)
    stats_b = dataloaders.normalization_stats(train_b)
  return Datasets(train_a, train_b, test_a, test_b, stats_a, stats_b)


def load_datasets(config: ml_collections.ConfigDict) -> Datasets:
  """Loads the KS datasets of `config`, or the synthetic ones if no path."""
  if not config.dataset_path:
    return synthetic_datasets(config)
  train_a, train_b, test_a, test_b = dataloaders.load_ks_snapshots(
      config.dataset_path,
      config.downsample_factor,
      config.num_test_trajectories,
  )
  return _with_stats(train_a, train_b, test_a, test_b, config)


def build_dataloaders(
    datasets: Datasets, config: ml_collections.ConfigDict
) -> tuple[dataloaders.UnpairedDataLoader, dataloaders.UnpairedDataLoader]:
  """Builds the training and evaluation loaders (training statistics)."""
  train_loader = dataloaders.UnpairedDataLoader(
      datasets.train_a,
      datasets.train_b,
      batch_size=config.batch_size,
      seed=config.seed,
      normalize=config.normalize,
      normalize_stats_a=datasets.stats_a,
      normalize_stats_b=datasets.stats_b,
  )
  eval_loader = dataloaders.UnpairedDataLoader(
      datasets.test_a,
      datasets.test_b,
      batch_size=config.eval_batch_size,
      seed=config.seed + 2,
      normalize=config.normalize,
      normalize_stats_a=datasets.stats_a,
      normalize_stats_b=datasets.stats_b,
  )
  return train_loader, eval_loader


def build_model(config: ml_collections.ConfigDict) -> models.CycleGANModel:
  """Builds the CycleGAN model with 1-D library networks."""
  generator_kwargs: dict[str, Any] = config.generator.to_dict()
  # Physical fields are not bounded; the library default `tanh` would clip.
  generator_kwargs["final_act_fun"] = _identity
  discriminator_kwargs: dict[str, Any] = config.discriminator.to_dict()
  discriminator_kwargs["kernel_size"] = tuple(
      discriminator_kwargs["kernel_size"]
  )
  model_config = models.CycleGANModelConfig(
      lambda_a=config.lambda_a,
      lambda_b=config.lambda_b,
      lambda_id=config.lambda_id,
      gan_mode=loss_functions.GanMode(config.gan_mode),
      lambda_gp=config.lambda_gp,
      use_crossed_dis_loss=config.use_crossed_dis_loss,
  )
  network = models.CycleGAN(
      model_config,
      cycle_gan.Generator(**generator_kwargs),
      cycle_gan.Generator(**generator_kwargs),
      cycle_gan.Discriminator(**discriminator_kwargs),
      cycle_gan.Discriminator(**discriminator_kwargs),
  )
  sample_shape = (config.batch_size, config.sample_length, 1)
  return models.CycleGANModel(
      cycle_gan=network, dim_inputs=(sample_shape, sample_shape)
  )


def build_optimizer(
    config: ml_collections.ConfigDict,
) -> optax.GradientTransformation:
  """Adam with a linear decay to zero after `decay_start_fraction`."""
  decay_start = int(config.decay_start_fraction * config.num_train_steps)
  schedule = optax.join_schedules(
      [
          optax.constant_schedule(config.learning_rate),
          optax.linear_schedule(
              config.learning_rate,
              0.0,
              max(config.num_train_steps - decay_start, 1),
          ),
      ],
      boundaries=[decay_start],
  )
  return optax.adam(schedule, b1=config.beta1)


def build_trainer(
    config: ml_collections.ConfigDict, model: models.CycleGANModel
) -> trainers.CycleGANTrainer:
  """Builds the (single-device) trainer of the experiment."""
  return trainers.CycleGANTrainer(
      model=model,
      rng=jax.random.PRNGKey(config.seed),
      optimizer_gen=build_optimizer(config),
      optimizer_dis=build_optimizer(config),
      n_critic=config.n_critic,
      pool_size=config.pool_size,
      ema_decay=config.ema_decay,
  )


def _standardize(
    samples: np.ndarray, stats: dataloaders.NormalizationStats | None
) -> Array:
  if stats is None:
    return jnp.asarray(samples)
  safe_std = np.where(stats["std"] == 0, 1.0, stats["std"])
  return (jnp.asarray(samples) - stats["mean"]) / safe_std


def _destandardize(
    samples: Array, stats: dataloaders.NormalizationStats | None
) -> Array:
  if stats is None:
    return samples
  return samples * stats["std"] + stats["mean"]


def translate_in_batches(
    translate: Callable[[Array], Array], samples: Array, batch_size: int
) -> Array:
  """Applies `translate` to `samples` in batches of `batch_size`."""
  outputs = [
      translate(samples[start : start + batch_size])
      for start in range(0, samples.shape[0], batch_size)
  ]
  return jnp.concatenate(outputs, axis=0)


def evaluate(
    config: ml_collections.ConfigDict,
    datasets: Datasets,
    state: trainers.CycleGANTrainState,
    model: models.CycleGANModel,
) -> dict[str, float]:
  """Metrics of the A-to-B translation of the held-out samples.

  Args:
    config: Experiment config.
    datasets: Datasets whose `test_a`/`test_b` are evaluated.
    state: Trained (possibly restored) train state.
    model: The model matching `state`.

  Returns:
    `pdf_l1_error`, `wasserstein1`, `spectral_log_ratio` of the translated
    low-fidelity samples against the held-out high-fidelity samples, the same
    three metrics for the untranslated samples (`*_reference`), and the
    relative `cycle_consistency_error` of `F(G(a))`. All values are computed
    in physical units.
  """
  use_ema = config.ema_decay is not None
  to_b = jax.jit(
      trainers.CycleGANTrainer.inference_fn_from_state_dict(
          state, model, direction="a2b", use_ema=use_ema
      )
  )
  to_a = jax.jit(
      trainers.CycleGANTrainer.inference_fn_from_state_dict(
          state, model, direction="b2a", use_ema=use_ema
      )
  )
  test_a = _standardize(datasets.test_a, datasets.stats_a)
  fake_b = translate_in_batches(to_b, test_a, config.eval_batch_size)
  recovered_a = translate_in_batches(to_a, fake_b, config.eval_batch_size)

  real_a = jnp.asarray(datasets.test_a)
  real_b = jnp.asarray(datasets.test_b)
  fake_b = _destandardize(fake_b, datasets.stats_b)
  recovered_a = _destandardize(recovered_a, datasets.stats_a)

  def _distribution_metrics(samples: Array, suffix: str) -> dict[str, float]:
    return {
        f"pdf_l1_error{suffix}": float(
            evaluation_metrics.pdf_l1_error(samples, real_b)
        ),
        f"wasserstein1{suffix}": float(
            evaluation_metrics.wasserstein1_1d(samples, real_b)
        ),
        f"spectral_log_ratio{suffix}": float(
            evaluation_metrics.spectral_log_ratio(samples, real_b)
        ),
    }

  return {
      **_distribution_metrics(fake_b, ""),
      **_distribution_metrics(real_a, "_reference"),
      "cycle_consistency_error": float(
          evaluation_metrics.cycle_consistency_error(real_a, recovered_a)
      ),
  }


def train_experiment(config: ml_collections.ConfigDict, workdir: str) -> None:
  """Trains the experiment described by `config`, checkpointing in `workdir`.

  Args:
    config: Experiment config.
    workdir: Directory for checkpoints and metrics.
  """
  datasets = load_datasets(config)
  train_loader, eval_loader = build_dataloaders(datasets, config)
  model = build_model(config)
  trainer = build_trainer(config, model)
  logging.info(
      "Training %s for %d steps in %s.",
      config.experiment_name,
      config.num_train_steps,
      workdir,
  )
  train.run(
      train_dataloader=train_loader,
      trainer=trainer,
      workdir=workdir,
      total_train_steps=config.num_train_steps,
      metric_aggregation_steps=config.metric_aggregation_steps,
      eval_dataloader=eval_loader,
      eval_every_steps=config.eval_every_steps,
      num_batches_per_eval=config.num_batches_per_eval,
      callbacks=(
          callbacks.TrainStateCheckpoint(
              base_dir=workdir,
              options=ocp.CheckpointManagerOptions(
                  save_interval_steps=config.save_interval_steps,
                  max_to_keep=config.max_checkpoints_to_keep,
              ),
          ),
          callbacks.ProgressReport(num_train_steps=config.num_train_steps),
      ),
  )


def evaluate_experiment(
    config: ml_collections.ConfigDict, workdir: str, output_path: str = ""
) -> dict[str, float]:
  """Evaluates the latest checkpoint of `workdir` and writes the metrics.

  Args:
    config: Experiment config.
    workdir: Directory with the `checkpoints` folder of `train_experiment`.
    output_path: Destination of the JSON metrics; `<workdir>/metrics.json` when
      empty.

  Returns:
    The metrics, see `evaluate`.
  """
  output_path = output_path or os.path.join(workdir, "metrics.json")
  datasets = load_datasets(config)
  model = build_model(config)
  state = trainers.CycleGANTrainState.restore_from_orbax_ckpt(
      os.path.join(workdir, "checkpoints")
  )
  metrics = evaluate(config, datasets, state, model)
  logging.info("Metrics at step %d: %s", int(state.step), metrics)
  out_path = epath.Path(output_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(metrics, indent=2))
  return metrics
