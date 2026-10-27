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

"""Experiment logic for 1-D optimal transport debiasing.

Loads Kuramoto-Sivashinsky (or deterministic synthetic) snapshots, fits an
entropic optimal transport solver between low-fidelity and high-fidelity
training snapshots, applies the barycentric or centered debiased transport map
to held-out snapshots, and computes distributional, spectral, and Sinkhorn
divergence evaluation metrics.
"""

import dataclasses
import json
import os

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from swirl_dynamics.projects.debiasing.cycle_gan import dataloaders
from swirl_dynamics.projects.debiasing.cycle_gan import evaluation_metrics
from swirl_dynamics.projects.debiasing.optimal_transport import divergence
from swirl_dynamics.projects.debiasing.optimal_transport import sinkhorn

Array = jax.Array
_SYNTHETIC_NUM_TRAJECTORIES = 24
_SYNTHETIC_NUM_TIMES = 8
_MAX_EPSILON_SAMPLES = 1024


@dataclasses.dataclass(frozen=True)
class Datasets:
  """Training and held-out snapshots of the two domains.

  Attributes:
    train_a: Training snapshots of domain A, `(num_samples, length, 1)`.
    train_b: Training snapshots of domain B, `(num_samples, length, 1)`.
    test_a: Held-out snapshots of domain A, `(num_samples, length, 1)`.
    test_b: Held-out snapshots of domain B, `(num_samples, length, 1)`.
  """

  train_a: np.ndarray
  train_b: np.ndarray
  test_a: np.ndarray
  test_b: np.ndarray


@dataclasses.dataclass(frozen=True)
class FitResult:
  """Fitted entropic optimal transport solver and training dual potentials.

  Attributes:
    solver: Configured `SinkhornSolver` instance used for the fit.
    output: `SinkhornOutput` from solving between `train_source` and
      `train_target`.
    train_source: Flattened source training samples of shape `(n, length)`.
    train_target: Flattened target training samples of shape `(m, length)`.
    epsilon: Resolved positive entropic regularization parameter.
  """

  solver: sinkhorn.SinkhornSolver
  output: sinkhorn.SinkhornOutput
  train_source: Array
  train_target: Array
  epsilon: float


def _smooth_fields(
    rng: np.random.Generator, num_samples: int, length: int, width: int
) -> np.ndarray:
  """Random fields smoothed by a circular moving average of `width` points."""
  noise = rng.normal(size=(num_samples, length))
  width = max(1, min(width, length))
  kernel = np.ones(width) / width
  fields = np.stack([
      np.convolve(np.tile(row, 3), kernel, mode="same")[length : 2 * length]
      for row in noise
  ])
  return fields[..., None].astype(np.float32)


def _periodic_modes(
    rng: np.random.Generator, num_samples: int, length: int
) -> np.ndarray:
  """Low-wavenumber periodic fields on `[0, 2*pi)` of shape `(N, length, 1)`."""
  grid = np.linspace(0.0, 2.0 * np.pi, length, endpoint=False)[None, :]
  coeff_1 = rng.normal(size=(num_samples, 2))
  coeff_2 = 0.5 * rng.normal(size=(num_samples, 2))
  wave = (
      coeff_1[:, 0:1] * np.cos(grid)
      + coeff_1[:, 1:2] * np.sin(grid)
      + coeff_2[:, 0:1] * np.cos(2.0 * grid)
      + coeff_2[:, 1:2] * np.sin(2.0 * grid)
  )
  return wave[..., None].astype(np.float32)


def synthetic_datasets(config: ml_collections.ConfigDict) -> Datasets:
  """Deterministic synthetic stand-in for the KS data (smoke tests only).

  Domain A has a shifted mean, smaller amplitude, and rougher circular-smoothed
  spectrum, whereas domain B has zero mean, larger amplitude, and a smoother
  spectrum, so entropic optimal transport substantially reduces `pdf_l1_error`,
  `wasserstein1`, and `sinkhorn_divergence`.

  Args:
    config: Experiment config (`sample_length`, `seed`).

  Returns:
    The synthetic training and test datasets.
  """
  rng = np.random.default_rng(config.seed)
  num_samples = _SYNTHETIC_NUM_TRAJECTORIES * _SYNTHETIC_NUM_TIMES
  length = int(config.sample_length)
  modes_a = _periodic_modes(rng, num_samples, length)
  modes_b = _periodic_modes(rng, num_samples, length)
  rough_noise = _smooth_fields(rng, num_samples, length, width=2)
  smooth_noise = _smooth_fields(rng, num_samples, length, width=6)

  domain_a = (1.2 + 0.6 * modes_a + 0.35 * rough_noise).astype(np.float32)
  domain_b = (1.6 * modes_b + 0.25 * smooth_noise).astype(np.float32)
  split = num_samples - _SYNTHETIC_NUM_TIMES * 4
  return Datasets(
      train_a=domain_a[:split],
      train_b=domain_b[:split],
      test_a=domain_a[split:],
      test_b=domain_b[split:],
  )


def load_datasets(config: ml_collections.ConfigDict) -> Datasets:
  """Loads the KS datasets of `config`, or synthetic ones when path is empty."""
  if not config.dataset_path:
    return synthetic_datasets(config)
  train_a, train_b, test_a, test_b = dataloaders.load_ks_snapshots(
      config.dataset_path,
      config.downsample_factor,
      config.num_test_trajectories,
  )
  return Datasets(
      train_a=train_a,
      train_b=train_b,
      test_a=test_a,
      test_b=test_b,
  )


def resolve_epsilon(
    x_train: Array,
    y_train: Array,
    config: ml_collections.ConfigDict,
) -> float:
  """Resolves a positive entropic regularization parameter `epsilon`.

  Args:
    x_train: Source training samples of shape `(n, length)`.
    y_train: Target training samples of shape `(m, length)`.
    config: Experiment config (`epsilon`, `epsilon_scale`).

  Returns:
    `float(config.epsilon)` when `config.epsilon > 0.0`, else
    `float(config.epsilon_scale * median(C))` computed on up to 1024 samples
    (falling back to `float(config.epsilon_scale)` if `median(C) <= 0.0`).
  """
  if config.epsilon > 0.0:
    return float(config.epsilon)
  x_sub = jnp.asarray(x_train[:_MAX_EPSILON_SAMPLES])
  y_sub = jnp.asarray(y_train[:_MAX_EPSILON_SAMPLES])
  cost = jnp.sum(jnp.square(x_sub[:, None, :] - y_sub[None, :, :]), axis=-1)
  med = float(jnp.median(cost))
  if med <= 0.0:
    return float(config.epsilon_scale)
  return float(config.epsilon_scale * med)


def fit_ot_map(
    config: ml_collections.ConfigDict, datasets: Datasets
) -> FitResult:
  """Fits the entropic optimal transport solver on training snapshots.

  Args:
    config: Experiment configuration.
    datasets: Training and held-out datasets.

  Returns:
    A `FitResult` holding the solver, `SinkhornOutput`, flattened training
    arrays, and resolved `epsilon`.
  """
  if config.enable_x64:
    jax.config.update("jax_enable_x64", True)
  dtype = (
      jnp.float64
      if (config.enable_x64 and jax.config.jax_enable_x64)
      else jnp.float32
  )

  x_all = datasets.train_a.reshape((datasets.train_a.shape[0], -1))
  y_all = datasets.train_b.reshape((datasets.train_b.shape[0], -1))

  rng = np.random.default_rng(config.seed)
  max_samples = int(config.max_train_samples)
  if x_all.shape[0] > max_samples:
    idx_x = rng.choice(x_all.shape[0], size=max_samples, replace=False)
    x_all = x_all[idx_x]
  if y_all.shape[0] > max_samples:
    idx_y = rng.choice(y_all.shape[0], size=max_samples, replace=False)
    y_all = y_all[idx_y]

  x_train = jnp.asarray(x_all, dtype=dtype)
  y_train = jnp.asarray(y_all, dtype=dtype)
  epsilon = resolve_epsilon(x_train, y_train, config)

  solver = sinkhorn.SinkhornSolver(
      epsilon=epsilon,
      num_iterations=int(config.num_iterations),
      threshold=float(config.threshold),
      criterion=config.criterion,
      inner_iterations=int(config.inner_iterations),
  )
  output = solver(x_train, y_train)
  return FitResult(
      solver=solver,
      output=output,
      train_source=x_train,
      train_target=y_train,
      epsilon=epsilon,
  )


def apply_ot_map(
    fit: FitResult,
    samples: np.ndarray | Array,
    config: ml_collections.ConfigDict,
) -> Array:
  """Applies the fitted entropic or debiased OT map to 2-D or 3-D samples.

  Args:
    fit: Fitted OT result from `fit_ot_map`.
    samples: Input snapshots of shape `(B, length)` or `(B, length, 1)`.
    config: Experiment config (`use_debiased_map`, `eval_batch_size`).

  Returns:
    Transported snapshots with the same rank and shape as `samples`.

  Raises:
    ValueError: If `samples.ndim` is not 2 or 3.
  """
  if samples.ndim not in (2, 3):
    raise ValueError(
        "samples must be a 2-D or 3-D array, got"
        f" samples.ndim={samples.ndim} with shape={samples.shape}."
    )
  queries = jnp.asarray(samples, dtype=fit.train_target.dtype)
  queries_2d = (
      queries.reshape((queries.shape[0], -1)) if queries.ndim == 3 else queries
  )
  batch_size = (
      int(config.eval_batch_size) if config.eval_batch_size > 0 else None
  )

  if config.use_debiased_map:
    map_fn = divergence.debiased_transport_fn(
        fit.solver,
        fit.train_source,
        fit.train_target,
        batch_size=batch_size,
    )
  else:
    map_fn = fit.solver.transport_fn(
        fit.output.gv,
        fit.train_target,
        batch_size=batch_size,
    )

  mapped_2d = map_fn(queries_2d)
  if samples.ndim == 3:
    return mapped_2d.reshape(samples.shape)
  return mapped_2d


def evaluate(
    config: ml_collections.ConfigDict,
    datasets: Datasets,
    fit: FitResult,
) -> tuple[Array, dict[str, float]]:
  """Evaluates the fitted OT map on held-out test snapshots.

  Args:
    config: Experiment config.
    datasets: Datasets whose `test_a` and `test_b` splits are evaluated.
    fit: Fitted OT solver and potentials from `fit_ot_map`.

  Returns:
    A tuple `(debiased, metrics)` where `debiased` has shape
    `(num_test, length, 1)` and `metrics` holds distributional, spectral,
    Sinkhorn divergence, and solver convergence diagnostics.
  """
  debiased = apply_ot_map(fit, datasets.test_a, config)
  real_a = jnp.asarray(datasets.test_a, dtype=debiased.dtype)
  real_b = jnp.asarray(datasets.test_b, dtype=debiased.dtype)

  max_div = (
      int(config.sinkhorn_divergence_max_samples)
      if config.sinkhorn_divergence_max_samples > 0
      else None
  )
  target_2d = real_b.reshape((real_b.shape[0], -1))[:max_div]

  def _distribution_metrics(samples: Array, suffix: str) -> dict[str, float]:
    samples_2d = samples.reshape((samples.shape[0], -1))[:max_div]
    sink_div = divergence.sinkhorn_divergence(
        samples_2d, target_2d, solver=fit.solver
    )
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
        f"sinkhorn_divergence{suffix}": float(sink_div),
    }

  assert fit.output.num_iterations is not None
  assert fit.output.marginal_error is not None
  assert fit.output.reg_ot_cost is not None
  metrics = {
      **_distribution_metrics(debiased, ""),
      **_distribution_metrics(real_a, "_reference"),
      "epsilon": float(fit.epsilon),
      "num_iterations": float(fit.output.num_iterations),
      "converged": float(bool(fit.output.converged)),
      "marginal_error": float(fit.output.marginal_error),
      "reg_ot_cost": float(fit.output.reg_ot_cost),
      "entropic_ot_cost": float(fit.output.entropic_ot_cost),
  }
  return debiased, metrics


def run_experiment(
    config: ml_collections.ConfigDict,
    workdir: str,
    output_path: str = "",
) -> dict[str, float]:
  """Fits and evaluates the OT debiasing experiment and writes JSON metrics.

  Args:
    config: Experiment configuration.
    workdir: Working directory for experiment outputs.
    output_path: Destination of the JSON metrics; defaults to
      `<workdir>/metrics.json` when empty.

  Returns:
    Dictionary of evaluation and solver diagnostic metrics.
  """
  output_path = output_path or os.path.join(workdir, "metrics.json")
  datasets = load_datasets(config)
  logging.info("Fitting OT map for %s in %s.", config.experiment_name, workdir)
  fit = fit_ot_map(config, datasets)
  _, metrics = evaluate(config, datasets, fit)
  logging.info("OT evaluation metrics: %s", metrics)
  out_path = epath.Path(output_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(metrics, indent=2))
  return metrics
