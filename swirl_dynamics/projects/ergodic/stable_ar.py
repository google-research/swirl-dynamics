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

"""Unified pipeline for autoregressive modeling.

This pipeline contains implementations of instances necessary for the swirl
dynamics framework, namely, data preparation, model, and trainer.
"""

import dataclasses
import functools
from typing import Any

from clu import metrics as clu_metrics
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from swirl_dynamics.lib.solvers import ode
from swirl_dynamics.projects.ergodic import choices
from swirl_dynamics.projects.ergodic import measure_distances
from swirl_dynamics.projects.ergodic import utils as ergodic_utils
from swirl_dynamics.templates import callbacks
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers


Array = jax.Array
PyTree = Any


@dataclasses.dataclass(kw_only=True)
class StableARModelConfig:
  """Config used by stable autoregressive (AR) models."""

  state_dimension: tuple[int, ...]
  dynamics_model: nn.Module
  integrator: choices.Integrator
  measure_dist: measure_distances.MeasureDistFn
  use_pushfwd: bool = False
  add_noise: bool = False
  noise_level: float = 1e-3
  measure_dist_lambda: float = 0.0
  measure_dist_k_lambda: float = 0.0
  num_lookback_steps: int = 1
  use_sobolev_norm: bool = False
  order_sobolev_norm: int = 1
  normalize_stats: dict[str, Array | None] | None = None
  mmd_bandwidth: tuple[float, ...] = (0.2, 0.5, 0.9, 1.3)


@dataclasses.dataclass(kw_only=True)
class StableARModel(models.BaseModel):
  """Model used for stable AR modeling."""

  conf: StableARModelConfig

  def __post_init__(self):
    pred_integrator = self.conf.integrator.dispatch()
    self.pred_integrator = functools.partial(
        pred_integrator, ode.nn_module_to_dynamics(self.conf.dynamics_model)
    )
    # TODO: Check if this is compatible with distributed training.
    self.vmapped_measure_dist = jax.vmap(self.conf.measure_dist, in_axes=(1, 1))

  def initialize(self, rng):
    init_input = jnp.ones((1,) + self.conf.state_dimension)
    return self.conf.dynamics_model.init(rng, init_input)

  def loss_fn(
      self,
      params: PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: PyTree,
  ) -> models.LossAndAux:
    """Computes training loss and metrics."""
    # For expected shape comments below, ... corresponds to:
    # - Lorenz: 3
    # - KS: spatial_dim, 1
    # - NS: spatial_dim, spatial_dim, 1
    true = batch["true"]
    x0 = batch["x0"]
    # When using data parallelism, it will add an extra dimension due to the
    # pmap_reshape, so this line is to avoid shape mismatches.
    tspan = batch["tspan"].reshape((-1,))
    rollout_weight = batch["rollout_weight"].reshape((-1,))

    if self.conf.add_noise:
      noise = self.conf.noise_level + jax.random.normal(rng, x0.shape)
      x0 += noise
    if self.conf.use_pushfwd:
      # Rollout for t-1 steps with stop gradient.
      # Expected shape: (bsz, num_rollout_steps+num_lookback_steps+1, ...).
      pred_pushfwd = jax.lax.stop_gradient(
          self.pred_integrator(
              x0, batch["tspan"][:-1], dict(params=params, **mutables)
          )
      )
      if self.conf.num_lookback_steps > 1:
        # Expected shape: (batch_size, num_lookback_steps, ...).
        pred_pushfwd = pred_pushfwd[:, -self.conf.num_lookback_steps :, ...]
      else:
        # Expected shape: (batch_size, ...) - no temporal dim.
        pred_pushfwd = pred_pushfwd[:, -1, ...]
      # Pushforward for final step.
      # Expected shape: (batch_size, ...) - no temporal dim.
      pred = self.pred_integrator(
          pred_pushfwd, batch["tspan"][-2:], dict(params=params, **mutables)
      )[:, -1, ...]

      # Computing losses.
      measure_dist = (
          self.conf.measure_dist(pred, true[:, 0, ...]) * rollout_weight[-1]
      )
      measure_dist_k = (
          self.conf.measure_dist(pred, true[:, -1, ...]) * rollout_weight[-1]
      )

      # Compare to true trajectory last step.
      if self.conf.use_sobolev_norm:
        # TODO: Rollout weighting not implemented for this case!
        # The spatial dimension is the length of the shape minus 2,
        # which accounts for the batch, frame, and channel dimensions.
        dim = len(pred.shape) - 2
        l2 = ergodic_utils.sobolev_norm(
            pred - true[:, -1, ...], s=self.conf.order_sobolev_norm, dim=dim
        )
      else:
        l2 = jnp.mean(
            jnp.square(pred - true[:, -1, ...]).mean(
                axis=tuple(range(1, pred.ndim))
            )
            * rollout_weight[-1]
        )

    else:
      # Regular unrolling without stop-gradient.
      # Expected shape: (bsz, num_rollout_steps, ...)
      pred = self.pred_integrator(x0, tspan, dict(params=params, **mutables))[
          :, self.conf.num_lookback_steps :, ...
      ]
      measure_dist = jnp.mean(
          jax.vmap(
              lambda p: self.conf.measure_dist(p, true[:, 0, ...]),
              in_axes=(1),
          )(pred)
          * rollout_weight
      )
      measure_dist_k = jnp.mean(
          self.vmapped_measure_dist(pred, true[:, 1:, ...]) * rollout_weight
      )

      # Compare to full reference trajectory.
      # TODO: this is code is repeated.
      if self.conf.use_sobolev_norm:
        # TODO: Rollout weighting not implemented for this case!
        dim = len(pred.shape) - 3
        l2 = ergodic_utils.sobolev_norm(
            pred - true[:, 1:, ...],
            s=self.conf.order_sobolev_norm,
            dim=dim,
        )
      else:
        l2 = jnp.mean(
            jnp.square(pred - true[:, 1:, ...]).mean(
                axis=tuple(range(2, pred.ndim))
            )
            * rollout_weight
        )

    # Gathering the metrics.
    loss = l2
    loss += self.conf.measure_dist_lambda * measure_dist
    loss += self.conf.measure_dist_k_lambda * measure_dist_k

    metric = dict(
        loss=loss,
        l2=l2,
        measure_dist=measure_dist,
        measure_dist_k=measure_dist_k,
        rollout=jnp.array(tspan.shape[0] - 1),
        max_rollout_decay=rollout_weight[-1],
    )

    return loss, (metric, mutables)

  # pytype: disable=bad-return-type
  def eval_fn(
      self,
      variables: PyTree,
      # Batch is a dict with keys: ['ic', 'true', 'tspan', 'normalize_stats'].
      batch: models.BatchType,
      rng: Array,
      **kwargs,
  ) -> models.ArrayDict:
    """Computes evaluation metrics.

    Args:
      variables: Weights and other parameters of the network.
      batch: Batch of data, in this case it is a dictionary with keys  ['ic',
        'true', 'tspan', 'normalize_stats'].
      rng: Seed for the random number generator.
      **kwargs: Extra keyword variables.

    Returns:
      A dictionary with all the evaluation variables.
    """
    tspan = batch["tspan"].reshape((-1,))
    # Keep extra step for plot functions.
    pred_trajs = self.pred_integrator(batch["ic"], tspan, variables)[
        :, self.conf.num_lookback_steps - 1 :, ...
    ]
    trajs = batch["true"]
    if (
        self.conf.normalize_stats is not None
        and self.conf.normalize_stats["mean"] is not None
        and self.conf.normalize_stats["std"] is not None
    ):
      trajs *= self.conf.normalize_stats["std"]
      trajs += self.conf.normalize_stats["mean"]
      pred_trajs *= self.conf.normalize_stats["std"]
      pred_trajs += self.conf.normalize_stats["mean"]

    # TODO: This only computes the local sinkhorn distance.
    sd = measure_distances.sinkhorn_div(
        pred_trajs[:, -1, ...], trajs[:, -1, ...]
    )
    dt = tspan[1] - tspan[0]
    return dict(
        sd=sd,
        dt=dt,
        trajs=trajs,
        pred_trajs=pred_trajs,
    )
  # pytype: enable=bad-return-type


@dataclasses.dataclass
class StableARTrainerConfig:
  """Config used by stable AR trainers."""

  rollout_weighting: choices.RolloutWeightingFn
  num_rollout_steps: int = 1
  num_lookback_steps: int = 1
  add_noise: bool = False
  use_curriculum: bool = False
  use_pushfwd: bool = False
  train_steps_per_cycle: int = 0
  time_steps_increase_per_cycle: int = 0


class StableARTrainer(trainers.BasicTrainer):
  """Trainer used for stable AR modeling."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    loss: clu_metrics.Average.from_output("loss")
    loss_std: clu_metrics.Std.from_output("loss")
    l2: clu_metrics.Average.from_output("l2")
    l2_std: clu_metrics.Std.from_output("l2")
    measure_dist: clu_metrics.Average.from_output("measure_dist")
    measure_dist_std: clu_metrics.Std.from_output("measure_dist")
    measure_dist_k: clu_metrics.Average.from_output("measure_dist_k")
    measure_dist_k_std: clu_metrics.Std.from_output("measure_dist_k")
    rollout: clu_metrics.Average.from_output("rollout")
    max_rollout_decay: clu_metrics.Average.from_output("max_rollout_decay")

  @flax.struct.dataclass
  class EvalMetrics(clu_metrics.Collection):
    sd: clu_metrics.Average.from_output("sd")
    dt: clu_metrics.Average.from_output("dt")
    all_trajs: clu_metrics.CollectingMetric.from_outputs(
        ("trajs", "pred_trajs")
    )

  def __init__(self, conf: StableARTrainerConfig, *args, **kwargs):
    self.conf = conf
    super().__init__(*args, **kwargs)

  def _preprocess_train_batch(
      self,
      batch_data: trainers.BatchType,
      num_time_steps: int,
  ) -> trainers.BatchType:
    """Internal method to prepocesses train batches based on num_time_steps.

    This method can be overriden by different trainers.

    Args:
      batch_data: training batch data yielded by the dataset.
      num_time_steps: number of rollout steps to be included in ground truth.

    Returns:
      The preprocessed batch data.
    """

    dt = jnp.mean(jnp.diff(batch_data["t"], axis=1))
    tspan = jnp.arange(num_time_steps) * dt
    rollout_weight = self.conf.rollout_weighting(num_time_steps)
    # `x0`: first "state" (which can be `num_lookback_steps` time steps).
    # `true`: num_rollout_steps + 1 states (where the first state corresponds to
    # x0, except when num_lookback_steps > 1, where true[:, 0] corresponds to
    # the last time step in x0).
    if self.conf.num_lookback_steps > 1:
      # Expected shape:
      # - Lorenz: (bsz, num_lookback_steps, 3)
      # - KS: (bsz, num_lookback_steps, spatial_dim, 1)
      # - NS: (bsz, num_lookback_steps, spatial_dim, spatial_dim, 1)
      x0 = batch_data["u"][:, : self.conf.num_lookback_steps, ...]
      # Expected shape: (bsz, num_rollout_steps + 1, ...),
      # where ... is same as just above.
      true = batch_data["u"][
          :,
          self.conf.num_lookback_steps
          - 1 : num_time_steps
          + self.conf.num_lookback_steps
          - 1,
          ...,
      ]
    else:
      # Expected shape (no temporal dim):
      # - Lorenz: (bsz, 3)
      # - KS: (bsz, spatial_dim, 1)
      # - NS: (bsz, spatial_dim, spatial_dim, 1)
      x0 = batch_data["u"][:, 0, ...]
      # Expected shape: (bsz, num_rollout_steps + 1, ...),
      # where ... is same as just above.
      true = batch_data["u"][:, :num_time_steps, ...]
    return dict(
        x0=x0,
        true=true,
        tspan=tspan,
        rollout_weight=rollout_weight,
    )

  def preprocess_train_batch(
      self, batch_data: trainers.BatchType, step: int, rng: Array
  ) -> trainers.BatchType:
    """Wrapper method for _preprocess_train_batch.

    Calculate the number of rollout/time steps to be included in the ground
    truth trajectory, then use the internal preprocess method to build batch.
    Args:
      batch_data: training batch data yielded by the dataset.
      step: the current training step (for scheduled preprocessing).
      rng: a Jax random key for use in case randomized preprocessing is needed.

    Returns:
      The preprocessed batch data.
    """
    # Curr training: increase number of rollout steps every
    # time_steps_increase_per_cycle steps.
    if self.conf.use_curriculum:
      cycle_idx = step // self.conf.train_steps_per_cycle
      num_time_steps = cycle_idx * self.conf.time_steps_increase_per_cycle
      num_time_steps += self.conf.num_rollout_steps + 1
    else:
      num_time_steps = self.conf.num_rollout_steps + 1
    # TODO: Should we remove this random sampling?
    if self.conf.use_pushfwd and num_time_steps > 2:
      num_time_steps = jax.random.randint(
          rng, (1,), minval=2, maxval=num_time_steps + 1
      )[0]
    # pytype: disable=attribute-error
    assert num_time_steps <= batch_data["u"].shape[1], (
        f"Not enough time steps in data ({batch_data['u'].shape[1]}) for"
        f" desired steps ({num_time_steps})."
    )
    # pytype: enable=attribute-error
    return self._preprocess_train_batch(batch_data, num_time_steps)

  def preprocess_eval_batch(
      self, batch_data: trainers.BatchType, rng: Array
  ) -> trainers.BatchType:
    """Preprocessed batch data."""
    if self.conf.num_lookback_steps > 1:
      ic = batch_data["u"][:, : self.conf.num_lookback_steps, ...]
    else:
      ic = batch_data["u"][:, 0, ...]
    dt = jnp.mean(jnp.diff(batch_data["t"], axis=1))
    tspan = jnp.arange(batch_data["t"].shape[1]) * dt  # pytype: disable=attribute-error
    return dict(
        ic=ic,
        true=batch_data["u"],
        tspan=tspan,
    )


class DistributedStableARTrainer(trainers.BasicDistributedTrainer):
  """Trainer used for stable AR modeling."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    loss: clu_metrics.Average.from_output("loss")
    loss_std: clu_metrics.Std.from_output("loss")
    l2: clu_metrics.Average.from_output("l2")
    l2_std: clu_metrics.Std.from_output("l2")
    measure_dist: clu_metrics.Average.from_output("measure_dist")
    measure_dist_std: clu_metrics.Std.from_output("measure_dist")
    measure_dist_k: clu_metrics.Average.from_output("measure_dist_k")
    measure_dist_k_std: clu_metrics.Std.from_output("measure_dist_k")
    rollout: clu_metrics.Average.from_output("rollout")
    max_rollout_decay: clu_metrics.Average.from_output("max_rollout_decay")

  @flax.struct.dataclass
  class EvalMetrics(clu_metrics.Collection):
    sd: clu_metrics.Average.from_output("sd")
    dt: clu_metrics.Average.from_output("dt")
    all_trajs: clu_metrics.CollectingMetric.from_outputs(
        ("trajs", "pred_trajs")
    )

  def __init__(self, conf: StableARTrainerConfig, *args, **kwargs):
    self.conf = conf
    super().__init__(*args, **kwargs)

  def _preprocess_train_batch(
      self,
      batch_data: trainers.BatchType,
      num_time_steps: int,
  ) -> trainers.BatchType:
    """Internal method to prepocesses train batches based on num_time_steps.

    This method can be overriden by different trainers.

    Args:
      batch_data: training batch data yielded by the dataset.
      num_time_steps: number of rollout steps to be included in ground truth.

    Returns:
      The preprocessed batch data.
    """

    dt = jnp.mean(jnp.diff(batch_data["t"], axis=1))
    tspan = jnp.arange(num_time_steps) * dt
    rollout_weight = self.conf.rollout_weighting(num_time_steps)
    # `x0`: first "state" (which can be `num_lookback_steps` time steps).
    # `true`: num_rollout_steps + 1 states (where the first state corresponds to
    # x0, except when num_lookback_steps > 1, where true[:, 0] corresponds to
    # the last time step in x0).
    if self.conf.num_lookback_steps > 1:
      x0 = batch_data["u"][:, : self.conf.num_lookback_steps, ...]
      true = batch_data["u"][
          :,
          self.conf.num_lookback_steps - 1 : num_time_steps + self.conf.num_lookback_steps - 1,  # pylint: disable=line-too-long
          ...,
      ]
    else:
      x0 = batch_data["u"][:, 0, ...]
      true = batch_data["u"][:, :num_time_steps, ...]

    batch_dict = dict(
        x0=x0,
        true=true,
        tspan=np.tile(tspan, (jax.device_count(), 1)),
        rollout_weight=np.tile(rollout_weight, (jax.device_count(), 1)),
    )

    return jax.jit(trainers.reshape_for_pmap)(batch_dict)

  def preprocess_train_batch(
      self, batch_data: trainers.BatchType, step: int, rng: Array
  ) -> trainers.BatchType:
    """Wrapper method for _preprocess_train_batch.

    Calculate the number of rollout/time steps to be included in the ground
    truth trajectory, then use the internal preprocess method to build batch.
    Args:
      batch_data: training batch data yielded by the dataset.
      step: the current training step (for scheduled preprocessing).
      rng: a Jax random key for use in case randomized preprocessing is needed.

    Returns:
      The preprocessed batch data.
    """
    # Curr training: increase number of rollout steps every N steps
    if self.conf.use_curriculum:
      cycle_idx = step // self.conf.train_steps_per_cycle
      num_time_steps = cycle_idx * self.conf.time_steps_increase_per_cycle
      num_time_steps += self.conf.num_rollout_steps + 1
    else:
      num_time_steps = self.conf.num_rollout_steps + 1
    # TODO: Should we remove this random sampling?
    if self.conf.use_pushfwd and num_time_steps > 2:
      num_time_steps = jax.random.randint(
          rng, (1,), minval=2, maxval=num_time_steps + 1
      )[0]
    return self._preprocess_train_batch(batch_data, num_time_steps)

  def preprocess_eval_batch(
      self, batch_data: trainers.BatchType, rng: Array
  ) -> trainers.BatchType:
    """Preprocessed batch data."""
    if self.conf.num_lookback_steps > 1:
      ic = batch_data["u"][:, : self.conf.num_lookback_steps, ...]
    else:
      ic = batch_data["u"][:, 0, ...]
    dt = jnp.mean(jnp.diff(batch_data["t"], axis=1))
    tspan = jnp.arange(batch_data["t"].shape[1]) * dt  # pytype: disable=attribute-error

    batch_dict = dict(
        ic=ic,
        true=batch_data["u"],
        tspan=np.tile(tspan, (jax.device_count(), 1)),
    )

    return jax.jit(trainers.reshape_for_pmap)(batch_dict)


class PlotFigures(callbacks.Callback):
  """Callback that plots figures to tensorboard.

  Each experiment should inheret this callback and implement the
  `on_eval_batch_end` method to perform its customized eval plotting.
  """

  @staticmethod
  def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Copied from:
    https://github.com/lanpa/tensorboardX/blob/b81f4c6ee218ca732dcb81893d18181ca9b8590e/tensorboardX/utils.py#L2-L37

    Args:
      figures: matplotlib.pyplot.figure or list of figures
      close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """

    try:
      import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
      import matplotlib.backends.backend_agg as plt_backend_agg  # pylint: disable=g-import-not-at-top
    except ModuleNotFoundError:
      print("please install matplotlib")

    def render_to_rgb(figure):
      canvas = plt_backend_agg.FigureCanvasAgg(figure)  # pylint: disable=undefined-variable
      canvas.draw()
      data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
      w, h = figure.canvas.get_width_height()
      image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
      image_chw = np.moveaxis(image_hwc, source=2, destination=0)
      if close:
        plt.close(figure)  # pylint: disable=undefined-variable
      return image_chw

    if isinstance(figures, list):
      images = [render_to_rgb(figure) for figure in figures]
      return np.stack(images)
    else:
      image = render_to_rgb(figures)
      return image
