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

"""Lorenz 63 custom plotting callback."""
from typing import Any, Mapping

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
from swirl_dynamics.projects.ergodic import stable_ar
from swirl_dynamics.templates import callbacks

Array = jax.Array
PyTree = Any


def lorenz63_dynamics(x: Array, t: Array, params: PyTree) -> Array:
  """Lorenz 63 dynamics.

  Time derivative governing Lorenz 63 equation:
    dot{x} = sigma(y - x)
    dot{y} = x(rho - z) - y
    dot{z} = xy - beta*z

  Arguments:
      x: Current state.
      t: Time (unused; system is autonomous).
      params: Not used.

  Returns:
    dot{x}: Time derivative of state vector.
  """

  del t, params
  sigma, rho, beta = 10.0, 28.0, 8.0 / 3
  return jnp.asarray([
      sigma * (x[1] - x[0]),
      x[0] * (rho - x[2]) - x[1],
      x[0] * x[1] - beta * x[2],
  ])


def plot_error(
    dt: Array, traj_length: list[int] | int, trajs: Array, pred_trajs: Array
):
  """Plot root mean squared error (RMSE) error over time."""

  def rmse_err(true, pred):
    """Helper method to compute RMSE error."""
    return jnp.sqrt(jnp.sum(jnp.square(true - pred)))

  batch_time_rmse = jax.vmap(jax.vmap(rmse_err, in_axes=(0, 0)), in_axes=(0, 0))
  batch_average_rmse = lambda t, p: jnp.mean(batch_time_rmse(t, p), axis=0)

  fig, ax = plt.subplots(figsize=(7, 5))
  ax.set_xlabel("t")
  ax.set_ylabel("RMSE")
  ax.set_title("Error across time")
  ax.set_xlim(0, traj_length * dt)
  ax.set_ylim(0, 100)
  rmse = batch_average_rmse(trajs, pred_trajs)
  ax.plot(jnp.arange(traj_length) * dt, rmse)
  return {"rmse": fig}


def plot_trajectory_hists(
    dt: Array, traj_lengths: list[int] | int, trajs: Array, pred_trajs: Array
):
  """Plots trajectory histograms."""
  fig = plt.figure(figsize=(14, 3 * len(traj_lengths)), constrained_layout=True)
  subfigs = fig.subfigures(nrows=len(traj_lengths), ncols=1)

  bins = 80
  vmin = 0
  vmax = 100
  x_range = [-28, 28]
  y_range = [-28, 28]
  z_range = [-10, 60]

  for step, subfig in zip(traj_lengths, subfigs):
    x = trajs[:, step - 1, :][:, 0]
    y = trajs[:, step - 1, :][:, 1]
    z = trajs[:, step - 1, :][:, 2]
    ax = subfig.subplots(nrows=1, ncols=6)

    # Plot x vs. y
    ax[0].set_title("RK4")
    ax[0].set_xlabel("x")
    ax[0].set_xticks([-20, 0, 20], [-20, 0, 20])
    ax[0].set_ylabel("y")
    ax[0].set_yticks([-20, 0, 20], [-20, 0, 20])
    range_ = [x_range, y_range]
    kwargs = dict(
        bins=bins,
        range=range_,
        cmap="viridis",
        cmin=1,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax[0].hist2d((x), (y), **kwargs)
    ax[1].set_title("Predicted")
    pred_x = pred_trajs[:, step - 1, :][:, 0]
    pred_y = pred_trajs[:, step - 1, :][:, 1]
    ax[1].set_xlabel("x")
    ax[1].set_xticks([-20, 0, 20], [-20, 0, 20])
    ax[1].set_yticks([], [])
    ax[1].hist2d((pred_x), (pred_y), **kwargs)

    # Plot x vs. z
    ax[2].set_title("RK4")
    ax[2].set_xlabel("x")
    ax[2].set_xticks([-20, 0, 20], [-20, 0, 20])
    ax[2].set_ylabel("z")
    ax[2].set_yticks([0, 20, 40], [0, 20, 40])
    range_ = [x_range, z_range]
    kwargs = dict(
        bins=bins,
        range=range_,
        cmap="viridis",
        cmin=1,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    im = ax[2].hist2d((x), (z), **kwargs)
    ax[3].set_title("Predicted")
    pred_x = pred_trajs[:, step - 1, :][:, 0]
    pred_z = pred_trajs[:, step - 1, :][:, 2]
    ax[3].set_xlabel("x")
    ax[3].set_xticks([-20, 0, 20], [-20, 0, 20])
    ax[3].set_yticks([], [])
    ax[3].hist2d((pred_x), (pred_z), **kwargs)

    # Plot y vs. z.
    ax[4].set_title("RK4")
    ax[4].set_xlabel("y")
    ax[4].set_ylabel("z")
    ax[4].set_xticks([-20, 0, 20], [-20, 0, 20])
    ax[4].set_yticks([0, 20, 40], [0, 20, 40])
    range_ = [y_range, z_range]
    kwargs = dict(
        bins=bins,
        range=range_,
        cmap="viridis",
        cmin=1,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax[4].hist2d((y), (z), **kwargs)
    ax[5].set_title("Predicted")
    pred_y = pred_trajs[:, step - 1, :][:, 1]
    pred_z = pred_trajs[:, step - 1, :][:, 2]
    ax[5].set_xlabel("y")
    ax[5].set_xticks([-20, 0, 20], [-20, 0, 20])
    ax[5].set_yticks([], [])
    ax[5].hist2d((pred_y), (pred_z), **kwargs)

    subfig.colorbar(
        im[-1], ax=ax.tolist(), pad=0.01, label="Count", extend="max"
    )
    subfig.suptitle(
        f"Rollout Time: {step*dt:0.2f} ({step} steps * {dt:0.2f} dt)",
        fontsize="large",
    )
  return {"traj_hists": fig}


def plot_correlations(dt, traj_length, trajs, pred_trajs):
  """Plots coordinate-wise correlations over time."""
  fig, ax = plt.subplots(
      nrows=1, ncols=3, sharey=True, figsize=(17, 5), tight_layout=True
  )
  ax[0].set_ylabel("Corr. Coeff.")
  fig.suptitle("Correlation: ground truth and pred. trajectories across time")
  for d, n in zip(range(3), ["x", "y", "z"]):
    ax[d].plot(
        jnp.arange(traj_length) * dt,
        jnp.ones(traj_length)*0.9,
        color="black", linestyle="dashed",
        label="0.9 threshold"
    )
    ax[d].plot(
        jnp.arange(traj_length) * dt,
        jnp.ones(traj_length)*0.8,
        color="red", linestyle="dashed",
        label="0.8 threshold"
    )
    ax[d].set_xlim(0, traj_length*dt)
    ax[d].set_xlabel("t")
    ax[d].set_title(n)
  for d in range(3):
    corrs = jax.vmap(jnp.corrcoef, in_axes=(1, 1))(
        trajs[:, :traj_length, d], pred_trajs[:, :traj_length, d]
    )[:, 1, 0]
    ax[d].plot(jnp.arange(traj_length) * dt, corrs)
  ax[-1].legend(frameon=False, bbox_to_anchor=(1, 1))
  return {"corr": fig}


class Lorenz63PlotFigures(stable_ar.PlotFigures):
  """Lorenz 63 plotting."""

  def __init__(self, corr_plot_steps: int = 20):
    super().__init__()
    # Correlation breaks down early, do not need all the steps
    self.corr_plot_steps = corr_plot_steps

  def on_eval_batches_end(
      self, trainer: callbacks.Trainer, eval_metrics: Mapping[str, Array]
  ) -> None:
    dt = eval_metrics["dt"]
    traj_length = eval_metrics["all_trajs"]["trajs"].shape[1]
    figs = plot_error(
        dt=dt,
        traj_length=traj_length,
        trajs=eval_metrics["all_trajs"]["trajs"],
        pred_trajs=eval_metrics["all_trajs"]["pred_trajs"],
    )
    figs.update(
        plot_trajectory_hists(
            dt=dt,
            traj_lengths=[traj_length // 2, traj_length],
            trajs=eval_metrics["all_trajs"]["trajs"],
            pred_trajs=eval_metrics["all_trajs"]["pred_trajs"],
        )
    )
    figs.update(
        plot_correlations(
            dt=dt,
            traj_length=min(traj_length, self.corr_plot_steps),
            trajs=eval_metrics["all_trajs"]["trajs"],
            pred_trajs=eval_metrics["all_trajs"]["pred_trajs"],
        )
    )
    figs_as_images = {
        k: self.figure_to_image(v).transpose(1, 2, 0) for k, v in figs.items()
    }
    self.metric_writer.write_images(
        trainer.train_state.int_step, figs_as_images
    )
