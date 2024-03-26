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

"""Kuramoto Sivashinsky 1D custom plotting callback."""
from typing import Any, Mapping

import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
from swirl_dynamics.projects.ergodic import stable_ar
from swirl_dynamics.projects.ergodic import utils
from swirl_dynamics.templates import callbacks
import xarray

Array = jax.Array
PyTree = Any


def plot_trajectories(
    dt, x_grid, traj_length, trajs, pred_trajs, case_ids=(11, 32, 67, 89)
):
  """Plot sample trajectories."""
  assert trajs.shape[0] > max(case_ids), (
      "Ground truth trajectories do not contain enough samples"
      f" ({trajs.shape[0]}) to select trajectory number {max(case_ids)}."
  )
  assert pred_trajs.shape[0] > max(case_ids), (
      "Prediced trajectories do not contain enough samples"
      f" ({pred_trajs.shape[0]}) to select trajectory number {max(case_ids)}."
  )
  plot_time = jnp.arange(traj_length) * dt
  t_max = plot_time.max()
  fig = plt.figure(figsize=(6, 6 * len(case_ids)), constrained_layout=True)
  subfigs = fig.subfigures(nrows=len(case_ids), ncols=1)
  for case_id, subfig in zip(case_ids, subfigs):
    ax = subfig.subplots(nrows=2, ncols=2, sharey=True)
    xarray.DataArray(
        trajs[case_id, ...].squeeze(axis=-1),
        dims=[r"$t$", r"$x$"],
        coords={r"$t$": plot_time, r"$x$": x_grid},
    ).plot.imshow(ax=ax[0, 0], cmap="bwr", vmin=-4, vmax=4, add_colorbar=False)
    ax[0, 0].set_ylim([0, t_max])
    ax[0, 0].grid(False)
    ax[0, 0].set_title("GT")
    ax[1, 0].axis("off")
    ax[1, 0].text(
        0.5,
        0.5,
        f"Traj #: {case_id}",
        fontsize=14,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[1, 0].transAxes,
    )

    xarray.DataArray(
        pred_trajs[case_id, ...].squeeze(axis=-1),
        dims=["", r"$x$"],
        coords={"": plot_time, r"$x$": x_grid},
    ).plot.imshow(
        ax=ax[0, 1],
        cmap="bwr",
        vmin=-4,
        vmax=4,
        robust=True,
        add_colorbar=False,
    )
    ax[0, 1].set_ylim([0, t_max])
    ax[0, 1].grid(False)
    ax[0, 1].set_title("Predicted")

    xarray.DataArray(
        jnp.abs((trajs[case_id] - pred_trajs[case_id]).squeeze()),
        dims=[r"Abs. error, $t$", r"$x$"],
        coords={r"Abs. error, $t$": plot_time, r"$x$": x_grid},
    ).plot.imshow(
        ax=ax[1, 1],
        cmap="Greys",
        vmin=0,
        vmax=4,
        robust=True,
        add_colorbar=False,
    )
    ax[1, 1].set_xlabel("")
    ax[1, 1].set_ylim([0, t_max])
    ax[1, 1].grid(False)

  return {"sample_traj": fig}


class KS1DPlotFigures(stable_ar.PlotFigures):
  """Kuramoto Sivashinsky 1D plotting."""

  def __init__(self, cos_sim_plot_steps: int = 500):
    super().__init__()
    # Correlation breaks down early, do not need all the steps.
    self.cos_sim_plot_steps = cos_sim_plot_steps

  def on_eval_batches_end(
      self, trainer: callbacks.Trainer, eval_metrics: Mapping[str, Array]
  ) -> None:
    dt = eval_metrics["dt"]
    traj_length = eval_metrics["all_trajs"]["trajs"].shape[1]
    figs = utils.plot_error_metrics(
        dt=dt,
        traj_length=traj_length,
        trajs=eval_metrics["all_trajs"]["trajs"],
        pred_trajs=eval_metrics["all_trajs"]["pred_trajs"],
    )
    figs.update(
        plot_trajectories(
            dt=dt,
            x_grid=jnp.arange(eval_metrics["all_trajs"]["trajs"].shape[-2]),
            traj_length=traj_length,
            trajs=eval_metrics["all_trajs"]["trajs"],
            pred_trajs=eval_metrics["all_trajs"]["pred_trajs"],
        )
    )
    figs.update(
        utils.plot_cos_sims(
            dt=dt,
            traj_length=min(traj_length, self.cos_sim_plot_steps),
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
