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

"""Navier Stokes 2D custom plotting callback."""
from typing import Any, Mapping, Sequence

import jax
import matplotlib.pyplot as plt
from swirl_dynamics.projects.ergodic import stable_ar
from swirl_dynamics.projects.ergodic import utils
from swirl_dynamics.templates import callbacks

Array = jax.Array
PyTree = Any


def plot_trajectories(
    dt: Array,
    traj_lengths: list[int] | int,
    trajs: Array,
    pred_trajs: Array,
    case_ids: Sequence[int] = (1, 3, 5, 7),
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
  fig = plt.figure(
      figsize=(3 * len(traj_lengths), 6 * len(case_ids)),
      constrained_layout=True,
  )
  subfigs = fig.subfigures(nrows=len(case_ids), ncols=1)
  for case_id, subfig in zip(case_ids, subfigs):
    ax = subfig.subplots(
        nrows=2, ncols=len(traj_lengths), sharex=True, sharey=True
    )
    ax[0, 0].set_ylabel(f"GT (Traj #: {case_id})")
    ax[1, 0].set_ylabel("Predicted")
    for c, traj_length in enumerate(traj_lengths):
      ax[0, c].imshow(trajs[case_id, traj_length, ...].squeeze(axis=-1))
      ax[0, c].set_title(f"Time: {traj_length*dt:0.2f}")
      ax[0, c].set_xticks([])
      ax[0, c].set_yticks([])
      ax[1, c].imshow(pred_trajs[case_id, traj_length, ...].squeeze(axis=-1))
      ax[1, c].set_xticks([])
      ax[1, c].set_yticks([])
  return {"sample_traj": fig}


class NS2dPlotFigures(stable_ar.PlotFigures):
  """Navier Stokes 2D plotting."""

  def __init__(self, cos_sim_plot_steps: int = 200):
    super().__init__()
    # Correlation breaks down early, do not need all the steps
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
            traj_lengths=[0, 1, 2, 5, traj_length // 2, traj_length - 1],
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
