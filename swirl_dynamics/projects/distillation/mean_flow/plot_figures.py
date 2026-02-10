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

"""Callback for visualizing generated samples on tensorboard."""

import dataclasses
from typing import Mapping

import jax
import matplotlib.pyplot as plt
from swirl_dynamics import templates


@templates.utils.primary_process_only
@dataclasses.dataclass
class PlotSamples(templates.MatplotlibFigureAsImage):
  """Plot samples.

  This callback plots the generated distilled samples as imshow plots.

  Attributes:
    fig_size: The matplotlib figure size (same for all plots).
    color_range: The (vmin, vmax) values for plotted samples.
    plot_every_n_samples: The number of every n samples to plot. This is useful
      to avoid plotting too many samples and slow down the tensorboard.
  """

  fig_size: tuple[int, int] = (5, 3)
  color_range: tuple[float, float] = (0, 1)
  plot_every_n_samples: int = 1

  def _make_imshow_plot(
      self, data: jax.Array, color_range: tuple[float, float]
  ) -> plt.Figure:
    fig, ax = plt.subplots(figsize=self.fig_size)
    im = ax.imshow(
        data,
        vmin=color_range[0],
        vmax=color_range[1],
        cmap="coolwarm",
        aspect="auto",
    )
    fig.colorbar(im, ax=ax)
    return fig

  def on_eval_batches_end(
      self,
      trainer: templates.BaseTrainer,
      eval_metrics: Mapping[str, jax.Array | Mapping[str, jax.Array]],
  ):
    figures = {}
    plot_data = eval_metrics["eval_plot_data"]

    # TODO: Add the labels of each plot.
    if isinstance(plot_data, dict):
      samples = plot_data["eval_samples"]
      for idx_batch in range(0, samples.shape[0], self.plot_every_n_samples):
        figures[f"sample_{idx_batch}"] = self._make_imshow_plot(
            samples[idx_batch, ..., 0], color_range=self.color_range
        )
    else:
      raise ValueError(
          "Plot data is expected to be a dict of samples, but got"
          f" {type(plot_data)}"
      )

    self.write_images(trainer.train_state.int_step, figures)
