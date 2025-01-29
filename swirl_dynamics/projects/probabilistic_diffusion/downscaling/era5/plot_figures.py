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

  This callback plots condition, observation (reference) and generated sample
  fields as pcolormesh plots. Each variable and time slice get their own plots.

  Attributes:
    fig_size: The matplotlib figure size (same for all plots).
    cond_color_range: The (vmin, vmax) values for condition fields.
    sample_color_range: The (vmin, vmax) values for observation and generated
      sample fields.
    plot_every_n_time_steps_cond: Time step interval at which to plot the
      condition fields (to reduce redundancy and storage costs).
    plot_every_n_time_steps_sample: Time step interval at which to plot the
      observation and sample fields (to reduce redundancy and storage costs).
  """

  fig_size: tuple[int, int] = (5, 3)
  cond_color_range: tuple[float, float] = (-3, 3)
  sample_color_range: tuple[float, float] = (-5, 5)
  plot_every_n_time_steps_cond: int = 1
  plot_every_n_time_steps_sample: int = 1

  def _make_pcolormesh_plot(
      self, data: jax.Array, color_range: tuple[float, float]
  ) -> plt.Figure:
    fig, ax = plt.subplots(figsize=self.fig_size)
    pcm = ax.pcolormesh(
        data, vmin=color_range[0], vmax=color_range[1], cmap="coolwarm"
    )
    fig.colorbar(pcm, ax=ax)
    return fig

  def on_eval_batches_end(
      self,
      trainer: templates.BaseTrainer,
      eval_metrics: Mapping[str, jax.Array | Mapping[str, jax.Array]],
  ):
    figures = {}
    plot_data = eval_metrics["eval_plot_data"]

    # All of `example_input`, `example_obs` and `example_sample` are expected to
    # have shape (batch, time, lat, lon, channels). Plots are made for the first
    # batch element and across all times and channels.
    cond = plot_data["example_input"]
    for t in range(0, cond.shape[1], self.plot_every_n_time_steps_cond):
      for i in range(cond.shape[-1]):
        figures[f"cond_time{t}_dim{i}"] = self._make_pcolormesh_plot(
            cond[0, t, ..., i], color_range=self.cond_color_range
        )

    obs = plot_data["example_obs"]
    for t in range(0, obs.shape[1], self.plot_every_n_time_steps_sample):
      for i in range(obs.shape[-1]):
        figures[f"obs_time{t}_dim{i}"] = self._make_pcolormesh_plot(
            obs[0, t, ..., i], color_range=self.sample_color_range
        )

    sample = plot_data["example_sample"]
    for t in range(0, sample.shape[1], self.plot_every_n_time_steps_sample):
      for i in range(sample.shape[-1]):
        figures[f"sample_time{t}_dim{i}"] = self._make_pcolormesh_plot(
            sample[0, t, ..., i], color_range=self.sample_color_range
        )

    self.write_images(trainer.train_state.int_step, figures)
