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

"""Callbacks for downscaling training pipelines."""

import dataclasses
from typing import Mapping

import jax
import matplotlib.pyplot as plt
from swirl_dynamics import templates


@templates.utils.primary_process_only
@dataclasses.dataclass
class PlotSamples(templates.MatplotlibFigureAsImage):
  """Sample plotting callback.

  To use this callback in the experiment pipeline, add to the gin file:

  ```
  templates.run_train:
    ...
    callbacks = [
        ...
        @callback_lib.PlotSamples(),
    ]
  ```
  """

  fig_size: tuple[int, int] = (5, 3)

  def _make_pcolormesh_plot(self, data: jax.Array) -> plt.Figure:
    fig, ax = plt.subplots(figsize=self.fig_size)
    pcm = ax.pcolormesh(data, cmap="coolwarm")
    fig.colorbar(pcm, ax=ax)
    return fig

  def on_eval_batches_end(
      self,
      trainer: templates.BaseTrainer,
      eval_metrics: Mapping[str, jax.Array | Mapping[str, jax.Array]],
  ):
    figures = {}
    plot_data = eval_metrics["gen_sample"]

    # Expected to have shape (batch, *spatial_dims, channels). Plots are made
    # for the first batch element and across all times and channels.
    sample = plot_data["gen_sample"]

    for i in range(sample.shape[-1]):
      figures[f"sample_dim{i}"] = self._make_pcolormesh_plot(sample[0, ..., i])

    self.write_images(trainer.train_state.int_step, figures)
