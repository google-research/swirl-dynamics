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

"""TensorBoard callback for 1D and 2D Improved Mean Flow sample grids."""

from collections.abc import Mapping
import dataclasses

import jax
import matplotlib.pyplot as plt
import numpy as np
from swirl_dynamics import templates


@templates.utils.primary_process_only
@dataclasses.dataclass
class PlotSamples(templates.MatplotlibFigureAsImage):
  """Plots generated 1D line fields or 2D `imshow` grids across `samples_{k}`.

  Attributes:
    fig_size: Matplotlib figure size `(width, height)` in inches.
    color_range: `(vmin, vmax)` range used for 2D `imshow` plots.
    plot_every_n_samples: Stride over the batch axis for selecting samples.
  """

  fig_size: tuple[int, int] = (5, 3)
  color_range: tuple[float, float] = (-2.0, 2.0)
  plot_every_n_samples: int = 1

  def _make_figure(self, sample: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=self.fig_size)
    if sample.ndim == 2:
      # 1D spatial field with shape (L, C).
      for ch in range(sample.shape[-1]):
        ax.plot(sample[:, ch], label=f"ch {ch}")
      if sample.shape[-1] > 1:
        ax.legend(loc="best")
      ax.set_xlabel("grid index")
    elif sample.ndim == 3:
      # 2D spatial field with shape (H, W, C).
      im = ax.imshow(
          sample[..., 0],
          vmin=self.color_range[0],
          vmax=self.color_range[1],
          cmap="coolwarm",
          aspect="auto",
      )
      fig.colorbar(im, ax=ax)
    else:
      plt.close(fig)
      raise ValueError(
          f"Expected 1D `(L, C)` or 2D `(H, W, C)` sample, got {sample.shape}."
      )
    ax.set_title(title)
    return fig

  def on_eval_batches_end(
      self,
      trainer: templates.BaseTrainer,
      eval_metrics: Mapping[str, jax.Array | Mapping[str, jax.Array]],
  ) -> None:
    plot_data = eval_metrics["eval_plot_data"]
    if not isinstance(plot_data, Mapping):
      raise ValueError(
          f"Expected `eval_plot_data` to be a Mapping, got {type(plot_data)}."
      )
    figures = {}
    for key, samples in plot_data.items():
      arr = np.asarray(samples)
      for idx in range(0, arr.shape[0], self.plot_every_n_samples):
        figures[f"{key}_sample_{idx}"] = self._make_figure(
            arr[idx], title=f"{key} (sample {idx})"
        )
    self.write_images(trainer.train_state.int_step, figures)
