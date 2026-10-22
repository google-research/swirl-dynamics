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

r"""Backwards-compatible re-export shim for stochastic interpolant flow models."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

import flax.linen as nn
import jax.numpy as jnp
from swirl_dynamics.lib.flow_matching.models import (
    Array,
    ArrayShape,
    CondDict,
    ConditionalStochasticInterpolantModel,
    Metrics,
    PyTree,
    RescaledUnet3d,
    ShapeDict,
    StochasticInterpolantCRPSModel,
    StochasticInterpolantFlowScoreModel,
    StochasticInterpolantLossFn,
    StochasticInterpolantModel,
    TimeRescaled,
    VariableDict,
    cond_sample_from_shape,
)
from swirl_dynamics.projects.debiasing.stochastic_interpolants import backbones


class RescaledUnet(backbones.UNet):
  """Rescaled flow model.

  Attributes:
    time_rescale: Factor for rescaling the time, which normally is in [0, 1] and
      the input to the UNet which is a noise level, which has a much wider range
      of values.
  """

  time_rescale: float = 1000.0

  @nn.compact
  def __call__(
      self,
      x: Array,
      sigma: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Runs rescaled Unet with noise input."""
    if x.shape[-1] != self.out_channels:
      raise ValueError(
          f"Number of channels in the input ({x.shape[-1]}) must "
          "match the number of channels in the output "
          f"{self.out_channels})."
      )

    if sigma.ndim < 1:
      sigma = jnp.broadcast_to(sigma, (x.shape[0],))

    if sigma.ndim != 1 or x.shape[0] != sigma.shape[0]:
      raise ValueError(
          "sigma must be 1D and have the same leading (batch) dimension as x"
          f" ({x.shape[0]})!"
      )

    time = sigma * self.time_rescale

    f_x = super().__call__(x, time, cond, is_training=is_training)

    return f_x


__all__ = [
    "Array",
    "ArrayShape",
    "CondDict",
    "ConditionalStochasticInterpolantModel",
    "Metrics",
    "PyTree",
    "RescaledUnet",
    "RescaledUnet3d",
    "ShapeDict",
    "StochasticInterpolantCRPSModel",
    "StochasticInterpolantFlowScoreModel",
    "StochasticInterpolantLossFn",
    "StochasticInterpolantModel",
    "TimeRescaled",
    "VariableDict",
    "cond_sample_from_shape",
]
