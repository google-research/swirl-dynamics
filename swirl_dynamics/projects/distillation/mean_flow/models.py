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

r"""Backwards-compatible re-export shim for graduated MeanFlow models."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import

import flax.linen as nn
import jax.numpy as jnp
from swirl_dynamics.lib.flow_matching.mean_flow import (
    Array,
    ArrayLike,
    ArrayShape,
    CondDict,
    ConditionalMeanFlowModel,
    ConditionalSymmetricMeanFlowModel,
    DTypeLike,
    MeanFlowModel,
    Metrics,
    PyTree,
    SamplingFn,
    ShapeDict,
    VariableDict,
    cond_sample_from_shape,
    logit_normal_dist,
    time_sampler_mean_flow,
)
from swirl_dynamics.projects.distillation.flow_map_matching import backbones


class RescaledMeanFlowUNet(backbones.FlowMapUNet):
  """Rescaled flow map model.

  Attributes:
    time_rescale: Factor for rescaling the time, which normally is in [0, 1] and
      the input to the UNet which is a noise level, which has a much wider range
      of values.
  """

  time_rescale: float = 1.0

  @nn.compact
  def __call__(
      self,
      x_s: Array,
      t: Array,
      s: Array,
      cond: dict[str, Array] | None = None,
      *,
      is_training: bool,
  ) -> Array:
    """Runs rescaled Unet with noise input."""
    if x_s.shape[-1] != self.out_channels:
      raise ValueError(
          f"Number of channels in the input ({x_s.shape[-1]}) must "
          "match the number of channels in the output "
          f"{self.out_channels})."
      )
    if t.ndim != s.ndim:
      raise ValueError(f"Time dimensions must be equal ({t.ndim} != {s.ndim}).")

    if s.ndim < 1:
      s = jnp.broadcast_to(s, (x_s.shape[0],))
      t = jnp.broadcast_to(t, (x_s.shape[0],))

    if s.ndim != 1 or x_s.shape[0] != s.shape[0] or x_s.shape[0] != t.shape[0]:
      raise ValueError(
          "s must be 1D and have the same leading (batch) dimension as x and t."
          f"Instead x_s has leading dimension ({x_s.shape[0]}),"
          f"s has leading dimension ({s.shape[0]}), and "
          f"t has leading dimension ({t.shape[0]})."
      )

    time_s = s * self.time_rescale
    time_t = t * self.time_rescale

    u_theta = super().__call__(
        x_s, time_t, time_s, cond, is_training=is_training
    )

    return u_theta


__all__ = [
    "Array",
    "ArrayLike",
    "ArrayShape",
    "CondDict",
    "ConditionalMeanFlowModel",
    "ConditionalSymmetricMeanFlowModel",
    "DTypeLike",
    "MeanFlowModel",
    "Metrics",
    "PyTree",
    "RescaledMeanFlowUNet",
    "SamplingFn",
    "ShapeDict",
    "VariableDict",
    "cond_sample_from_shape",
    "logit_normal_dist",
    "time_sampler_mean_flow",
]
