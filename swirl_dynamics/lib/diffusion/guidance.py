# Copyright 2023 The swirl_dynamics Authors.
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

"""Modules for a posteriori (post-processing) guidance."""

from collections.abc import Callable
from typing import Protocol

import flax
import jax
import jax.numpy as jnp

Array = jax.Array
DenoiseFn = Callable[[Array, Array], Array]


class Guidance(Protocol):

  def __call__(
      self, denoise_fn: DenoiseFn, guidance_input: Array | None
  ) -> DenoiseFn:
    """Constructs guided denoise function."""
    ...


@flax.struct.dataclass
class InfillFromSlices:
  """N-dimensional infilling guided by slices.

  Example usage::

    # 2D infill given every 8th pixel along both dimensions
    # (assuming that the lead dimension is for batch)
    slices = tuple(slice(None), slice(None, None, 8), slice(None, None, 8))
    sr_guidance = InfillFromSlices(slices)

    # Post process a trained denoiser function via function composition
    # `guidance_input` must have compatible shape s.t.
    # `image[slices] = guidance_input` would not result in errors
    guided_denoiser = sr_guidance(denoiser, guidance_input=jnp.array(0.0))

    # Run guided denoiser the same way as a normal one
    denoised = guided_denoiser(noised, sigma=jnp.array(0.1))

  Attributes:
    slices: The slices of the input to guide denoising (i.e. the rest is
      infilled). The `guidance_input` provided when calling this method must be
      compatible with these slices.
    guide_strength: The strength of the guidance relative to the raw denoiser.
  """

  slices: tuple[slice, ...]
  guide_strength: float = 0.5

  def __call__(self, denoise_fn: DenoiseFn, guidance_input: Array) -> DenoiseFn:
    """Constructs denoise function guided by values on specified slices."""

    def _guided_denoise(x: Array, sigma: Array) -> Array:
      def constraint(xt: Array) -> tuple[Array, Array]:
        denoised = denoise_fn(xt, sigma)
        return jnp.sum((denoised[self.slices] - guidance_input) ** 2), denoised

      constraint_grad, denoised = jax.grad(constraint, has_aux=True)(x)
      # normalize wrt the fraction of values being conditioned
      cond_fraction = jnp.prod(jnp.asarray(x[self.slices].shape)) / jnp.prod(
          jnp.asarray(x.shape)
      )
      guide_strength = self.guide_strength / cond_fraction
      denoised -= guide_strength * constraint_grad
      return denoised.at[self.slices].set(guidance_input)

    return _guided_denoise
