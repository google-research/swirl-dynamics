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

"""Modules for guidance transforms for denoising functions."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import flax
import jax
import jax.numpy as jnp

Array = jax.Array
PyTree = Any
ArrayMapping = Mapping[str, Array]
DenoiseFn = Callable[[Array, Array, ArrayMapping | None], Array]


class Transform(Protocol):
  """Transforms a denoising function to follow some guidance.

  One may think of these transforms as instances of Python decorators,
  specifically made for denoising functions. Each transform takes a base
  denoising function and extends it (often using some additional data) to build
  a new denoising function with the same interface.
  """

  def __call__(
      self, denoise_fn: DenoiseFn, guidance_inputs: ArrayMapping
  ) -> DenoiseFn:
    """Constructs a guided denoising function.

    Args:
      denoise_fn: The base denoising function.
      guidance_inputs: A dictionary containing inputs used to construct the
        guided denoising function. Note that all transforms *share the same
        input dict*, therefore all transforms should use different fields from
        this dict (unless absolutely intended) to avoid potential name clashes.

    Returns:
      The guided denoising function.
    """
    ...


@flax.struct.dataclass
class InfillFromSlices:
  """N-dimensional infilling guided by known values on slices.

  Example usage::

    # 2D infill given every 8th pixel along both dimensions (assuming that the
    # lead dimension is for batch).
    slices = tuple(slice(None), slice(None, None, 8), slice(None, None, 8))
    sr_guidance = InfillFromSlices(slices, guide_strength=0.1)

    # Post-process a trained denoiser function via function composition.
    # The `observed_slices` arg must have compatible shape such that
    # `image[slices] = observed_slices` would not raise errors.
    guided_denoiser = sr_guidance(denoiser, {"observed_slices": jnp.array(0.0)})

    # Run guided denoiser the same way as a normal one
    denoised = guided_denoiser(noised, sigma=jnp.array(0.1), cond=None)

  Attributes:
    slices: The slices of the input to guide denoising (i.e. the rest is being
      infilled).
    guide_strength: The strength of the guidance relative to the raw denoiser.
      It will be rescaled based on the fraction of values being conditioned.
  """

  slices: tuple[slice, ...]
  guide_strength: float = 0.5

  def __call__(
      self, denoise_fn: DenoiseFn, guidance_inputs: ArrayMapping
  ) -> DenoiseFn:
    """Constructs denoise function guided by values on specified slices."""

    def _guided_denoise(
        x: Array, sigma: Array, cond: ArrayMapping | None = None
    ) -> Array:
      def constraint(xt: Array) -> tuple[Array, Array]:
        denoised = denoise_fn(xt, sigma, cond)
        error = jnp.sum(
            (denoised[self.slices] - guidance_inputs["observed_slices"]) ** 2
        )
        return error, denoised

      constraint_grad, denoised = jax.grad(constraint, has_aux=True)(x)
      # Rescale based on the fraction of values being conditioned.
      cond_fraction = jnp.prod(jnp.asarray(x[self.slices].shape)) / jnp.prod(
          jnp.asarray(x.shape)
      )
      guide_strength = self.guide_strength / cond_fraction
      denoised -= guide_strength * constraint_grad
      return denoised.at[self.slices].set(guidance_inputs["observed_slices"])

    return _guided_denoise


@flax.struct.dataclass
class ClassifierFreeHybrid:
  """Classifier-free guidance for a hybrid (cond/uncond) denoising model.

  This guidance technique, introduced by Ho and Salimans
  (https://arxiv.org/abs/2207.12598), aims to improve the quality of denoised
  images by combining conditional and unconditional denoising outputs.
  The guided denoise function is given by:

    D̃(x, σ, c) = (1 + w) * D(x, σ, c) - w * D(x, σ, Ø),

  where

    - x: The noisy input.
    - σ: The noise level.
    - c: The conditioning information (e.g., class label).
    - Ø: A special masking condition (typically zeros) indicating unconditional
      denoising.
    - w: The guidance strength, controlling the influence of each denoising
      output. A value of 0 indicates non-guided denoising.

  Attributes:
    guidance_strength: The strength of guidance (i.e. w). The original paper
      reports optimal values of 0.1 and 0.3 for 64x64 and 128x128 ImageNet
      respectively.
    cond_mask_keys: A collection of keys in the conditions dictionary that will
      be masked. If `None`, all conditions are masked.
    cond_mask_value: The values that the conditions will be masked by. This
      value must be consistent with the masking applied at training.
  """

  guidance_strength: float = 0.0
  cond_mask_keys: Sequence[str] | None = None
  cond_mask_value: float = 0.0

  def __call__(
      self, denoise_fn: DenoiseFn, guidance_inputs: ArrayMapping
  ) -> DenoiseFn:
    """Constructs denoise function with classifier free guidance."""

    def _guided_denoise(x: Array, sigma: Array, cond: ArrayMapping) -> Array:
      masked_cond = {
          k: (
              v  # pylint: disable=g-long-ternary
              if self.cond_mask_keys is not None
              and k not in self.cond_mask_keys
              else jnp.ones_like(v) * self.cond_mask_value
          )
          for k, v in cond.items()
      }
      uncond_denoised = denoise_fn(x, sigma, masked_cond)
      return (1 + self.guidance_strength) * denoise_fn(
          x, sigma, cond
      ) - self.guidance_strength * uncond_denoised

    return _guided_denoise
