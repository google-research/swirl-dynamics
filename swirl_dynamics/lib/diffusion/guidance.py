# Copyright 2025 The swirl_dynamics Authors.
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
from typing import Any, Literal, Protocol

import chex
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
  guide_strength: chex.Numeric = 0.5

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
            (denoised[self.slices] - guidance_inputs['observed_slices']) ** 2
        )
        return error, denoised

      constraint_grad, denoised = jax.grad(constraint, has_aux=True)(x)
      # Rescale based on the fraction of values being conditioned.
      cond_fraction = jnp.prod(jnp.asarray(x[self.slices].shape)) / jnp.prod(
          jnp.asarray(x.shape)
      )
      guide_strength = self.guide_strength / cond_fraction
      denoised -= guide_strength * constraint_grad
      return denoised.at[self.slices].set(guidance_inputs['observed_slices'])

    return _guided_denoise


@flax.struct.dataclass
class InterlockingFrames:
  """Condition on the first and last frame to be equal in a short trajectory.

  The main point of this guidance term is to interlock contigous temporal chunks
  into a larger one by imposing boundary conditions at the interfaces of the
  chunks. Each chunk is a short temporal sequence produced by a diffusion model
  with a given batch size. To ensure the spatio-temporal coherence the boundary
  conditions are imposed at each step of the evolution of the SDE in diffusion
  time.

  For a target number of frames to generate, total_num_frames, we decompose the
  total sequence in several chunks (num_chunks), whose number is given by the
  temporal generation length of the backbone model (minus the overlap).
  Then the batch dimension of the backbone diffusion model is fixed to
  num_chunks. Thus we generate num_chunks of short trajectories simultaneously,
  and we concatenate them (removing the overlapping regions).

  We can generate more than one long sequence, thus that input of the guidence
  is a 5-tensor with dimensions:
  (batch_size, num_chunks, num_frames_per_chunk, height, width, channels).

  Attributes:
    guide_strength: Strength of the conditioning relative to unconditioned
      score.
    style: How the boundaries are imposed following either "mean" or "swap". For
      "mean" we compute the mean between the overlap of two adjacent chunks,
      whereas for swap we exchange the values within the overlapping region. The
      rationale for the second is to not change the statistics by averaging two
      Gaussians.
    overlap_length: The length of the overlap which we impose to be the same
      across the boundaries.
  """

  guide_strength: chex.Numeric = 0.5
  style: Literal['average', 'swap'] = 'average'
  overlap_length: int = 1

  def __call__(
      self, denoise_fn: DenoiseFn, guidance_inputs: ArrayMapping | None = None
  ) -> DenoiseFn:
    """Constructs denoise function conditioned on overlaping values."""

    def _guided_denoise(
        x: Array, sigma: Array, cond: ArrayMapping | None = None
    ) -> Array:
      """Guided denoise function.

      Args:
        x: The tensor to denoise with dims (num_trajectories, num_chunks_traj,
          num_frames_per_chunk, height, width, channels).
        sigma: The noise level.
        cond: The dictionary with the conditioning.

      Returns:
        An estimate of the denoised tensor guided by the interlocking
        constraint.
      """
      if x.ndim != 6:
        raise ValueError(
            f'Invalid input dimension: {x.shape}, a 6-tensor is expected.'
        )

      def constraint(xt: Array) -> tuple[Array, Array]:
        denoised = denoise_fn(xt, sigma, cond)
        return (
            jnp.sum(
                (
                    denoised[:, 1:, : self.overlap_length]
                    - denoised[:, :-1, -self.overlap_length :]
                )
                ** 2
            ),
            denoised,
        )

      constraint_grad, denoised = jax.grad(constraint, has_aux=True)(x)
      denoised -= self.guide_strength * constraint_grad

      # Interchanging information at each side of the interface.
      if self.style not in ['average', 'swap']:
        raise ValueError(
            f'Invalid style: {self.style}. Expected either"average" or "swap".'
        )
      elif self.style == 'swap':
        cond_value_first = denoised[:, 1:, : self.overlap_length]
        denoised = denoised.at[:, 1:, : self.overlap_length].set(
            denoised[:, :-1, -self.overlap_length :]
        )
        denoised = denoised.at[:, :-1, -self.overlap_length :].set(
            cond_value_first
        )

      # Average the values at each side of the interface.
      elif self.style == 'average':
        average_value = 0.5 * (
            denoised[:, 1:, : self.overlap_length]
            + denoised[:, :-1, -self.overlap_length :]
        )
        denoised = denoised.at[:, 1:, : self.overlap_length].set(average_value)
        denoised = denoised.at[:, :-1, -self.overlap_length :].set(
            average_value
        )

      return denoised

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

  guidance_strength: chex.Numeric = 0.0
  cond_mask_keys: Sequence[str] | None = None
  cond_mask_value: chex.Numeric = 0.0

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


@flax.struct.dataclass
class ClassifierFreeContrastive:
  """Contrastive classifier-free guidance.

  This technique adapts the principles of classifier-free guidance (Ho and
  Salimans) to provide more nuanced control over the generation process. Instead
  of combining conditional and unconditional outputs, it contrasts a target
  conditioning with another user-supplied conditioning. This allows the model to
  actively steer the generation towards desired attributes while simultaneously
  moving away from undesired ones.

  The guided denoise function is given by:

    D̃(x, σ, c_pos, c_neg) = (1 + w) * D(x, σ, c_pos) - w * D(x, σ, c_neg)

  where

    - x: The noisy input.
    - σ: The noise level.
    - c_pos: The positive conditioning information, representing the target
      attributes for the generation.
    - c_neg: The negative conditioning information, representing attributes
      to avoid or steer away from.
    - w: The guidance strength or contrast scale. It controls how strongly
      the generation is pushed towards c_pos and away from c_neg.
      When w > 0, the output emphasizes characteristics of c_pos that
      are distinct from c_neg. A value of 0 results in
      D̃(x, σ, c_pos, c_neg) = D(x, σ, c_pos) (i.e., standard conditional
      denoising based on the positive input only, with no contrastive effect).

  Note that this transform can be used to implement vanilla classifier-free
  guidance by setting c_neg = jnp.zeros_like(c_pos). However, it supports one
  contrastive condition only.

  Attributes:
    guidance_strength: The strength of guidance (i.e. w).
    cond_key: The key in the conditions dictionary that is contrasted, i.e. the
      corresponding key will be used as the positive condition.
    contrast_cond_key: The key in the guidance inputs dictionary whose value
      will be used as the negative condition.
  """

  guidance_strength: chex.Numeric = 0.0
  cond_key: str = ''
  contrast_cond_key: str = ''

  def __call__(
      self, denoise_fn: DenoiseFn, guidance_inputs: ArrayMapping
  ) -> DenoiseFn:
    """Constructs denoise function with classifier free guidance."""

    def _guided_denoise(x: Array, sigma: Array, cond: ArrayMapping) -> Array:
      contrast_cond_value = jnp.stack(
          [guidance_inputs[self.contrast_cond_key]] * x.shape[0], axis=0
      )
      if contrast_cond_value.shape != cond[self.cond_key].shape:
        raise ValueError(
            f'Contrast cond value shape {contrast_cond_value.shape} does not'
            f' match cond key shape {cond[self.cond_key].shape}.'
        )
      contrastive_cond = {
          k: v if k != self.cond_key else contrast_cond_value
          for k, v in cond.items()
      }
      constrastive_denoised = denoise_fn(x, sigma, contrastive_cond)
      return (1 + self.guidance_strength) * denoise_fn(
          x, sigma, cond
      ) - self.guidance_strength * constrastive_denoised

    return _guided_denoise
