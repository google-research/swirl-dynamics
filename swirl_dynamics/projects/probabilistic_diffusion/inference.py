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

"""Inference modules."""

from collections.abc import Sequence
import dataclasses
from typing import Any, Protocol

import flax.linen as nn
import jax
import numpy as np
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.lib.diffusion import samplers
from swirl_dynamics.projects.probabilistic_diffusion import trainers

Array = jax.Array
PyTree = Any


class CondSampler(Protocol):
  """The conditional sampler interface."""

  def __call__(
      self, num_samples: int, rng: Array, cond: PyTree, guidance_inputs: PyTree
  ) -> Array:
    ...


class PreprocTransform(Protocol):
  """The pre-processing transform interface."""

  def __call__(
      self, cond: PyTree, guidance_inputs: PyTree
  ) -> tuple[PyTree, PyTree]:
    ...


@dataclasses.dataclass(frozen=True)
class StandardizeCondField:
  """Standardizes a field in the condictioning dict."""

  cond_field: str
  mean: np.ndarray | Array
  std: np.ndarray | Array

  def __call__(
      self, cond: PyTree, guidance_inputs: PyTree
  ) -> tuple[PyTree, PyTree]:
    cond[self.cond_field] = (cond[self.cond_field] - self.mean) / self.std
    return cond, guidance_inputs


class PostprocTransform(Protocol):
  """The post-processing transform interface."""

  def __call__(self, cond_samples: Array) -> Array:
    ...


@dataclasses.dataclass(frozen=True)
class RescaleSamples:
  """Rescales the samples linearly (inverse standardization)."""

  mean: np.ndarray | Array
  std: np.ndarray | Array

  def __call__(self, cond_samples: Array) -> Array:
    return cond_samples * self.std + self.mean


def chain(
    cond_sampler: CondSampler,
    preprocessors: Sequence[PreprocTransform] = (),
    postprocessors: Sequence[PostprocTransform] = (),
) -> CondSampler:
  """Chains a conditional sampler together with pre- and post-processors."""

  def _cond_sample(
      num_samples: int, rng: Array, cond: PyTree, guidance_inputs: PyTree
  ) -> Array:
    for preproc_fn in preprocessors:
      cond, guidance_inputs = preproc_fn(cond, guidance_inputs)
    cond_samples = cond_sampler(num_samples, rng, cond, guidance_inputs)
    for postproc_fn in postprocessors:
      cond_samples = postproc_fn(cond_samples)
    return cond_samples

  return _cond_sample


def get_trained_denoise_fn(
    denoiser: nn.Module,
    ckpt_dir: str,
    step: int | None = None,
    use_ema: bool = True,
) -> samplers.DenoiseFn:
  """Loads checkpoint and creates an inference denoising function."""
  return trainers.DenoisingTrainer.inference_fn_from_state_dict(
      state=trainers.DenoisingModelTrainState.restore_from_orbax_ckpt(
          ckpt_dir, step=step
      ),
      denoiser=denoiser,
      use_ema=use_ema,
  )


def get_inference_fn_from_sampler(
    sampler: samplers.Sampler, **kwargs
) -> CondSampler:
  """Creates a conditional sampling inference function for evaluation."""

  def _gen_cond_sample(
      num_samples: int, rng: Array, cond: PyTree, guidance_inputs: PyTree
  ) -> Array:
    samples, _ = sampler.generate(
        num_samples, rng, cond=cond, guidance_inputs=guidance_inputs, **kwargs
    )
    return samples

  return _gen_cond_sample


# ****************
# Helpers
# ****************


def read_stats_from_hdf5(
    file_path: str, variables: Sequence[str], group: str = "mean"
):
  """Reads statistics from a hdf5 file and applies a 90 degree rotation."""
  variables = [f"{group}/{v}" for v in variables]
  arrays = hdf5_utils.read_arrays_as_tuple(file_path, keys=variables)
  out = np.stack(arrays, axis=-1)
  return np.rot90(out, k=1, axes=(0, 1))
