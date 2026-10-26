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

"""GAN loss functions for the CycleGAN debiasing package.

The loss conventions follow the PyTorch CycleGAN reference implementation
(`GANLoss` in `models/networks.py` of
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
"""

from collections.abc import Callable
import dataclasses
import enum

import jax
import jax.numpy as jnp

Array = jax.Array


class GanMode(enum.StrEnum):
  """Adversarial criteria that can be applied to discriminator outputs."""

  LSGAN = "lsgan"
  BCE_WITH_LOGITS = "bce_with_logits"
  WGANGP = "wgangp"


@dataclasses.dataclass(frozen=True)
class GanLoss:
  """Adversarial criterion `loss(prediction, target_is_real)` for a `GanMode`.

  The criterion is applied to a discriminator prediction map of arbitrary
  shape; the scalar target label is broadcast to that shape and every mode
  returns the mean over the map, i.e. a shape-`()` array:

  * `LSGAN`: mean squared error `mean((p - t)^2)` (Mao et al., 2017).
  * `BCE_WITH_LOGITS`: binary cross-entropy on logits, in the numerically
    stable form `mean((1 - t) p + softplus(-p))`.
  * `WGANGP`: Wasserstein critic loss, `-mean(p)` for real targets and
    `mean(p)` for fake targets; the gradient penalty is a separate term, see
    `gradient_penalty`.

  Attributes:
    gan_mode: The adversarial criterion, as a `GanMode` or its string value.
      Normalized to the enum at construction; unknown modes raise `ValueError`.
    target_real_label: Label of samples from the target distribution. Values
      below 1.0 implement one-sided label smoothing.
    target_fake_label: Label of generated samples.
  """

  gan_mode: GanMode | str
  target_real_label: float = 1.0
  target_fake_label: float = 0.0

  def __post_init__(self):
    """Normalizes `gan_mode` to the enum and rejects unknown modes."""
    mode = self.gan_mode
    if isinstance(mode, str):
      try:
        mode = GanMode(mode)
      except ValueError as err:
        raise ValueError(
            f"Unknown gan_mode {mode!r}; expected one of"
            f" {[m.value for m in GanMode]}."
        ) from err
    if not isinstance(mode, GanMode):
      raise ValueError(
          "gan_mode must be a GanMode or one of"
          f" {[m.value for m in GanMode]}, got {mode!r}."
      )
    object.__setattr__(self, "gan_mode", mode)

  def get_target_tensor(self, prediction: Array, target_is_real: bool) -> Array:
    """Creates the label array matching the shape of `prediction`.

    Args:
      prediction: The prediction map of a discriminator.
      target_is_real: Static Python `bool` (not a traced JAX scalar) indicating
        whether the ground-truth label is the real or the fake label.

    Returns:
      An array of the same shape as `prediction` filled with the target label.
    """
    if target_is_real:
      target_label = self.target_real_label
    else:
      target_label = self.target_fake_label
    return jnp.full_like(prediction, target_label)

  def __call__(self) -> Callable[[Array, bool], Array]:
    """Returns the criterion `loss(prediction, target_is_real)`.

    `target_is_real` must be a static Python `bool` (not a traced JAX scalar).
    """

    def _loss(prediction: Array, target_is_real: bool) -> Array:
      if self.gan_mode == GanMode.LSGAN:
        target = self.get_target_tensor(prediction, target_is_real)
        return jnp.mean(jnp.square(prediction - target))
      elif self.gan_mode == GanMode.BCE_WITH_LOGITS:
        target = self.get_target_tensor(prediction, target_is_real)
        # Stable rewrite of -[t log(sigmoid(p)) + (1 - t) log(1 - sigmoid(p))]
        # as (1 - t) p + softplus(-p), with softplus computed as
        # max(-p, 0) + log(exp(-max(-p, 0)) + exp(-p - max(-p, 0))).
        max_val = jnp.clip(-prediction, 0, None)
        softplus_neg = max_val + jnp.log(
            jnp.exp(-max_val) + jnp.exp(-prediction - max_val)
        )
        return jnp.mean((1.0 - target) * prediction + softplus_neg)
      elif self.gan_mode == GanMode.WGANGP:
        sign = -1.0 if target_is_real else 1.0
        return sign * jnp.mean(prediction)
      else:
        raise ValueError(f"Unsupported gan_mode {self.gan_mode!r}.")

    return _loss


def l1_loss(prediction: Array, target: Array) -> Array:
  """Mean absolute error between `prediction` and `target`.

  Args:
    prediction: Predicted array.
    target: Target array of the same shape as `prediction`.

  Returns:
    The mean of `|prediction - target|` over all entries (a shape-`()` array).

  Raises:
    ValueError: If the two shapes differ.
  """
  if target.shape != prediction.shape:
    raise ValueError(
        f"Target shape {target.shape} must coincide with the prediction shape"
        f" {prediction.shape}."
    )
  return jnp.mean(jnp.abs(prediction - target))


def gradient_penalty(
    critic_fn: Callable[[Array], Array],
    real: Array,
    fake: Array,
    rng: Array,
    eps_stab: float = 1e-12,
) -> Array:
  """Gradient penalty of WGAN-GP (Gulrajani et al., 2017) on interpolates.

  With per-sample interpolates `x_hat = alpha * real + (1 - alpha) * fake`,
  `alpha ~ U(0, 1)`, and the per-sample critic score `s(x_hat)` (the mean of
  the critic's prediction map), the penalty is

    GP = mean_b [ (sqrt(||grad_x s(x_hat_b)||_2^2 + eps_stab) - 1)^2 ].

  The per-sample gradients are obtained in a single backward pass of
  `sum_b s(x_hat_b)`, which equals `vmap(grad(s))` if and only if the critic
  has no cross-sample coupling (true for instance, group or no normalisation;
  false for batch normalisation).

  Args:
    critic_fn: Maps a batch of samples to a prediction map with a leading batch
      axis.
    real: Batch of real samples.
    fake: Batch of generated samples of the same shape (gradients through them
      are stopped).
    rng: Random key for the interpolation coefficients.
    eps_stab: Stabiliser inside the square root, which keeps the gradient of the
      penalty finite where the critic gradient vanishes.

  Returns:
    The scalar penalty.

  Raises:
    ValueError: If `real` and `fake` differ in shape.
  """
  if real.shape != fake.shape:
    raise ValueError(
        f"Real and fake batches must have the same shape, got {real.shape}"
        f" and {fake.shape}."
    )
  alpha = jax.random.uniform(
      rng, (real.shape[0],) + (1,) * (real.ndim - 1), dtype=real.dtype
  )
  interpolates = alpha * real + (1.0 - alpha) * jax.lax.stop_gradient(fake)

  def _summed_scores(x: Array) -> Array:
    scores = critic_fn(x)
    return jnp.sum(jnp.mean(scores, axis=tuple(range(1, scores.ndim))))

  grads = jax.grad(_summed_scores)(interpolates)
  squared_norms = jnp.sum(jnp.square(grads), axis=tuple(range(1, grads.ndim)))
  return jnp.mean(jnp.square(jnp.sqrt(squared_norms + eps_stab) - 1.0))
