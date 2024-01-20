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

"""Define different GAN loss functions.

This code is inspired by the pytorch code in
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/networks.py#L210
"""

import dataclasses
import enum
from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


class GanMode(enum.Enum):
  LSGAN = "lsgan"
  BCE_WITH_LOGITS = "bce_with_logits"
  WGANGP = "wgangp"


@dataclasses.dataclass(frozen=True)
class GanLoss:
  """Define different GAN objectives.

  Attributes:
    gan_mode: Specifies the type of loss function to be used.
    target_real_label: the label for data in the target distribution that will
      be compared against.
    target_fake_label: label of the data out of the target distribution.

  The GANLoss class abstracts away the need to create the target label tensor
  that has the same size as the input.
  """
  gan_mode: GanMode
  target_real_label: float = 1.0
  target_fake_label: float = 0.0

  def setup(self):
    if self.gan_mode not in [
        GanMode.LSGAN,
        GanMode.BCE_WITH_LOGITS,
        GanMode.WGANGP,
    ]:
      raise NotImplementedError(
          "gan loss %s is not implemented" % self.gan_mode
      )

  def get_target_tensor(self, prediction: Array, target_is_real: bool) -> Array:
    """Create label arrays with the same size as the input.

    Args:
      prediction: The prediction from a discriminator.
      target_is_real: if the ground truth label is for real images or fake
        images

    Returns:
        A label array filled with ground truth label with the same shape as the
        input.
    """
    if target_is_real:
      target_label = self.target_real_label
    else:
      target_label = self.target_fake_label
    target = jnp.ones_like(a=prediction)
    return target_label * target

  def __call__(self) -> Callable[[Array, bool], Array]:

    def _loss(prediction: Array, target_is_real: bool) -> Array:
      # Creating the target labels.
      target_array = self.get_target_tensor(prediction, target_is_real)

      # Dummy value of the loss.
      loss_value = jnp.array([-1.0])

      if self.gan_mode in [GanMode.LSGAN, GanMode.BCE_WITH_LOGITS]:
        if target_array.shape != prediction.shape:
          raise ValueError(
              f"Target shapes ({target_array.shape}) must coincide with the",
              f" input shapes ({prediction.shape})",
          )
        # Using Mean Square Error (MSE) loss.
        if self.gan_mode == GanMode.LSGAN:
          loss_value = jnp.mean((prediction - target_array) ** 2)

        # Using BCEWithLogitsLoss.
        elif self.gan_mode == GanMode.BCE_WITH_LOGITS:
          max_val = jnp.clip(-prediction, 0, None)
          loss_value = (
              prediction * (1 - target_array)
              + max_val
              + jnp.log(jnp.exp(-max_val) + jnp.exp((-prediction - max_val)))
          )
          loss_value = jnp.mean(loss_value)  # default to mean loss

      # Using the loss by Wasserstein GAN.
      elif self.gan_mode == GanMode.WGANGP:
        if target_is_real:
          loss_value = -jnp.mean(prediction)
        else:
          loss_value = jnp.mean(prediction)

      return loss_value

    return _loss


def l1_loss(prediction: Array, target_array: Array) -> Array:
  if target_array.shape != prediction.shape:
    raise ValueError(
        f"Target shapes ({target_array.shape}) must coincide with the input",
        f" shapes ({prediction.shape})",
    )
  absolute = jnp.abs(target_array - prediction)
  return jnp.mean(absolute)
