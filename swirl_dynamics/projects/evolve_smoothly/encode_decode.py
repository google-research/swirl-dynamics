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

"""EncodeDecode model and trainer.

This pipeline trains an encoder to output the parameters of a pre-determined
ansatz such that it represents snapshots well. In its current form, the input
snapshots are assumed to be on a equi-spaced periodic grid. During training,
the encoder reconstructs the snapshots by evaluating the ansatz on the same
grid points. The reconstruction loss is minimized under a consistency
regularization.
"""

from collections.abc import Callable
import functools
from typing import Any

from clu import metrics as clu_metrics
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import metrics
from swirl_dynamics.lib.networks import encoders
from swirl_dynamics.projects.evolve_smoothly import ansatzes
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
PyTree = Any


# Helpers from instantiating encoder models
def create_resnet_encoder(ansatz: ansatzes.Ansatz, **kwargs) -> nn.Module:
  return encoders.EncoderResNet(dim_out=ansatz.num_params, **kwargs)


class EncodeDecode(models.BaseModel):
  """Encode-decode training with consistency loss."""

  def __init__(
      self,
      ansatz: ansatzes.Ansatz,
      encoder: nn.Module,
      snapshot_dims: tuple[int, ...],
      consistency_weight: float = 0.0,
  ) -> None:
    self.ansatz = ansatz
    self.encoder = encoder
    self.snapshot_dims = snapshot_dims
    self.consistency_weight = consistency_weight

  def initialize(self, rng: Array) -> models.ModelVariable:
    """Initializes the variables of the encoder."""
    enc_output, enc_vars = self.encoder.init_with_output(
        rng, jnp.ones((1,) + self.snapshot_dims)
    )
    # sanity check: encoder output should be usable as ansatz parameters
    if enc_output.shape[-1] != self.ansatz.num_params:
      raise ValueError(
          f"Encoder output dimension ({enc_output.shape[-1]}) not equal to the"
          f" number of parameters ({self.ansatz.num_params}) in the ansatz!"
      )
    return enc_vars

  def loss_fn(
      self,
      params: PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: PyTree,
  ) -> models.LossAndAux:
    """Computes the reconstruction and consistency loss."""
    del rng
    snapshots, grid = batch["u"], batch["x"]
    encoder_var = dict(params=params, **mutables)
    encoding, mutables = self.encoder.apply(
        encoder_var, snapshots, is_training=True, mutable=list(mutables.keys())
    )
    reconstruction = jax.vmap(self.ansatz.batch_evaluate, in_axes=(0, 0))(
        encoding, grid
    ).reshape(snapshots.shape)
    reencoding, mutables = self.encoder.apply(
        encoder_var,
        reconstruction,
        is_training=True,
        mutable=list(mutables.keys()),
    )
    reconstruction_loss = jnp.mean(jnp.square(reconstruction - snapshots))
    consistency_loss = jnp.mean(jnp.square(reencoding - encoding))
    loss = reconstruction_loss + self.consistency_weight * consistency_loss
    metric = dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        consistency_loss=consistency_loss,
    )
    return loss, (metric, mutables)

  def eval_fn(
      self, variables: PyTree, batch: models.BatchType, rng: Array
  ) -> models.ArrayDict:
    """Evaluates mean, worst-case and std relative l2 errors."""
    del rng
    snapshots, grid = batch["u"], batch["x"]
    encoding = self.encoder.apply(variables, snapshots, is_training=False)
    reconstruction = jax.vmap(self.ansatz.batch_evaluate, in_axes=(0, 0))(
        encoding, grid
    ).reshape(snapshots.shape)
    reencoding = self.encoder.apply(
        variables, reconstruction, is_training=False
    )
    rrmse = functools.partial(
        metrics.mean_squared_error,
        sum_axes=(-1, -2),
        relative=True,
        squared=False,
    )
    return dict(
        reconstruction_rel_l2=rrmse(pred=reconstruction, true=snapshots),
        consistency_rel_l2=rrmse(pred=reencoding, true=encoding),
    )

  @staticmethod
  def inference_fn(
      variables: PyTree, encoder: nn.Module
  ) -> Callable[[ArrayLike], Array]:
    """Returns an encoder inference function."""

    def encode(u: ArrayLike) -> Array:
      return encoder.apply(variables, u, is_training=False)

    return encode


TrainState = train_states.BasicTrainState


class EncodeDecodeTrainer(trainers.BasicTrainer[EncodeDecode, TrainState]):
  """Trainer for the EncodeDecode model."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    train_loss: clu_metrics.Average.from_output("loss")
    train_loss_std: clu_metrics.Average.from_output("loss")
    train_reconstruction_loss: clu_metrics.Average.from_output(
        "reconstruction_loss"
    )
    train_consistency_loss: clu_metrics.Average.from_output("consistency_loss")

  @flax.struct.dataclass
  class EvalMetrics(clu_metrics.Collection):
    eval_reconstruction_rel_l2: clu_metrics.Average.from_output(
        "reconstruction_rel_l2"
    )
    eval_consistency_rel_l2: clu_metrics.Average.from_output(
        "consistency_rel_l2"
    )

  @staticmethod
  def build_inference_fn(
      state: TrainState, encoder: nn.Module
  ) -> Callable[[ArrayLike], Array]:
    """Builds an encoder inference function from a train state."""
    return EncodeDecode.inference_fn(state.model_variables, encoder)
