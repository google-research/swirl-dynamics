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

"""BatchDecode model and trainer.

This training pipeline is primarily used to validate the expressive power of a
selected ansatz before encode-decode training. We create a number of ansatz
instances and train them on different snapshots concurrently using jax
vectorization.

While low errors are promising signs that the selected ansatz may have desirable
expressive power, this estimate can be overly optimistic since using an encoder
to compute the ansatz weights brings additional constraints (e.g. continuity),
which are expected to lower the reconstruction accuracy.
"""

from typing import Any

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import metrics
from swirl_dynamics.projects.evolve_smoothly import ansatzes
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers

PyTree = Any


class BatchDecode(models.BaseModel):
  """Simultaneously trains a bundle of ansatz on different snapshots."""

  def __init__(self, ansatz: ansatzes.Ansatz, num_snapshots: int) -> None:
    self.ansatz = ansatz
    self.num_snapshots = num_snapshots

  def initialize(self, rng: jax.Array) -> models.ModelVariable:
    """Initializes the variables of the ansatz."""
    return jax.vmap(self.ansatz.model.init, in_axes=(0, None))(
        jax.random.split(rng, self.num_snapshots),
        jnp.ones((1, self.ansatz.input_dim)),
    )

  def loss_fn(
      self,
      params: PyTree,
      batch: models.BatchType,
      rng: jax.Array,
      mutables: PyTree,
  ) -> models.LossAndAux:
    """Computes the l2 reconstruction loss."""
    del rng
    ypred = jax.vmap(self.ansatz.batch_evaluate, in_axes=(0, None), out_axes=1)(
        {"params": params}, batch["x"]
    )
    loss = jnp.mean(jnp.square(batch["u"] - ypred))
    return loss, ({"loss": loss}, mutables)

  def eval_fn(
      self, variables: PyTree, batch: models.BatchType, rng: jax.Array
  ) -> models.ArrayDict:
    """Evaluates mean, worst-case and std relative l2 errors."""
    del rng
    ypred = jax.vmap(self.ansatz.batch_evaluate, in_axes=(0, None), out_axes=1)(
        variables, batch["x"]
    )
    rrmse = metrics.mean_squared_error(
        pred=ypred,
        true=batch["u"],
        sum_axes=(-1, -2),
        relative=True,
        squared=False,
    )
    return dict(rrmse=rrmse)


class BatchDecodeTrainer(trainers.BasicTrainer):
  """Trainer for the AnsatzDecode model."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    train_loss: clu_metrics.Average.from_output("loss")
    train_loss_std: clu_metrics.Average.from_output("loss")

  @flax.struct.dataclass
  class EvalMetrics(clu_metrics.Collection):
    eval_rrmse_mean: clu_metrics.Average.from_output("rrmse")
    eval_rrmse_std: clu_metrics.Std.from_output("rrmse")
