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

"""Trainers for ReFlow models."""

from collections.abc import Callable

from clu import metrics as clu_metrics
import flax
import jax
import optax
from swirl_dynamics.projects.debiasing.rectified_flow import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array = jax.Array
VariableDict = trainers.VariableDict
TrainState = train_states.BasicTrainState


class ReFlowTrainer(
    trainers.BasicTrainer[models.ReFlowModel, TrainState]
):
  """Single-device trainer for rectified flow models."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    train_loss: clu_metrics.Average.from_output("loss")
    train_loss_std: clu_metrics.Std.from_output("loss")

  EvalMetrics = clu_metrics.Collection.create(  # pylint: disable=invalid-name
      **{
          f"eval_time_lvl{i}": clu_metrics.Average.from_output(
              f"time_lvl{i}"
          )
          for i in range(models.ReFlowModel.num_eval_time_levels)
      }
  )

  def initialize_train_state(self, rng: Array) -> TrainState:
    init_vars = self.model.initialize(rng)
    mutables, params = flax.core.pop(init_vars, "params")
    return TrainState.create(
        replicate=self.is_distributed,
        params=params,
        opt_state=self.optimizer.init(params),
        flax_mutables=mutables,
    )

  @property
  def update_train_state(
      self,
  ) -> Callable[[TrainState, VariableDict, VariableDict], TrainState]:
    """Returns function that updates the train state."""

    def _update_train_state(
        train_state: TrainState,
        grads: VariableDict,
        mutables: VariableDict,
    ) -> TrainState:
      updates, new_opt_state = self.optimizer.update(
          grads, train_state.opt_state, train_state.params
      )
      new_params = optax.apply_updates(train_state.params, updates)

      return train_state.replace(
          step=train_state.step + 1,
          opt_state=new_opt_state,
          params=new_params,
          flax_mutables=mutables,
      )

    return _update_train_state

  @staticmethod
  def inference_fn_from_state_dict(
      state: TrainState, *args, **kwargs
  ):
    return models.ReFlowModel.inference_fn(
        state.model_variables, *args, **kwargs
    )


class DistributedReFlowTrainer(
    ReFlowTrainer,
    trainers.BasicDistributedTrainer[models.ReFlowModel, TrainState],
):
  """Multi-device trainer for rectified flow models."""

  # TODO(lzepedanunez): Write a test for this trainer.

  # MRO: ReFlowTrainer > BasicDistributedTrainer > BasicTrainer
  ...
