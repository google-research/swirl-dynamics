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

"""Trainers for ReFlow models."""
from collections.abc import Callable
import functools

from clu import metrics as clu_metrics
import flax
import jax
import optax
from swirl_dynamics.projects.debiasing.rectified_flow import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array = jax.Array
VariableDict = trainers.VariableDict


class ReFlowTrainState(train_states.BasicTrainState):
  """Train state with an additional field tracking the EMA params."""

  # EMA params is accessed through `ema_state.ema`.
  ema_state: optax.EmaState | None = None

  @property
  def ema_variables(self) -> flax.core.FrozenDict:
    if self.ema_state:
      return flax.core.FrozenDict({"params": self.ema_state.ema})
    else:
      raise ValueError("EMA state is none.")

TrainState = ReFlowTrainState


class ReFlowTrainer(
    trainers.BasicTrainer[models.ReFlowModel, TrainState]
):
  """Single-device trainer for rectified flow models."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    train_loss: clu_metrics.Average.from_output("loss")
    train_loss_std: clu_metrics.Std.from_output("loss")

  @functools.cached_property
  def EvalMetrics(self):
    denoising_metrics = {
        f"eval_time_lvl{i}": clu_metrics.Average.from_output(f"time_lvl{i}")
        for i in range(self.model.num_eval_time_levels)
    }
    return clu_metrics.Collection.create(**denoising_metrics)

  def __init__(self, ema_decay: float, *args, **kwargs):
    """Initializes the trainer along the ema state."""
    self.ema = optax.ema(ema_decay)
    super().__init__(*args, **kwargs)

  def initialize_train_state(self, rng: Array) -> TrainState:
    init_vars = self.model.initialize(rng)
    mutables, params = flax.core.pop(init_vars, "params")
    return TrainState.create(
        replicate=self.is_distributed,
        params=params,
        opt_state=self.optimizer.init(params),
        flax_mutables=mutables,
        ema_state=self.ema.init(params),
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
      _, new_ema_state = self.ema.update(new_params, train_state.ema_state)
      return train_state.replace(
          step=train_state.step + 1,
          opt_state=new_opt_state,
          params=new_params,
          flax_mutables=mutables,
          ema_state=new_ema_state,
      )

    return _update_train_state

  @staticmethod
  def inference_fn_from_state_dict(
      state: TrainState, *args, use_ema: bool = True, **kwargs
  ):
    if use_ema:
      if isinstance(state.ema_state, dict):
        variables = flax.core.FrozenDict({"params": state.ema_state["ema"]})
      else:
        variables = state.ema_variables
    else:
      variables = state.model_variables
    return models.ReFlowModel.inference_fn(variables, *args, **kwargs)


class DistributedReFlowTrainer(
    ReFlowTrainer,
    trainers.BasicDistributedTrainer[models.ReFlowModel, TrainState],
):
  """Multi-device trainer for rectified flow models."""

  # TODO: Write a test for this trainer.
