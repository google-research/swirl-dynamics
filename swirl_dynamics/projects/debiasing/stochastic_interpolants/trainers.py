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

"""Trainers for Stochastic Interpolants models.

This scripts defines the trainers for the stochastic interpolants models, which
are defined following the templates in `swirl_dynamics.templates.trainers`.
They define the training and evaluation metrics based on the losses defined in
`swirl_dynamics.projects.debiasing.stochastic_interpolants.models`.

This script also defines the train state for the stochastic interpolants models,
which is a wrapper around the `BasicTrainState` defined in
`swirl_dynamics.templates.train_states`.
"""

from collections.abc import Callable
import functools
from typing import TypeAlias, TypeVar

from clu import metrics as clu_metrics
import flax
import jax
import optax
from swirl_dynamics.projects.debiasing.stochastic_interpolants import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array: TypeAlias = jax.Array
VariableDict: TypeAlias = trainers.VariableDict
Collection: TypeAlias = type[clu_metrics.Collection]


class StochasticInterpolantTrainState(train_states.BasicTrainState):
  """Train state with an additional field tracking the EMA params."""

  # EMA params is accessed through `ema_state.ema`.
  ema_state: optax.EmaState | None = None

  @property
  def ema_variables(self) -> flax.core.FrozenDict:
    if self.ema_state:
      return flax.core.FrozenDict({"params": self.ema_state.ema})
    else:
      raise ValueError("EMA state is none.")


TrainState: TypeAlias = StochasticInterpolantTrainState
Model = TypeVar("Model", bound=models.StochasticInterpolantModel)
State = TypeVar("State", bound=TrainState)


class StochasticInterpolantTrainer(trainers.BasicTrainer[Model, State]):
  """Single-device trainer for stochastic interpolants models."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    train_loss: clu_metrics.Average.from_output("loss")
    train_loss_std: clu_metrics.Std.from_output("loss")

  @functools.cached_property
  def EvalMetrics(self) -> Collection:
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

  @classmethod
  def inference_fn_from_state_dict(
      cls, state: TrainState, *args, use_ema: bool = True, **kwargs
  ):
    if use_ema:
      if isinstance(state.ema_state, dict):
        variables = flax.core.FrozenDict({"params": state.ema_state["ema"]})
      else:
        variables = state.ema_variables
    else:
      variables = state.model_variables
    return models.StochasticInterpolantModel.inference_fn(
        variables, *args, **kwargs
    )


class StochasticInterpolantFlowScoreTrainer(StochasticInterpolantTrainer):
  """Single-device trainer for stochastic interpolants score models.

  This trainer inherits most of its functionality from the
  `StochasticInterpolantTrainer` but adds additional metrics for the score and
  the flow. The metrics are defined in the `TrainMetrics` class.
  """

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    train_loss: clu_metrics.Average.from_output("loss")
    train_loss_std: clu_metrics.Std.from_output("loss")
    train_loss_flow: clu_metrics.Average.from_output("loss_flow")
    train_loss_flow_std: clu_metrics.Std.from_output("loss_flow")
    train_loss_score: clu_metrics.Average.from_output("loss_score")
    train_loss_score_std: clu_metrics.Std.from_output("loss_score")


class DistributedStochasticInterpolantTrainer(
    StochasticInterpolantTrainer,
    trainers.BasicDistributedTrainer[
        models.StochasticInterpolantModel, TrainState
    ],
):
  """Multi-device trainer for rectified flow models."""

  # TODO: Write a test for this trainer.


class DistributedStochasticInterpolantFlowScoreTrainer(
    StochasticInterpolantFlowScoreTrainer,
    trainers.BasicDistributedTrainer[
        models.StochasticInterpolantModel, TrainState
    ],
):
  """Multi-device trainer for stochastic interpolants score models."""

  # TODO: Write a test for this trainer.
