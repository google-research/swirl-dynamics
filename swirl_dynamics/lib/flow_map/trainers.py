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

"""Unified EMA train state and trainers for flow-map and mean-flow models."""

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, TypeVar, cast

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
import optax
from swirl_dynamics.lib.flow_map import flow_map
from swirl_dynamics.lib.flow_map import mean_flow
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array: TypeAlias = jax.Array
VariableDict: TypeAlias = trainers.VariableDict
Collection: TypeAlias = type[clu_metrics.Collection]
ModelT = TypeVar("ModelT", bound=models.BaseModel)


@flax.struct.dataclass
class _Std(clu_metrics.Std):
  """`clu.metrics.Std` avoiding `jnp.clip(..., a_min=...)` on PyPI `clu`."""

  def compute(self) -> Array:
    mean = self.total / self.count
    variance = jnp.clip(self.sum_of_squares / self.count - mean**2, min=0.0)
    return jnp.sqrt(variance)


class FlowMapTrainState(train_states.BasicTrainState):
  """Train state tracking EMA parameters alongside mutable Flax collections."""

  ema_state: optax.EmaState | Mapping[str, Any] | None = None

  @property
  def ema_variables(self) -> flax.core.FrozenDict:
    """Returns frozen EMA variable dict preserving `flax_mutables`."""
    if self.ema_state is None:
      raise ValueError("EMA state is none.")
    ema_params = (
        self.ema_state["ema"]
        if isinstance(self.ema_state, Mapping)
        else self.ema_state.ema
    )
    return flax.core.freeze(dict(params=ema_params, **self.flax_mutables))


MeanFlowTrainState: TypeAlias = FlowMapTrainState
TrainState: TypeAlias = FlowMapTrainState


class EmaFlowMapTrainer(trainers.BasicTrainer[ModelT, FlowMapTrainState]):
  """Shared single-device EMA trainer base for flow-map and mean-flow models.

  Subclasses customize logged metrics by overriding `_build_train_metrics` and
  `_build_eval_metrics`, and implement `inference_fn_from_state_dict`.
  """

  def __init__(self, ema_decay: float, *args: Any, **kwargs: Any):
    """Initializes the EMA state transform and metric collections."""
    self.ema = optax.ema(ema_decay)
    super().__init__(*args, **kwargs)
    # Assign dynamically to match `templates.trainers.BaseTrainer` conventions.
    setattr(self, "TrainMetrics", self._build_train_metrics())
    setattr(self, "EvalMetrics", self._build_eval_metrics())

  def _build_train_metrics(self) -> Collection:
    """Builds the training `clu.metrics.Collection` class."""
    return clu_metrics.Collection.create(
        train_loss=clu_metrics.Average.from_output("loss"),
        train_loss_std=_Std.from_output("loss"),
    )

  def _build_eval_metrics(self) -> Collection:
    """Builds the evaluation `clu.metrics.Collection` class."""
    return clu_metrics.Collection.create(
        eval_loss=clu_metrics.Average.from_output("eval_loss"),
    )

  def initialize_train_state(self, rng: Array) -> FlowMapTrainState:
    """Initializes model variables, optimizer state, and EMA state."""
    init_vars = cast(
        flax.core.FrozenDict | dict[str, Any], self.model.initialize(rng)
    )
    mutables, params = flax.core.pop(init_vars, "params")
    return FlowMapTrainState.create(
        replicate=self.is_distributed,
        params=params,
        opt_state=self.optimizer.init(params),
        flax_mutables=mutables,
        ema_state=self.ema.init(params),
    )

  @property
  def update_train_state(
      self,
  ) -> Callable[
      [FlowMapTrainState, VariableDict, VariableDict], FlowMapTrainState
  ]:
    """Returns function updating parameters, optimizer, mutables, and EMA."""

    def _update_train_state(
        train_state: FlowMapTrainState,
        grads: VariableDict,
        mutables: VariableDict,
    ) -> FlowMapTrainState:
      updates, new_opt_state = self.optimizer.update(
          grads, train_state.opt_state, train_state.params
      )
      new_params = optax.apply_updates(train_state.params, updates)
      if not isinstance(train_state.ema_state, optax.EmaState):
        raise TypeError(
            "Expected `train_state.ema_state` to be `optax.EmaState`, got "
            f"{type(train_state.ema_state)!r}."
        )
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
  def _variables_from_state(
      state: FlowMapTrainState, *, use_ema: bool = True
  ) -> flax.core.FrozenDict:
    """Extracts EMA or raw model variables (including mutables) from `state`."""
    return state.ema_variables if use_ema else state.model_variables


class LagrangianFlowMapTrainer(
    EmaFlowMapTrainer[flow_map.LagrangianFlowMapModel]
):
  """Single-device EMA trainer for Lagrangian flow-map models."""

  def _build_eval_metrics(self) -> Collection:
    return clu_metrics.Collection.create(
        eval_loss=clu_metrics.Average.from_output("eval_loss"),
        eval_plot_data=clu_metrics.CollectingMetric.from_outputs(
            tuple(
                f"{num_steps}" for num_steps in self.model.number_of_eval_steps
            )
        ),
    )

  @classmethod
  def inference_fn_from_state_dict(
      cls,
      state: FlowMapTrainState,
      *args: Any,
      use_ema: bool = True,
      **kwargs: Any,
  ) -> Callable[..., Array]:
    """Builds a forward flow-map inference callable from `state`."""
    variables = cls._variables_from_state(state, use_ema=use_ema)
    return flow_map.LagrangianFlowMapModel.inference_fn(
        variables, *args, **kwargs
    )


class DistributedLagrangianFlowMapTrainer(
    LagrangianFlowMapTrainer,
    trainers.BasicDistributedTrainer[
        flow_map.LagrangianFlowMapModel, FlowMapTrainState
    ],
):
  """Multi-device `pmap` EMA trainer for Lagrangian flow-map models."""


class LagrangianSelfDistilledFlowMapTrainer(
    EmaFlowMapTrainer[flow_map.LagrangianSelfDistilledFlowMapModel]
):
  """Single-device EMA trainer for Lagrangian self-distilled flow maps."""

  def _build_train_metrics(self) -> Collection:
    return clu_metrics.Collection.create(
        train_loss=clu_metrics.Average.from_output("loss"),
        train_loss_std=_Std.from_output("loss"),
        train_loss_x=clu_metrics.Average.from_output("loss_x"),
        train_loss_x_std=_Std.from_output("loss_x"),
        train_loss_v=clu_metrics.Average.from_output("loss_v"),
        train_loss_v_std=_Std.from_output("loss_v"),
    )

  def _build_eval_metrics(self) -> Collection:
    return clu_metrics.Collection.create(
        eval_loss=clu_metrics.Average.from_output("eval_loss"),
        eval_loss_x=clu_metrics.Average.from_output("eval_loss_x"),
        eval_loss_v=clu_metrics.Average.from_output("eval_loss_v"),
        eval_plot_data=clu_metrics.CollectingMetric.from_outputs(
            tuple(
                f"{num_steps}" for num_steps in self.model.number_of_eval_steps
            )
        ),
    )

  @classmethod
  def inference_fn_from_state_dict(
      cls,
      state: FlowMapTrainState,
      *args: Any,
      use_ema: bool = True,
      **kwargs: Any,
  ) -> Callable[..., Array]:
    """Builds forward LSD flow map `x + (t - s) * u_theta` from `state`."""
    variables = cls._variables_from_state(state, use_ema=use_ema)
    return flow_map.LagrangianSelfDistilledFlowMapModel.inference_fn(
        variables, *args, **kwargs
    )


class DistributedLagrangianSelfDistilledFlowMapTrainer(
    LagrangianSelfDistilledFlowMapTrainer,
    trainers.BasicDistributedTrainer[
        flow_map.LagrangianSelfDistilledFlowMapModel, FlowMapTrainState
    ],
):
  """Multi-device `pmap` EMA trainer for Lagrangian self-distilled flow maps."""


ConditionalLagrangianSelfDistilledFlowMapTrainer = (
    LagrangianSelfDistilledFlowMapTrainer
)
DistributedConditionalLagrangianSelfDistilledFlowMapTrainer = (
    DistributedLagrangianSelfDistilledFlowMapTrainer
)


class MeanFlowTrainer(EmaFlowMapTrainer[mean_flow.MeanFlowModel]):
  """Single-device EMA trainer for MeanFlow models."""

  def _build_train_metrics(self) -> Collection:
    return clu_metrics.Collection.create(
        train_loss=clu_metrics.Average.from_output("loss"),
        train_loss_std=_Std.from_output("loss"),
        train_raw_mse=clu_metrics.Average.from_output("raw_mse"),
        train_raw_mse_std=_Std.from_output("raw_mse"),
    )

  def _build_eval_metrics(self) -> Collection:
    return clu_metrics.Collection.create(
        eval_loss=clu_metrics.Average.from_output("eval_loss"),
        eval_plot_data=clu_metrics.CollectingMetric.from_outputs((
            "eval_samples",
        )),
    )

  @classmethod
  def inference_fn_from_state_dict(
      cls,
      state: FlowMapTrainState,
      *args: Any,
      use_ema: bool = True,
      **kwargs: Any,
  ) -> Callable[..., Array]:
    """Builds a MeanFlow velocity inference callable from `state`."""
    variables = cls._variables_from_state(state, use_ema=use_ema)
    return mean_flow.MeanFlowModel.inference_fn(variables, *args, **kwargs)


class DistributedMeanFlowTrainer(
    MeanFlowTrainer,
    trainers.BasicDistributedTrainer[
        mean_flow.MeanFlowModel, FlowMapTrainState
    ],
):
  """Multi-device `pmap` EMA trainer for MeanFlow models."""
