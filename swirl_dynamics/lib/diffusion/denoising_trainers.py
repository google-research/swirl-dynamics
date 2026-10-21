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

"""Trainers for denoising models."""

from collections.abc import Callable, Mapping
import functools
from typing import Any, TypeVar

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
import optax
from swirl_dynamics.lib.diffusion import denoising_models as models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array = jax.Array
VariableDict = trainers.VariableDict


@flax.struct.dataclass
class _Std(clu_metrics.Std):
  """`clu.metrics.Std` avoiding `jnp.clip(..., a_min=...)` on PyPI `clu`."""

  def compute(self) -> Array:
    mean = self.total / self.count
    variance = jnp.clip(self.sum_of_squares / self.count - mean**2, min=0.0)
    return jnp.sqrt(variance)


class DenoisingModelTrainState(train_states.BasicTrainState):
  """Train state with an additional field tracking the EMA params."""

  ema_state: optax.EmaState | Mapping[str, Any] | None = None

  @property
  def ema_variables(self) -> flax.core.FrozenDict:
    if self.ema_state is None:
      raise ValueError("EMA state is none.")
    ema_params = (
        self.ema_state["ema"]
        if isinstance(self.ema_state, Mapping)
        else self.ema_state.ema
    )
    return flax.core.freeze(dict(params=ema_params, **self.flax_mutables))


TrainState = TypeVar("TrainState", bound=DenoisingModelTrainState)
DenoisingModel = TypeVar("DenoisingModel", bound=models.DenoisingModel)


class DenoisingTrainer(trainers.BasicTrainer[DenoisingModel, TrainState]):
  """Single-device trainer for denoising models.

  Note that inherited `eval_step` evaluates raw `train_state.model_variables`
  rather than `train_state.ema_variables` (matching the legacy
  `projects.probabilistic_diffusion` trainer). Use
  `inference_fn_from_state_dict(..., use_ema=True)` for EMA-parameter inference.
  """

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):  # pyrefly: ignore[bad-override]
    train_loss: clu_metrics.Average.from_output("loss")  # pyrefly: ignore[invalid-annotation]
    train_loss_std: _Std.from_output("loss")  # pyrefly: ignore[invalid-annotation]

  @functools.cached_property
  def EvalMetrics(self):  # pyrefly: ignore[bad-override]
    denoising_metrics = {
        f"eval_denoise_lvl{i}": clu_metrics.Average.from_output(f"sigma_lvl{i}")
        for i in range(self.model.num_eval_noise_levels)
    }
    return clu_metrics.Collection.create(**denoising_metrics)

  def __init__(self, ema_decay: float, *args, **kwargs):
    self.ema = optax.ema(ema_decay)
    super().__init__(*args, **kwargs)

  def initialize_train_state(self, rng: Array) -> TrainState:
    init_vars = self.model.initialize(rng)
    mutables, params = flax.core.pop(init_vars, "params")
    return DenoisingModelTrainState.create(  # pyrefly: ignore[bad-return]
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
      _, new_ema_state = self.ema.update(new_params, train_state.ema_state)  # pyrefly: ignore[bad-argument-type]
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
    """Builds a denoising inference function from an unreplicated train state.

    When using `DistributedDenoisingTrainer`, pass
    `trainer.unreplicated_train_state` rather than the device-replicated
    `trainer.train_state`.

    Args:
      state: An unreplicated `DenoisingModelTrainState`.
      *args: Positional arguments forwarded to `DenoisingModel.inference_fn`.
      use_ema: Whether to evaluate using `state.ema_variables` instead of raw
        `state.model_variables`.
      **kwargs: Keyword arguments forwarded to `DenoisingModel.inference_fn`.

    Returns:
      The denoising inference callable.
    """
    variables = state.ema_variables if use_ema else state.model_variables
    return models.DenoisingModel.inference_fn(variables, *args, **kwargs)


class DistributedDenoisingTrainer(
    DenoisingTrainer[DenoisingModel, TrainState],
    trainers.BasicDistributedTrainer[DenoisingModel, TrainState],
):
  """Multi-device trainer for denoising models.

  Note that `flax_mutables` are not automatically aggregated across replicas;
  stateful layers (e.g. `nn.BatchNorm`) must configure `axis_name="batch"` to
  synchronize mutable state across devices, and `inference_fn_from_state_dict`
  should be called with `trainer.unreplicated_train_state`.
  """

  # MRO: DenoisingTrainer > BasicDistributedTrainer > BasicTrainer
  ...
