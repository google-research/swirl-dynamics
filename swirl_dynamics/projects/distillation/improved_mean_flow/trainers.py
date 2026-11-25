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

"""Single-device and distributed EMA trainers for Improved Mean Flows."""

from collections.abc import Callable
from typing import TypeAlias

from clu import metrics as clu_metrics
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import flow_map
from swirl_dynamics.projects.distillation.improved_mean_flow import models as imf_models
from swirl_dynamics.templates import trainers

Array: TypeAlias = jax.Array
Collection: TypeAlias = type[clu_metrics.Collection]
TrainState: TypeAlias = flow_map.FlowMapTrainState


@flax.struct.dataclass
class _Std(clu_metrics.Std):
  """`clu.metrics.Std` avoiding `jnp.clip(..., a_min=...)` on PyPI `clu`."""

  def compute(self) -> Array:
    mean = self.total / self.count
    variance = jnp.clip(self.sum_of_squares / self.count - mean**2, min=0.0)
    return jnp.sqrt(variance)


class ImprovedMeanFlowTrainer(
    flow_map.EmaFlowMapTrainer[imf_models.ImprovedMeanFlowModel]
):
  """Single-device EMA trainer for Improved Mean Flow models."""

  def _build_train_metrics(self) -> Collection:
    return clu_metrics.Collection.create(
        train_loss=clu_metrics.Average.from_output("loss"),
        train_loss_std=_Std.from_output("loss"),
        train_loss_u=clu_metrics.Average.from_output("loss_u"),
        train_loss_u_std=_Std.from_output("loss_u"),
        train_loss_v=clu_metrics.Average.from_output("loss_v"),
        train_loss_v_std=_Std.from_output("loss_v"),
        train_raw_mse=clu_metrics.Average.from_output("raw_mse"),
        train_raw_mse_std=_Std.from_output("raw_mse"),
    )

  def _build_eval_metrics(self) -> Collection:
    return clu_metrics.Collection.create(
        eval_loss=clu_metrics.Average.from_output("eval_loss"),
        eval_loss_v=clu_metrics.Average.from_output("eval_loss_v"),
        eval_plot_data=clu_metrics.CollectingMetric.from_outputs(
            tuple(f"samples_{k}" for k in self.model.number_of_eval_steps)
        ),
    )

  @classmethod
  def inference_fn_from_state_dict(
      cls,
      state: TrainState,
      mean_flow_model: nn.Module,
      *,
      use_ema: bool = True,
      cfg_mode: imf_models.CfgMode = "none",
      use_aux_v_head: bool = False,
      default_cfg_scale: float = 1.0,
      default_cfg_interval: tuple[float, float] = (0.0, 1.0),
  ) -> Callable[..., Array]:
    """Builds an iMF velocity inference callable from a `TrainState`."""
    variables = cls._variables_from_state(state, use_ema=use_ema)
    return imf_models.ConditionalImprovedMeanFlowModel.inference_fn(
        variables,
        mean_flow_model,
        cfg_mode=cfg_mode,
        use_aux_v_head=use_aux_v_head,
        default_cfg_scale=default_cfg_scale,
        default_cfg_interval=default_cfg_interval,
    )


class DistributedImprovedMeanFlowTrainer(
    ImprovedMeanFlowTrainer,
    trainers.BasicDistributedTrainer[
        imf_models.ImprovedMeanFlowModel, TrainState
    ],
):
  """Multi-device `pmap` EMA trainer for Improved Mean Flow models."""
