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

"""Train states for gradient descent mini-batch training.

Train state classes are data containers that hold the model variables, optimizer
states, plus everything else that collectively represent a complete snapshot of
the training. In other words, by saving/loading a train state, one
saves/restores the training progress.
"""

import functools
from typing import Self

import flax
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

EMPTY_DICT = flax.core.freeze({})
FrozenVariableDict = flax_scope.FrozenVariableDict


class TrainState(flax.struct.PyTreeNode):
  """Base train state class.

  Attributes:
    step: A counter that holds the number of gradient steps applied.
  """

  step: jax.Array

  @functools.cached_property  # cache to avoid unreplicating repeatedly
  def int_step(self) -> int:
    """Returns the step as an int.

    This method works on both regular and replicated objects. It detects whether
    the current object is replicated by looking at the dimensions, and
    unreplicates the `step` field if necessary before returning it.
    """
    return int(self.step[0] if self.step.ndim > 0 else self.step)

  @classmethod
  def restore_from_orbax_ckpt(
      cls,
      ckpt_dir: str,
      step: int | None = None,
      field: str = "default",
      ref_state: Self | None = None,
  ) -> Self:
    """Restores train state from an orbax checkpoint directory.

    Args:
      ckpt_dir: A directory which may contain checkpoints at different steps. A
        checkpoint manager will be instantiated in this folder to load a
        checkpoint at the desired step.
      step: The training step to restore checkpoint from. Retores the latest
        step if `None`.
      field: The field of the checkpoint containing the train state to be
        restored.
      ref_state: A reference state instance. If provided, the restored state
        will be the same type with its leaves replaced by values in the
        checkpoint. Otherwise, the restored object will be raw dictionaries,
        which should be fine for inference but will become problematic to resume
        training from.

    Returns:
      Restored train state.
    """
    mngr = ocp.CheckpointManager(
        ckpt_dir, item_handlers={field: ocp.StandardCheckpointHandler()}
    )
    if ref_state is not None:
      restored = mngr.restore(
          step or mngr.latest_step(),
          args=ocp.args.Composite(
              **{field: ocp.args.StandardRestore(item=ref_state)}
          ),
      )
      if restored[field] is None:
        raise ValueError(f"Field `{field}` not found in the checkpoint.")
      return restored[field]
    else:
      restored = mngr.restore(step or mngr.latest_step())
      state_fields = restored[field]
      if state_fields is None:
        raise ValueError(f"Field `{field}` not found in the checkpoint.")
      return cls(**state_fields)

  @classmethod
  def create(cls, replicate: bool = False, **kwargs) -> Self:
    """Creates a new train state with step count 0."""
    state = cls(step=jnp.array(0), **kwargs)
    return state if not replicate else flax.jax_utils.replicate(state)


class BasicTrainState(TrainState):
  """Train state that stores optimizer state, flax model params and mutables.

  Attributes:
    params: The parameters of the model.
    opt_state: The optimizer state of the parameters.
    flax_mutables: The flax mutable fields (e.g. batch stats for batch norm
      layers) of the model being trained.
  """

  params: FrozenVariableDict
  opt_state: optax.OptState
  flax_mutables: FrozenVariableDict = EMPTY_DICT

  @property
  def model_variables(self) -> FrozenVariableDict:
    """Assembles model variable for inference."""
    return flax.core.freeze(dict(params=self.params, **self.flax_mutables))
