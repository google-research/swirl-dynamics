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
from typing import TypeVar

import flax
from flax.core import scope as flax_scope
import jax
import jax.numpy as jnp
import optax
from orbax import checkpoint

# TODO(wanzy): use typing.Self after python 3.11 (PEP 673)
TState = TypeVar("TState", bound="TrainState")

EMPTY_DICT = flax.core.freeze({})
FrozenVariableDict = flax_scope.FrozenVariableDict


class TrainState(flax.struct.PyTreeNode):
  """Base train state class.

  Attributes:
    step: a counter that holds the number of gradient steps applied.
    _rng: a Jax random key to use for training if needed. It should never be
      accessed directly (instead use `split_rng()` method above to retrieve the
      split rng while simultaneously updating the state).
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
      ref_state: TState | None = None,
  ) -> TState:
    """Restores train state from an orbax checkpoint."""
    # NOTE: if `ref_state` is not provided, the loaded object will contain raw
    # dictionaries, which should be fine for inference but may become
    # problematic to continue training with
    mngr = checkpoint.CheckpointManager(
        ckpt_dir, checkpoint.PyTreeCheckpointer()
    )
    if ref_state is not None:
      return mngr.restore(step or mngr.latest_step(), items=ref_state)
    else:
      return cls(**mngr.restore(step or mngr.latest_step()))

  @classmethod
  def create(cls, replicate: bool = False, **kwargs) -> TState:
    """Creates a new train state with step count 0."""
    state = cls(step=jnp.array(0), **kwargs)
    return state if not replicate else flax.jax_utils.replicate(state)


class BasicTrainState(TrainState):
  """Train state that stores optimizer state, flax model params and mutables.

  Attributes:
    params: the parameters of the model as a PyTree.
    opt_state: optimizer state of the parameters.
    flax_mutables: flax mutable fields (e.g. batch stats for batch norm layers)
      of the model being trained, also as a PyTree.
  """

  params: FrozenVariableDict
  opt_state: optax.OptState
  flax_mutables: FrozenVariableDict = EMPTY_DICT

  @property
  def model_variables(self) -> FrozenVariableDict:
    """Assembles model variable for inference."""
    return flax.core.freeze(dict(params=self.params, **self.flax_mutables))
