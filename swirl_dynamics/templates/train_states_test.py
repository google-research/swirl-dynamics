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

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint
from swirl_dynamics.templates import train_states

jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


class TrainStateTest(parameterized.TestCase):

  def test_create_replciated_state(self):
    state = train_states.TrainState.create(replicate=True)
    self.assertEqual(state.step.shape, (jax.local_device_count(),))

  @parameterized.product(replicate=(True, False))
  def test_int_step(self, replicate):
    state = train_states.TrainState.create(replicate=replicate)
    self.assertEqual(state.int_step, 0)

  def test_load_state_from_ckpt(self):
    state = train_states.TrainState.create(replicate=False)
    save_dir = self.create_tempdir().full_path
    save_dir = os.path.join(save_dir, "checkpoints")
    checkpoint.CheckpointManager(
        save_dir, checkpoint.PyTreeCheckpointer()
    ).save(step=100, items=state, force=True)
    loaded_state = train_states.TrainState.restore_from_orbax_ckpt(save_dir)
    jax.tree_util.tree_map(np.testing.assert_array_equal, loaded_state, state)


class BasicTrainStateTest(absltest.TestCase):

  def test_load_state_from_ckpt(self):
    model_variables = flax.core.freeze({
        "params": {"dense": {"bias": jnp.zeros(5), "kernel": jnp.ones((5, 5))}},
        "batch_stats": {
            "BatchNorm_0": {"bias": jnp.ones(10), "scale": jnp.ones(10)}
        },
    })
    mutables, params = flax.core.pop(model_variables, "params")
    state = train_states.BasicTrainState.create(
        params=params,
        opt_state=optax.sgd(0.1).init(params),
        flax_mutables=mutables,
    )
    save_dir = self.create_tempdir().full_path
    save_dir = os.path.join(save_dir, "checkpoints")
    checkpoint.CheckpointManager(
        save_dir, checkpoint.PyTreeCheckpointer()
    ).save(step=0, items=state, force=True)
    with self.subTest("No ref state"):
      loaded_variables = train_states.BasicTrainState.restore_from_orbax_ckpt(
          ckpt_dir=save_dir
      ).model_variables
      jax.tree_util.tree_map(
          np.testing.assert_array_equal, loaded_variables, model_variables
      )
    with self.subTest("With ref state"):
      loaded_state = train_states.BasicTrainState.restore_from_orbax_ckpt(
          ckpt_dir=save_dir,
          ref_state=state.replace(step=66),
      )
      jax.tree_util.tree_map(np.testing.assert_array_equal, loaded_state, state)


if __name__ == "__main__":
  absltest.main()
