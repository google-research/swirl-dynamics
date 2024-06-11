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

import io
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from clu import metric_writers
import flax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from swirl_dynamics.templates import callbacks
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers
from swirl_dynamics.templates import utils

jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS
mock = absltest.mock


def _mock_train_state(step, replicate=False):
  state = train_states.TrainState(step=jnp.array(step))
  return flax.jax_utils.replicate(state) if replicate else state


class TestTrainer(trainers.BaseTrainer):

  @property
  def train_step(self):
    def _train_step(train_state, batch):
      del batch
      return train_state, {}

    return _train_step

  @property
  def eval_step(self):
    def _eval_step(train_state, batch):
      del train_state, batch
      return {}

    return _eval_step

  def initialize_train_state(self, rng):
    del rng
    return train_states.TrainState.create(replicate=self.is_distributed)


class TrainStateCheckpointCallbackTest(parameterized.TestCase):

  @parameterized.product(
      num_train_steps=(5, 10),
      save_interval_steps=(2, 3),
      max_to_keep=(2, 4),
      replicated_state=(True, False),
  )
  def test_saves_correct_number_of_files(
      self, num_train_steps, save_interval_steps, max_to_keep, replicated_state
  ):
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps, max_to_keep=max_to_keep
    )
    work_dir = self.create_tempdir().full_path
    callback = callbacks.TrainStateCheckpoint(
        base_dir=work_dir, options=options
    )

    trainer = TestTrainer(model=mock.Mock(), rng=jax.random.PRNGKey(0))
    trainer.is_distributed = replicated_state
    callback.on_train_begin(trainer)

    for step in range(num_train_steps):
      trainer.train_state = _mock_train_state(step + 1, replicated_state)
      callback.on_train_batches_end(trainer, train_metrics={})

    callback.ckpt_manager.wait_until_finished()

    files = os.listdir(callback.save_dir)
    num_ckpts = min(num_train_steps // save_interval_steps + 1, max_to_keep)
    self.assertLen(files, num_ckpts)

  @parameterized.product(
      replicated_state=(True, False),
  )
  def test_restore_saved_state(self, replicated_state):
    work_dir = self.create_tempdir().full_path
    TestTrainer.is_distributed = replicated_state
    old_trainer = TestTrainer(model=mock.Mock(), rng=jax.random.PRNGKey(0))
    old_trainer.train_state = _mock_train_state(20, replicated_state)
    old_callback = callbacks.TrainStateCheckpoint(base_dir=work_dir)
    old_callback.last_eval_metric = {}
    old_callback.on_train_batches_end(old_trainer, train_metrics={})
    old_callback.ckpt_manager.wait_until_finished()
    # restore
    new_callback = callbacks.TrainStateCheckpoint(base_dir=work_dir)
    new_trainer = TestTrainer(model=mock.Mock(), rng=jax.random.PRNGKey(0))
    new_callback.on_train_begin(new_trainer)
    self.assertIsInstance(new_trainer.train_state, train_states.TrainState)
    self.assertEqual(new_trainer.train_state.int_step, 20)


class ProgressReportCallbackTest(parameterized.TestCase):

  @parameterized.parameters(
      {"num_train_steps": 10, "every_steps": 2},
      {"num_train_steps": 20, "every_steps": 3},
      {"num_train_steps": 8, "every_steps": 8},
      {"num_train_steps": 5, "every_steps": 10},
  )
  def test_reports_timing_metrics(self, num_train_steps, every_steps):
    work_dir = self.create_tempdir().full_path
    callback = callbacks.ProgressReport(
        num_train_steps=num_train_steps, every_steps=every_steps
    )
    trainer = mock.Mock(spec=trainers.BaseTrainer)
    callback.metric_writer = metric_writers.create_default_writer(work_dir)
    callback.on_train_begin(trainer)
    for step in range(num_train_steps):
      trainer.train_state = _mock_train_state(step + 1)
      callback.on_train_batches_end(trainer, mock.Mock())

    callback.metric_writer.flush()
    written_metrics = utils.load_scalars_from_tfevents(work_dir)
    # length should be ceil(num_train_steps/every_steps)
    self.assertLen(written_metrics.keys(), -(num_train_steps // -every_steps))
    if every_steps <= num_train_steps:
      self.assertIn("steps_per_sec", written_metrics[every_steps].keys())


class TqdmProgressBarTest(absltest.TestCase):

  def test_reports_monitors(self):
    total_train_steps = 100
    callback = callbacks.TqdmProgressBar(
        total_train_steps=total_train_steps,
        train_monitors=["train_loss"],
        eval_monitors=["eval_accuracy"],
    )
    trainer = mock.Mock(spec=trainers.BaseTrainer)
    trainer.train_state = _mock_train_state(0)
    with mock.patch("sys.stderr", io.StringIO()) as stderr:
      callback.on_train_begin(trainer)
      with self.subTest("FirstTrainMonitorDisplay"):
        callback.on_train_batches_end(trainer, {"train_loss": jnp.array(0.1)})
        self.assertRegex(stderr.getvalue(), "train_loss=0.1")

      with self.subTest("FirstEvalMonitorDisplay"):
        callback.on_eval_batches_end(trainer, {"eval_accuracy": jnp.array(0.9)})
        trainer.train_state = _mock_train_state(total_train_steps // 2)
        callback.on_train_batches_end(trainer, {"train_loss": jnp.array(0.5)})
        self.assertRegex(
            stderr.getvalue(), f" {total_train_steps // 2}/{total_train_steps} "
        )
        self.assertRegex(stderr.getvalue(), "eval_accuracy=0.9.*train_loss=0.5")

      with self.subTest("TrainMonitorUpdate"):
        trainer.train_state = _mock_train_state(total_train_steps)
        callback.on_train_batches_end(trainer, {"train_loss": jnp.array(0.3)})
        self.assertRegex(
            stderr.getvalue(), f" {total_train_steps}/{total_train_steps} "
        )
        self.assertRegex(stderr.getvalue(), "eval_accuracy=0.9.*train_loss=0.3")

      with self.subTest("TrainEndCloseBar"):
        callback.on_train_end(trainer)
        self.assertRegex(
            stderr.getvalue(), f"{total_train_steps}/{total_train_steps}"
        )



if __name__ == "__main__":
  absltest.main()
