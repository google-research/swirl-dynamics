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

import os

from absl.testing import absltest
from absl.testing import parameterized
from clu import metrics as clu_metrics
import grain.python as pygrain
import jax.numpy as jnp
import numpy as np
from swirl_dynamics.templates import callbacks
from swirl_dynamics.templates import train
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers
from swirl_dynamics.templates import utils

mock = absltest.mock

MSGS = {
    "setup": "Train start",
    "train_begin": "Train batches begin",
    "train_end": "Train batches end",
    "eval_begin": "Eval batches begin",
    "eval_end": "Eval batches end",
    "end": "Train end",
}


def _mock_train_state(step):
  return train_states.BasicTrainState(
      step=jnp.array(step),
      params=mock.Mock(),
      opt_state=mock.Mock(),
      flax_mutables=mock.Mock(),
  )


class TestCallback(callbacks.Callback):
  """Callback that writes various messages indicating callsite locations."""

  def __init__(self, save_dir):
    self.save_dir = save_dir
    self.log_file = open(os.path.join(self.save_dir, "log.txt"), "w")

  def on_train_begin(self, trainer):
    self.log_file.write(MSGS["setup"] + "\n")

  def on_train_batches_begin(self, trainer):
    self.log_file.write(MSGS["train_begin"] + "\n")

  def on_train_batches_end(self, trainer, train_metrics):
    self.log_file.write(MSGS["train_end"] + "\n")

  def on_eval_batches_begin(self, trainer):
    self.log_file.write(MSGS["eval_begin"] + "\n")

  def on_eval_batches_end(self, trainer, eval_metrics):
    self.log_file.write(MSGS["eval_end"] + "\n")

  def on_train_end(self, trainer):
    self.log_file.write(MSGS["end"] + "\n")
    self.log_file.close()


def _expected_execution(train_steps, eval_period):
  """Expected execution stages of `TestCallback`."""
  d, r = divmod(train_steps, eval_period)
  train_batch = ["train_begin", "train_end"]
  eval_batch = ["eval_begin", "eval_end"]
  lines = (
      (train_batch * eval_period + eval_batch) * d
      + train_batch * r
      + eval_batch * int(r != 0)
  )
  return ["setup", *lines, "end"]


class TrainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    source = pygrain.RangeDataSource(start=1, stop=10, step=1)
    self.dummy_dataloader = pygrain.load(source, seed=12, batch_size=1)

    # mock trainer with constant metrics returned
    self.test_trainer = mock.Mock(spec=trainers.BasicTrainer)
    self.test_trainer.train_state = _mock_train_state(0)
    train_metrics = mock.Mock(spec=clu_metrics.Collection)
    train_metrics.compute.return_value = {"loss": 0.1}
    self.test_trainer.train.return_value = train_metrics

    eval_metrics = mock.Mock(spec=clu_metrics.Collection)
    eval_metrics.compute.return_value = {"accuracy": 0.6}
    self.test_trainer.eval.return_value = eval_metrics

  @parameterized.parameters(
      {"agg_steps": 2, "train_steps": 10, "writing_steps": (2, 4, 6, 8, 10)},
      # save at last step if `train_steps` doesn't divide `agg_steps`
      {"agg_steps": 3, "train_steps": 10, "writing_steps": (3, 6, 9, 10)},
  )
  def test_writes_train_metrics(self, agg_steps, train_steps, writing_steps):
    """Test training iterations by examining metrics."""
    workdir = self.create_tempdir().full_path
    train.run(
        train_dataloader=self.dummy_dataloader,
        trainer=self.test_trainer,
        workdir=workdir,
        metric_aggregation_steps=agg_steps,
        total_train_steps=train_steps,
    )
    written = utils.load_scalars_from_tfevents(workdir)
    self.assertLen(written.keys(), np.ceil(train_steps / agg_steps))
    self.assertTrue(all([step in written.keys() for step in writing_steps]))
    self.assertIn("loss", written[train_steps].keys())

  @parameterized.parameters(
      # train_steps % agg_steps == 0; train_steps % eval_period == 0
      {"agg_steps": 2, "train_steps": 20, "eval_period": 4},
      # train_steps % agg_steps == 0; train_steps % eval_period != 0
      {"agg_steps": 2, "train_steps": 20, "eval_period": 6},
      # train_steps % agg_steps != 0
      {"agg_steps": 3, "train_steps": 20, "eval_period": 9},
  )
  def test_writes_eval_metrics(self, agg_steps, train_steps, eval_period):
    """Test evaluation iterations by examining metrics."""
    workdir = self.create_tempdir().full_path
    train.run(
        train_dataloader=self.dummy_dataloader,
        trainer=self.test_trainer,
        workdir=workdir,
        metric_aggregation_steps=agg_steps,
        total_train_steps=train_steps,
        eval_dataloader=self.dummy_dataloader,
        eval_every_steps=eval_period,
        num_batches_per_eval=1,
    )
    written = utils.load_scalars_from_tfevents(workdir)
    for step in np.arange(eval_period, train_steps, eval_period):
      self.assertIn("loss", written[step].keys())
      self.assertIn("accuracy", written[step].keys())
    self.assertIn("accuracy", written[train_steps].keys())

  def test_raises_eval_period_divisibility_error(self):
    """Test error when eval period is not divisible by aggregation steps."""
    with self.assertRaisesRegex(ValueError, "must be an integer multiple of"):
      train.run(
          train_dataloader=self.dummy_dataloader,
          trainer=self.test_trainer,
          workdir=self.create_tempdir().full_path,
          metric_aggregation_steps=10,
          total_train_steps=100,
          eval_dataloader=self.dummy_dataloader,
          eval_every_steps=42,
          num_batches_per_eval=1,
      )

  @parameterized.parameters(
      {"train_steps": 5, "eval_period": 3},  # eval_period % train_steps != 0
      {"train_steps": 12, "eval_period": 4},  # train_steps % train_steps == 0
      {"train_steps": 8, "eval_period": 8},  # train_steps = eval_period
      {"train_steps": 5, "eval_period": 10},  # train_steps < eval_period
  )
  def test_triggers_callbacks(self, train_steps, eval_period):
    workdir = self.create_tempdir().full_path
    train.run(
        train_dataloader=self.dummy_dataloader,
        trainer=self.test_trainer,
        workdir=workdir,
        metric_aggregation_steps=1,
        total_train_steps=train_steps,
        eval_dataloader=self.dummy_dataloader,
        eval_every_steps=eval_period,
        num_batches_per_eval=1,
        callbacks=[TestCallback(workdir)],
    )
    expected_stages = _expected_execution(train_steps, eval_period)
    with open(os.path.join(workdir, "log.txt")) as log_file:
      for stage in expected_stages:
        self.assertEqual(log_file.readline().rstrip(), MSGS[stage])


if __name__ == "__main__":
  absltest.main()
