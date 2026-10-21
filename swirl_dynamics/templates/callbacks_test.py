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

import dataclasses
import datetime
import io
import os
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from clu import metric_writers
import flax
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy as preservation_policy_lib
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
    new_trainer.is_distributed = replicated_state
    new_callback.on_train_begin(new_trainer)
    self.assertIsInstance(new_trainer.train_state, train_states.TrainState)
    self.assertEqual(new_trainer.train_state.int_step, 20)

  def test_checkpoint_uses_current_step_eval_metrics(self):
    work_dir = self.create_tempdir().full_path
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=2,
        best_fn=lambda metrics: metrics["eval_loss"],
        best_mode="min",
        max_to_keep=1,
    )
    callback = callbacks.TrainStateCheckpoint(
        base_dir=work_dir, options=options
    )
    trainer = TestTrainer(model=mock.Mock(), rng=jax.random.PRNGKey(0))
    trainer.is_distributed = False
    callback.on_train_begin(trainer)

    # Step 2 is an eval step: saving is deferred until on_eval_batches_end.
    callback._eval_scheduled_steps.add(2)
    trainer.train_state = _mock_train_state(2, replicate=False)
    callback.on_train_batches_end(
        trainer, train_metrics={"train_loss": jnp.array(1.0)}
    )
    self.assertIsNone(callback.ckpt_manager.latest_step())
    callback.on_eval_batches_end(
        trainer, eval_metrics={"eval_loss": jnp.array(0.5)}
    )
    callback.ckpt_manager.wait_until_finished()
    self.assertEqual(callback.ckpt_manager.latest_step(), 2)
    self.assertEqual(callback.last_eval_metric["eval_loss"], 0.5)
    self.assertNotIn(2, callback._eval_scheduled_steps)

    # Step 4 has worse eval_loss (0.8 > 0.5, wrapped in clu.values.Scalar):
    # best_fn retains step 2 rather than overwriting with stale 0.5.
    callback._eval_scheduled_steps.add(4)
    trainer.train_state = _mock_train_state(4, replicate=False)
    step4_train: dict[str, utils.Scalar] = {
        "train_loss": utils.values.Scalar(0.9)
    }
    step4_eval: dict[str, utils.Scalar] = {
        "eval_loss": utils.values.Scalar(0.8)
    }
    callback.on_train_batches_end(trainer, train_metrics=step4_train)
    callback.on_eval_batches_end(trainer, eval_metrics=step4_eval)
    callback.ckpt_manager.wait_until_finished()
    self.assertCountEqual(callback.ckpt_manager.all_steps(), [2])

    # Step 6 improves eval_loss (0.2 < 0.5): best checkpoint advances to 6.
    callback._eval_scheduled_steps.add(6)
    trainer.train_state = _mock_train_state(6, replicate=False)
    callback.on_train_batches_end(
        trainer, train_metrics={"train_loss": jnp.array(0.4)}
    )
    callback.on_eval_batches_end(
        trainer, eval_metrics={"eval_loss": jnp.array(0.2)}
    )
    callback.ckpt_manager.wait_until_finished()
    self.assertCountEqual(callback.ckpt_manager.all_steps(), [6])


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




# Dummy model and trainer for testing InitializeFromCheckpoint
class DummyModel(trainers.models.BaseModel):

  def initialize(self, rng: jax.Array) -> train_states.FrozenVariableDict:
    return flax.core.freeze({
        "params": {"w": jax.random.normal(rng, shape=(2, 2))},
        "m": jnp.ones(2),  # Mutable variables
    })

  def loss_fn(self, params, batch, rng, flax_mutables):
    del batch, rng
    new_mutables = {"m": flax_mutables["m"] + 1.0}
    return jnp.sum(params["w"] ** 2), ({"loss": 0.1}, new_mutables)

  def eval_fn(self, variables, batch, rng):  # pyrefly: ignore[bad-override]
    return {}


@flax.struct.dataclass
class DummyMetrics(trainers.clu_metrics.Collection):
  loss: trainers.clu_metrics.Average.from_output("loss")  # pyrefly: ignore[invalid-annotation]


class DummyTrainer(trainers.BasicTrainer):
  TrainMetrics = DummyMetrics  # pylint: disable=invalid-name  # pyrefly: ignore[bad-override]
  EvalMetrics = DummyMetrics  # pylint: disable=invalid-name  # pyrefly: ignore[bad-override]


class DummyDistributedTrainer(trainers.BasicDistributedTrainer):
  TrainMetrics = DummyMetrics  # pylint: disable=invalid-name  # pyrefly: ignore[bad-override]
  EvalMetrics = DummyMetrics  # pylint: disable=invalid-name  # pyrefly: ignore[bad-override]


class InitializeFromCheckpointTest(parameterized.TestCase):

  def _create_trainer(self, is_distributed, rng_key):
    model = DummyModel()
    optimizer = trainers.optax.adam(1e-3)
    if is_distributed:
      trainer = DummyDistributedTrainer(
          optimizer, model, jax.random.PRNGKey(rng_key)
      )
    else:
      trainer = DummyTrainer(optimizer, model, jax.random.PRNGKey(rng_key))
    return trainer

  def _save_checkpoint(self, trainer, work_dir, step):
    new_step = jnp.array(step)
    if trainer.is_distributed:
      new_step = flax.jax_utils.replicate(new_step)
    trainer.train_state = trainer.train_state.replace(step=new_step)
    # Simulate some training to change model state
    params = trainer.train_state.params
    flax_mutables = trainer.train_state.flax_mutables

    params = jax.tree.map(lambda x: x + float(step), params)
    flax_mutables = jax.tree.map(lambda x: x + float(step), flax_mutables)

    trainer.train_state = trainer.train_state.replace(
        params=params, flax_mutables=flax_mutables
    )

    save_callback = callbacks.TrainStateCheckpoint(base_dir=work_dir)
    save_callback.last_eval_metric = {}
    save_callback.on_train_batches_end(trainer, train_metrics={})
    save_callback.ckpt_manager.wait_until_finished()
    return os.path.join(work_dir, "checkpoints")

  def _assert_state_correct(self, trainer, original_state, ckpt_state):
    # Step and opt_state should NOT be restored
    self.assertEqual(trainer.train_state.int_step, 0)
    jax.tree.map(
        np.testing.assert_array_equal,
        trainer.unreplicated_train_state.opt_state,
        original_state.opt_state,
    )

    # Params and mutables SHOULD be restored from ckpt_state
    jax.tree.map(
        np.testing.assert_array_equal,
        trainer.unreplicated_train_state.params,
        ckpt_state.params,
    )
    jax.tree.map(
        np.testing.assert_array_equal,
        trainer.unreplicated_train_state.flax_mutables,
        ckpt_state.flax_mutables,
    )

  def _run_train_steps(self, trainer, num_steps):
    batch = {"data": np.ones((jax.local_device_count(), 2, 2))}
    for i in range(num_steps):
      step_rng = trainer.get_train_rng(trainer.train_state.int_step, num=1)[0]
      preproc_rng = trainer.get_train_rng(
          trainer.train_state.int_step + i, num=1
      )[0]
      current_step = trainer.train_state.int_step
      processed_batch = trainer.preprocess_train_batch(
          batch, current_step, preproc_rng
      )

      if trainer.is_distributed:
        step_rng = jax.random.split(step_rng, jax.local_device_count())

      trainer.train_state, _ = trainer._compiled_train_step(
          trainer.train_state, processed_batch, step_rng
      )
    return trainer

  @parameterized.product(
      src_distributed=[False, True],
      tgt_distributed=[False, True],
  )
  def test_initialize_and_train(self, src_distributed, tgt_distributed):
    work_dir = self.create_tempdir().full_path
    ckpt_step = 5

    # Create and save source checkpoint
    src_trainer = self._create_trainer(src_distributed, rng_key=0)
    ckpt_dir = self._save_checkpoint(src_trainer, work_dir, ckpt_step)
    ckpt_state = src_trainer.unreplicated_train_state

    # Create target trainer
    tgt_trainer = self._create_trainer(tgt_distributed, rng_key=1)
    original_tgt_state = tgt_trainer.unreplicated_train_state

    # Initialize from checkpoint
    restore_callback = callbacks.InitializeFromCheckpoint(
        checkpoint_dir=ckpt_dir,
        fields_to_override=["params", "flax_mutables"],
    )
    restore_callback.on_train_begin(tgt_trainer)

    # Verify state
    self._assert_state_correct(tgt_trainer, original_tgt_state, ckpt_state)

    # Verify training can run
    tgt_trainer = self._run_train_steps(tgt_trainer, 3)
    self.assertEqual(tgt_trainer.train_state.int_step, 3)


@dataclasses.dataclass
class _TestCheckpointInfo:
  step: int
  metrics: dict[str, Any] | None
  time: datetime.datetime = dataclasses.field(
      default_factory=lambda: datetime.datetime(2026, 1, 1)
  )


class ParetoFrontierTest(parameterized.TestCase):

  def test_empty_checkpoints_and_empty_metric_fns(self):
    ctx = preservation_policy_lib.PreservationContext()
    policy = callbacks.ParetoFrontier(metric_fns=[])
    self.assertEqual(policy.should_preserve([], context=ctx), [])
    ckpts = [
        _TestCheckpointInfo(step=10, metrics={"loss": 1.0}),
        _TestCheckpointInfo(step=20, metrics={"loss": 0.5}),
    ]
    self.assertEqual(policy.should_preserve(ckpts, context=ctx), [False, True])

  @parameterized.parameters(0, -3)
  def test_invalid_max_to_keep_raises(self, bad_max):
    with self.assertRaisesRegex(ValueError, "max_to_keep"):
      callbacks.ParetoFrontier(
          metric_fns=[lambda m: m["loss"]], max_to_keep=bad_max
      )

  def test_max_to_keep_smaller_than_num_metrics_raises(self):
    with self.assertRaisesRegex(ValueError, "max_to_keep"):
      callbacks.ParetoFrontier(
          metric_fns=[
              lambda m: m["a"],
              lambda m: m["b"],
              lambda m: m["c"],
          ],
          max_to_keep=2,
      )

  def test_missing_nonfinite_and_clu_scalar_metrics(self):
    ctx = preservation_policy_lib.PreservationContext()
    policy = callbacks.ParetoFrontier(
        metric_fns=[lambda m: m["a"], lambda m: m["b"]], max_to_keep=2
    )
    all_invalid = [
        _TestCheckpointInfo(step=10, metrics=None),
        _TestCheckpointInfo(step=20, metrics={"a": float("nan"), "b": 1.0}),
        _TestCheckpointInfo(step=30, metrics={"a": 1.0}),  # KeyError on "b"
    ]
    self.assertEqual(
        policy.should_preserve(all_invalid, context=ctx), [False, False, True]
    )

    clu_ckpts = [
        _TestCheckpointInfo(
            step=10,
            metrics={"a": utils.values.Scalar(0.4), "b": jnp.array(0.8)},
        ),
        _TestCheckpointInfo(
            step=20,
            metrics={"a": utils.values.Scalar(0.9), "b": jnp.array(0.9)},
        ),
    ]
    # Step 10 is on the frontier; step 20 is dominated but preserved as latest.
    self.assertEqual(
        policy.should_preserve(clu_ckpts, context=ctx), [True, True]
    )

  def test_frontier_dominance_and_anchor_latest_pruning(self):
    ctx = preservation_policy_lib.PreservationContext()
    policy = callbacks.ParetoFrontier(
        metric_fns=[lambda m: m["crps"], lambda m: m["grad"]], max_to_keep=3
    )
    ckpts = [
        _TestCheckpointInfo(10, {"crps": 0.9, "grad": 1.5}),  # dominated
        _TestCheckpointInfo(20, {"crps": 0.8, "grad": 1.0}),  # dominated
        _TestCheckpointInfo(30, {"crps": 0.68, "grad": 0.9}),  # anchor 1
        _TestCheckpointInfo(40, {"crps": 0.72, "grad": 0.4}),  # anchor 2
        _TestCheckpointInfo(50, {"crps": 0.75, "grad": 0.6}),  # dominated
        _TestCheckpointInfo(60, {"crps": 0.70, "grad": 0.5}),  # frontier old
        _TestCheckpointInfo(70, {"crps": 0.69, "grad": 0.6}),  # frontier new
        _TestCheckpointInfo(80, {"crps": 0.95, "grad": 1.2}),  # latest
    ]
    # Anchors {30, 40} + most recent non-anchor frontier {70} + latest {80}.
    res = policy.should_preserve(ckpts, context=ctx)
    kept_steps = [ckpts[i].step for i, keep in enumerate(res) if keep]
    self.assertEqual(kept_steps, [30, 40, 70, 80])


if __name__ == "__main__":
  absltest.main()
