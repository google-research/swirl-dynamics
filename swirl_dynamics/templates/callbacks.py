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

"""Training callback library."""

from collections.abc import Mapping, Sequence
import os
import time

from clu import metric_writers
from clu import periodic_actions
import flax
import gin
import jax
import numpy as np
from orbax import checkpoint
from swirl_dynamics.templates import trainers
import tqdm.auto as tqdm

Array = jax.Array
Trainer = trainers.BaseTrainer


class Callback:
  """Abstract base class for callbacks.

  Callbacks are self-contained programs containing some common, reusable logic
  that is non-essential (such as saving model checkpoints, reporting progress,
  profiling, the absence of which would not "break" training) to model training.
  The instance methods of these objects are hooks that get executed at various
  phases of training (i.e. fixed positions inside `train.run` function).

  The execution  (in `train.run`) observes the following flow::

    callbacks.on_train_begin()
    while training:
      callbacks.on_train_batches_begin()
      run_train_steps()
      callbacks.on_train_batches_end()
      if should_run_evaluation:
        callbacks.on_eval_batches_begin()
        run_eval_steps()
        callbacks.on_eval_batches_end()
    callbacks.on_train_end()

  The hooks may read and/or overwrite the trainer state and/or train/eval
  metrics, and have access to a metric_writer that writes desired info/variables
  to the working directory in tensorflow event format.

  When multiple (i.e. a list of) callbacks are used, the
  `on_{train/eval}_batches_end` methods are called in reverse order (so that
  together with `on_{train/eval}_batches_begin` calls they resemble
  the `__exit__` and `__enter__` methods of python contexts).
  """

  @property
  def metric_writer(self) -> metric_writers.MultiWriter:
    assert hasattr(self, "_metric_writer")
    return self._metric_writer

  @metric_writer.setter
  def metric_writer(self, writer: metric_writers.MultiWriter) -> None:
    self._metric_writer = writer

  def on_train_begin(self, trainer: Trainer) -> None:
    """Called before the training loop starts."""

  def on_train_batches_begin(self, trainer: Trainer) -> None:
    """Called before a training segment begins."""

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: Mapping[str, Array]
  ) -> None:
    """Called after a training segment ends."""

  def on_eval_batches_begin(self, trainer: Trainer) -> None:
    """Called before an evaluation segment begins."""

  def on_eval_batches_end(
      self, trainer: Trainer, eval_metrics: Mapping[str, Array]
  ) -> None:
    """Called after an evaluation segment ends."""

  def on_train_end(self, trainer: Trainer) -> None:
    """Called when training ends."""


class TrainStateCheckpoint(Callback):
  """Callback that periodically saves train state checkpoints."""

  def __init__(
      self,
      base_dir: str,
      folder_prefix: str = "checkpoints",
      options: checkpoint.CheckpointManagerOptions | None = None,
  ):
    self.save_dir = os.path.join(base_dir, folder_prefix)
    self.ckpt_manager = checkpoint.CheckpointManager(
        self.save_dir, checkpoint.PyTreeCheckpointer(), options=options
    )

  def on_train_begin(self, trainer: Trainer) -> None:
    """Sets up directory, saves initial or restore the most recent state."""
    self.last_eval_metric = {}
    # retrieve from existing checkpoints if possible
    if self.ckpt_manager.latest_step() is not None:
      train_state = self.ckpt_manager.restore(
          self.ckpt_manager.latest_step(), items=trainer.train_state
      )
      # Replicate the restored train state (always unreplicated) for a
      # distributed trainer.
      trainer.train_state = (
          flax.jax_utils.replicate(train_state)
          if trainer.is_distributed
          else train_state
      )

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: Mapping[str, Array]
  ) -> None:
    assert self.last_eval_metric is not None
    self.ckpt_manager.save(
        step=trainer.train_state.int_step,
        # This always saves the unreplicated train state.
        # Converting to np array seems necessary for multi-host environments.
        items=jax.tree_map(np.array, trainer.unreplicated_train_state),
        metrics=dict(**train_metrics, **self.last_eval_metric),
    )

  def on_eval_batches_end(
      self, trainer: Trainer, eval_metrics: Mapping[str, Array]
  ) -> None:
    del trainer
    self.last_eval_metric = eval_metrics

  def on_train_end(self, trainer: Trainer) -> None:
    # always save at the end of training
    if self.ckpt_manager.latest_step() != trainer.train_state.int_step:
      self.ckpt_manager.save(
          trainer.train_state.int_step,
          items=jax.tree_map(np.array, trainer.unreplicated_train_state),
          force=True,
      )


class ProgressReport(Callback):
  """Callback that reports progress during training.

  Wraps `clu.periodic_actions.ReportProgress`, which reports progress summary
  via `platform.work_unit().set_notes()`, and logs two additional timing
  metrics, namely `uptime` and `steps_per_sec`.
  """

  def __init__(
      self,
      num_train_steps: int | None = None,
      every_steps: int | None = None,
      every_secs: float | None = 60.0,
  ):
    self.num_train_steps = num_train_steps
    self.every_steps = every_steps
    self.every_secs = every_secs
    self.report_progress = None

  def on_train_begin(self, trainer: Trainer) -> None:
    del trainer
    self.report_progress = periodic_actions.ReportProgress(
        num_train_steps=self.num_train_steps,
        every_steps=self.every_steps,
        every_secs=self.every_secs,
        writer=self.metric_writer,
    )

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: Mapping[str, Array]
  ) -> None:
    del train_metrics
    assert self.report_progress is not None
    self.report_progress(trainer.train_state.int_step, time.time())


class TqdmProgressBar(Callback):
  """Tqdm progress bar callback to monitor training progress in real time."""

  def __init__(
      self,
      total_train_steps: int | None,
      train_monitors: Sequence[str],
      eval_monitors: Sequence[str] = (),
  ):
    """ProgressBar constructor.

    Args:
      total_train_steps: the total number of training steps, which is displayed
        as the maximum progress on the bar.
      train_monitors: keys in the training metrics whose values are updated on
        the progress bar after every training metric aggregation.
      eval_monitors: same as `train_monitors` except applying to evaluation.
    """
    self.total_train_steps = total_train_steps
    self.train_monitors = train_monitors
    self.eval_monitors = eval_monitors
    self.current_step = 0
    self.eval_postfix = {}  # keeps record of the most recent eval monitor
    self.bar = None

  def on_train_begin(self, trainer: Trainer) -> None:
    del trainer
    self.bar = tqdm.tqdm(total=self.total_train_steps, unit="step")

  def on_train_batches_end(
      self, trainer: Trainer, train_metrics: Mapping[str, Array]
  ) -> None:
    assert self.bar is not None
    self.bar.update(trainer.train_state.int_step - self.current_step)
    self.current_step = trainer.train_state.int_step
    postfix = {
        monitor: train_metrics[monitor] for monitor in self.train_monitors
    }
    self.bar.set_postfix(**postfix, **self.eval_postfix)

  def on_eval_batches_end(
      self, trainer: Trainer, eval_metrics: Mapping[str, Array]
  ) -> None:
    del trainer
    self.eval_postfix = {
        monitor: eval_metrics[monitor] for monitor in self.eval_monitors
    }

  def on_train_end(self, trainer: Trainer) -> None:
    del trainer
    assert self.bar is not None
    self.bar.close()


class LogGinConfig(Callback):
  """Write gin config string as text in TensorBoard."""

  def on_train_begin(self, trainer: trainers.BaseTrainer) -> None:
    if jax.process_index() == 0:
      config_str = gin.operative_config_str()
      self.metric_writer.write_texts(
          0, {"config": gin.markdown(config_str), "raw_config_str": config_str}
      )
      self.metric_writer.flush()
