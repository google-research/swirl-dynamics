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

"""Trainer classes for use in gradient descent mini-batch training."""

import abc
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar

from clu import metrics as clu_metrics
import flax
import jax
import numpy as np
import optax
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states

Array = jax.Array
BatchType = Mapping[str, jax.typing.ArrayLike]
Metrics = clu_metrics.Collection
PyTree = Any
VariableDict = train_states.FrozenVariableDict

M = TypeVar("M")  # Model
S = TypeVar("S", bound=train_states.TrainState)  # Train state

PMAP_AXIS_NAME = "batch"


class BaseTrainer(Generic[M, S], metaclass=abc.ABCMeta):
  """Abstract base trainer for gradient descent mini-batch training.

  At a high level, this class provides some boilerplates for running
  training and evaluation, which should make it easy for one to subclass and
  further customize to specific models and training algorithms.

  Subclasses must provide concrete implementations for training & evaluation
  step functions and trainer state initialization following the required
  inferface. In addition, for standardized metrics handling one should define
  class attributes `TrainMetrics` and `EvalMetrics` (as `clu.metrics.Collection`
  classes), in order to collect model outputs and process them for metrics
  computation.
  """

  is_distributed: bool = False

  @flax.struct.dataclass
  class TrainMetrics(Metrics):
    """Training metrics definition.

    The `train_step` function of this trainer must return an instance of this
    class (corresponding to the metrics computed on a training batch).
    See `clu.metrics.Collection` for how to define this class.
    """

  @flax.struct.dataclass
  class EvalMetrics(Metrics):
    """Evaluation metrics definition.

    The `eval_step` function of this trainer must return an instance of this
    class (corresponding to the metrics computed on an evaluation batch).
    See `clu.metrics.Collection` for how to define this class.
    """

  def __init__(self, model: M, rng: Array):
    self.model = model

    train_rng, eval_rng, self._init_rng = jax.random.split(rng, 3)
    # fold in process index so that each host has different root rngs
    self._train_rng = jax.random.fold_in(train_rng, jax.process_index())
    self._eval_rng = jax.random.fold_in(eval_rng, jax.process_index())

    self._train_state = self.initialize_train_state(self._init_rng)
    self._compiled_train_step = self._maybe_pmap(jax.jit(self.train_step))
    self._compiled_eval_step = self._maybe_pmap(jax.jit(self.eval_step))

  @property
  def train_state(self) -> S:
    return self._train_state

  @train_state.setter
  def train_state(self, train_state: S) -> None:
    self._train_state = train_state

  # **********************************
  # Convenience properties/functions
  # **********************************

  def get_train_rng(self, step: int, num: int = 1) -> Array:
    rng = jax.random.fold_in(self._train_rng, step)
    return jax.random.split(rng, num=num)

  def get_eval_rng(self, num: int = 1) -> Array:
    rng, self._eval_rng = jax.random.split(self._eval_rng)
    return jax.random.split(rng, num=num)

  @property
  def unreplicated_train_state(self) -> S:
    """Returns unreplicated train state (for checkpointing or inference)."""
    return self._maybe_unreplicate(self.train_state)

  def _maybe_pmap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
    return jax.pmap(fn, axis_name=PMAP_AXIS_NAME) if self.is_distributed else fn

  def _maybe_unreplicate(self, tree: PyTree) -> PyTree:
    return flax.jax_utils.unreplicate(tree) if self.is_distributed else tree

  def _maybe_replicate(self, tree: PyTree) -> PyTree:
    return flax.jax_utils.replicate(tree) if self.is_distributed else tree

  def _maybe_split(self, rng: Array) -> Array:
    if self.is_distributed:
      rng = jax.random.split(rng, num=jax.local_device_count())
    return rng

  # **********************************
  # Base train and eval loop logic (should not require overriding in subclass)
  # **********************************

  def train(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
    """Runs training for a specified number of steps."""
    # `.int_step` may involve unreplicate (expensive), so we do it only once.
    step0 = self.train_state.int_step
    train_metrics = self.TrainMetrics.empty()
    for step in range(num_steps):
      cur_step = step0 + step
      with jax.profiler.StepTraceAnnotation("train", step_num=cur_step):
        train_rng, preproc_rng = self.get_train_rng(step=cur_step, num=2)
        batch = self.preprocess_train_batch(
            batch_data=next(batch_iter),
            step=cur_step,
            rng=preproc_rng,
        )
        self.train_state, metrics_update = self._compiled_train_step(
            self.train_state, batch, self._maybe_split(train_rng)
        )
        # NOTE: In a distributed setting, the metrics are expected to be
        # gathered and reduced inside the `train_step` function. See
        # `clu_metrics.Collection.gather_from_model_output()` for a convenient
        # way to do this.
        metrics_update = self._maybe_unreplicate(metrics_update)
        train_metrics = train_metrics.merge(metrics_update)
    return train_metrics

  def eval(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
    """Runs evaluation for a specified number of steps."""
    eval_metrics = self.EvalMetrics.empty()
    for _ in range(num_steps):
      eval_rng, preproc_rng = self.get_eval_rng(num=2)
      batch = self.preprocess_eval_batch(
          batch_data=next(batch_iter), rng=preproc_rng
      )
      metrics_update = self._compiled_eval_step(
          self.train_state, batch, self._maybe_split(eval_rng)
      )
      metrics_update = self._maybe_unreplicate(metrics_update)
      eval_metrics = eval_metrics.merge(metrics_update)
    return eval_metrics

  # **********************************
  # Mandatory Hooks
  # **********************************

  @abc.abstractmethod
  def initialize_train_state(self, rng: Array) -> S:
    """Instantiate the initial train state.

    Args:
      rng: the jax random key for initializing the model variables and the train
        state.

    Returns:
      The initialized train state, which should contain the initial model
      parameters and other variables that represent training progress.
    """
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def train_step(self) -> Callable[[S, BatchType], tuple[S, Metrics]]:
    """Returns the train step function.

    One should define the train step function locally inside this property and
    return it as output. The function is meant to be specific to the interfaces
    of the model, using it to compute and apply gradient updates. It is also
    expected to return a `TrainMetrics` object constructed from model outputs.

    The step function will be jit-compiled and called on every training data
    batch. Metrics accumulation and merging across batches is automatically
    handled (at the base trainer level).

    Example::

      @property
      def train_step(self):

        def _train_step(state, batch):
          grad = jax.grad(self.model.loss_fn)(state.params, batch)
          state = state.apply_grads(grad)
          metrics = compute_train_metrics(self.model, state.params, batch)
          return state, self.TrainMetrics.single_from_model_output(**metrics)

        return _train_step
    """
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def eval_step(self) -> Callable[[S, BatchType, Array], Metrics]:
    """Returns the evaluation step function.

    Same as `BaseTrainer.train_step`, except for evaluation step.
    """
    raise NotImplementedError

  # **********************************
  # Optional Hooks
  # **********************************

  def preprocess_train_batch(
      self, batch_data: BatchType, step: int, rng: Array
  ) -> BatchType:
    """Preprocesses batch data before calling the training step function.

    This hook allows one to change the training batch on top of the dataset
    level transformations. For example, the batch can be modified based on the
    number of training steps already applied (i.e. scheduled preprocessing), or
    randomly perturbed in desired ways. Only python transformations should be
    applied here (compiled transformations should punted to the dataset level).
    The default is simply no preprocessing and passing the training batch as is.

    Args:
      batch_data: training batch data yielded by the dataset.
      step: the current training step (for scheduled preprocessing).
      rng: a Jax random key for use in case randomized preprocessing is needed.

    Returns:
      The preprocessed batch data.
    """
    del step, rng
    return batch_data

  def preprocess_eval_batch(
      self, batch_data: BatchType, rng: Array
  ) -> BatchType:
    """Preprocesses batch before calling the eval step function.

    Same as `preprocess_train_batch`, except for evaluation batch data. The
    default is to use the same preprocessing as for training.

    Args:
      batch_data: evaluation batch data yielded by the dataset.
      rng: a Jax random key for use in case randomized preprocessing is needed.

    Returns:
      The preprocessed evaluation batch data.
    """
    return self.preprocess_train_batch(batch_data=batch_data, step=0, rng=rng)

  @staticmethod
  def build_inference_fn(state: S, **kwargs) -> Callable[..., Any]:
    """Builds an inference function from a train state."""
    raise NotImplementedError


BasicModel = TypeVar("BasicModel", bound=models.BaseModel)
BasicTrainState = TypeVar("BasicTrainState", bound=train_states.BasicTrainState)


class BasicTrainer(BaseTrainer[BasicModel, BasicTrainState]):
  """A basic trainer for models subclassed from `BaseModel`.

  This trainer provides an implementation for the train and evaluation step
  functions, conforming to the abstract interfaces of `BaseModel`. For a
  concrete model, one should additionally define `TrainMetrics` and
  `EvalMetrics` based on the outputs of `model.loss_fn` and `model.eval_fn`
  respectively (see below).
  """

  @flax.struct.dataclass
  class TrainMetrics(BaseTrainer.TrainMetrics):
    """Training metrics definition.

    This class should be defined based on the expected aux output of
    `model.loss_fn`, which is a `(metric_vars, mutables)` tuple. `metric_vars`
    is a dictionary whose entries are used to construct an instance of this
    class. For example, for the following expected format of `metric_vars`::

      metric_vars = {"loss": ..., "accuracy": ..., "prediction": ...}

    one may define::

      class TrainMetrics(BaseTrainer.TrainMetrics):
        train_loss: clu_metrics.Average.from_output("loss")
        train_acc: clu_metrics.Average.from_output("accuracy")
        train_pred: clu_metrics.CollectingMetric.from_outputs(("prediction",))

    which means the `loss`, `accuracy` and `prediction` keys of `metric_vars`
    will be processed to compute logged metrics named `train_loss`, `train_acc`
    and `train_pred` respectively. The type of processing follows the type
    declaration of the class attributes, i.e. averaging for `train_loss` and
    `train_acc`, collecting for `train_pred`. An instance of this class is
    constructed in the train step function as::

      # `single_from_output` for single device
      metrics = TrainMetrics.single_from_output(**metric_vars)

    NOTE: KeyError will be thrown if `metric_vars` does not contain a key
    required per definition of `TrainMetrics`. However, it is fine to have ones
    unused by this class - they will be ignored when passed to
    `single_from_output`.
    """

  @flax.struct.dataclass
  class EvalMetrics(Metrics):
    """Evaluation metrics definition.

    Same as `TrainMetrics`, except applying to the output of `model.eval_fn`.
    """

  def __init__(self, optimizer: optax.GradientTransformation, *args, **kwargs):
    self.optimizer = optimizer
    super().__init__(*args, **kwargs)

  @property
  def train_step(
      self,
  ) -> Callable[
      [BasicTrainState, BatchType, Array],
      tuple[BasicTrainState, Metrics],
  ]:
    def _train_step(
        train_state: BasicTrainState, batch: BatchType, rng: Array
    ) -> tuple[BasicTrainState, Metrics]:
      """Performs gradient step and compute training metrics."""
      grad_fn = jax.grad(self.model.loss_fn, argnums=0, has_aux=True)
      grads, (metrics, mutables) = grad_fn(
          train_state.params, batch, rng, train_state.flax_mutables
      )
      # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
      new_state = self.update_train_state(  # pytype: disable=wrong-keyword-args  # always-use-property-annotation
          train_state=train_state, grads=grads, mutables=mutables
      )
      # pylint: enable=no-value-for-parameter, unexpected-keyword-arg
      metrics_update = self.TrainMetrics.single_from_model_output(**metrics)
      return new_state, metrics_update

    return _train_step

  @property
  def update_train_state(
      self,
  ) -> Callable[[BasicTrainState, VariableDict, VariableDict], BasicTrainState]:
    """Returns function that updates the train state."""

    def _update_train_state(
        train_state: BasicTrainState,
        grads: VariableDict,
        mutables: VariableDict,
    ) -> BasicTrainState:
      updates, new_opt_state = self.optimizer.update(
          grads, train_state.opt_state, train_state.params
      )
      new_params = optax.apply_updates(train_state.params, updates)
      return train_state.replace(
          step=train_state.step + 1,
          opt_state=new_opt_state,
          params=new_params,
          flax_mutables=mutables,
      )

    return _update_train_state

  @property
  def eval_step(
      self,
  ) -> Callable[[BasicTrainState, BatchType, Array], Metrics]:
    def _eval_step(
        train_state: BasicTrainState, batch: BatchType, rng: Array
    ) -> Metrics:
      """Use model to compute the evaluation metrics."""
      eval_metrics = self.model.eval_fn(
          variables=train_state.model_variables, batch=batch, rng=rng
      )
      # `eval_metrics` must include keys required by `EvalMetrics` definition
      return self.EvalMetrics.single_from_model_output(**eval_metrics)

    return _eval_step

  def initialize_train_state(self, rng: Array) -> BasicTrainState:
    """Initializes the model variables and the train state."""
    init_vars = self.model.initialize(rng)
    mutables, params = flax.core.pop(init_vars, "params")
    return train_states.BasicTrainState.create(
        replicate=self.is_distributed,
        params=params,
        opt_state=self.optimizer.init(params),
        flax_mutables=mutables,
    )


def reshape_for_pmap(batch: BatchType) -> BatchType:
  """Reshapes a batch according to the number of local devices."""
  batch_size = jax.tree_util.tree_flatten(batch)[0][0].shape[0]
  if batch_size % jax.local_device_count() != 0:
    raise ValueError(
        f"Batch size {batch_size} is not divisible by device count"
        f" {jax.local_device_count()}!"
    )

  leading_dims = [jax.local_device_count(), -1]
  return jax.tree_util.tree_map(
      lambda x: np.reshape(x, leading_dims + list(x.shape[1:])), batch
  )


class BasicDistributedTrainer(BasicTrainer[BasicModel, BasicTrainState]):
  """The distributed extension of `BasicTrainer`."""

  is_distributed: bool = True

  @property
  def train_step(
      self,
  ) -> Callable[
      [BasicTrainState, BatchType, Array],
      tuple[BasicTrainState, Metrics],
  ]:
    def _train_step(
        train_state: BasicTrainState, batch: BatchType, rng: Array
    ) -> tuple[BasicTrainState, Metrics]:
      """Performs gradient step and compute training metrics."""
      grad_fn = jax.grad(self.model.loss_fn, argnums=0, has_aux=True)
      grads, (metrics, mutables) = grad_fn(
          train_state.params, batch, rng, train_state.flax_mutables
      )
      # Collective operations like `pmean` are computed over GLOBAL devices
      grads = jax.lax.pmean(grads, axis_name=PMAP_AXIS_NAME)
      # No need to modify `update_state` here because pmapped version should
      # just work
      # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
      new_state = self.update_train_state(  # pytype: disable=wrong-keyword-args  # always-use-property-annotation
          train_state=train_state, grads=grads, mutables=mutables
      )
      # pylint: enable=no-value-for-parameter, unexpected-keyword-arg
      # This takes care of both `gather` (local) and `reduce` (global) so a
      # single unreplicate call returns the computed metrics outside
      metrics_update = self.TrainMetrics.gather_from_model_output(
          axis_name=PMAP_AXIS_NAME, **metrics
      )
      return new_state, metrics_update

    return _train_step

  @property
  def eval_step(
      self,
  ) -> Callable[[BasicTrainState, BatchType, Array], Metrics]:
    def _eval_step(
        train_state: BasicTrainState, batch: BatchType, rng: Array
    ) -> Metrics:
      """Use model to compute the evaluation metrics."""
      eval_metrics = self.model.eval_fn(
          variables=train_state.model_variables, batch=batch, rng=rng
      )
      return self.EvalMetrics.gather_from_model_output(
          axis_name=PMAP_AXIS_NAME, **eval_metrics
      )

    return _eval_step

  def preprocess_train_batch(
      self, batch_data: BatchType, step: int, rng: Array
  ) -> BatchType:
    del step, rng
    # Preprocessed batch should always be reshaped for distributed training
    return reshape_for_pmap(batch_data)
