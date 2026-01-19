# Copyright 2025 The swirl_dynamics Authors.
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
from collections.abc import Callable, Iterator, Mapping, Sequence
import functools
import itertools
from typing import Any, Generic, TypeAlias, TypeVar

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states

Array: TypeAlias = jax.Array
PyTree: TypeAlias = Any
BatchType: TypeAlias = Mapping[str, PyTree]
Metrics: TypeAlias = clu_metrics.Collection
VariableDict: TypeAlias = train_states.FrozenVariableDict

M = TypeVar("M")  # Model
S = TypeVar("S", bound=train_states.TrainState)  # Train state


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

  def __init__(self, model: M, rng: Array, num_steps_to_compile: int = 1):
    """Initializer.

    Args:
      model: The model to train and evaluate.
      rng: A PRNG key for the trainer.
      num_steps_to_compile: The number of calls to train_step that should be
        unrolled and compiled into a JAX for loop when running train(...,
        num_steps). This amortizes the cost of starting function execution on an
        accelerator over this number of steps.

    Raises:
      ValueError: if num_steps_to_compile is invalid.
    """
    self.model = model
    self._train_rng, self._eval_rng, self._init_rng = jax.random.split(rng, 3)

    if num_steps_to_compile < 1:
      raise ValueError(f"{num_steps_to_compile=} must be positive.")

    self._num_steps_to_compile = num_steps_to_compile
    self._train_state = self.initialize_train_state(self._init_rng)

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

  # ***************************************
  # JIT-compilation of train and eval steps
  # ***************************************

  @functools.cached_property
  def _compiled_train_step(
      self,
  ) -> Callable[[Array, S, BatchType, Metrics], tuple[S, Metrics]]:
    """Compiles training step function."""
    return jax.jit(self._multi_train_step)

  @functools.cached_property
  def _compiled_eval_step(self) -> Callable[[S, BatchType, Array], Metrics]:
    """Compiles evaluation step function."""
    return jax.jit(self.eval_step)

  # ******************************
  # Base train and eval loop logic
  # ******************************

  def train(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
    """Runs training for a specified number of steps.

    Runs num_steps steps of train_step using data from batch_iter. If
    num_steps_to_compile is larger than 1, then that many steps are unrolled and
    precompiled for more efficient execution.

    Args:
      batch_iter: An iterator over batches of training data. Each element should
        correspond to all the examples used by the process where this is called.
        When running distributed training, this method will reshape the batches
        into a shape suitable for pmap over the devices controlled by the
        process.
      num_steps: Number of training steps before collecting metrics. Must be a
        multiple of num_steps_to_compile.

    Returns:
      The metrics at the end of running num_steps training steps.
    """
    # `.int_step` may involve unreplicate (expensive), so we do it only once.
    step0 = self.train_state.int_step
    if num_steps % self._num_steps_to_compile != 0:
      raise ValueError(
          f"{num_steps=} must be an exact multiple of"
          f" {self._num_steps_to_compile=}."
      )

    train_rng, data_rng = self.get_train_rng(step=step0, num=2)
    super_batch_iter = self._get_super_batch_iter(batch_iter, step0, data_rng)
    train_metrics = self.TrainMetrics.empty()

    for cur_step in range(step0, step0 + num_steps, self._num_steps_to_compile):
      with jax.profiler.StepTraceAnnotation("train", step_num=cur_step):
        super_batch = next(super_batch_iter)
        step_train_rng = jax.random.fold_in(train_rng, cur_step)
        self.train_state, train_metrics = self._compiled_train_step(
            step_train_rng, self.train_state, super_batch, train_metrics
        )
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
          self.train_state, batch, eval_rng
      )
      # This runs on CPU to accommodate potential collecting-type metrics.
      eval_metrics = eval_metrics.merge(metrics_update)
    return eval_metrics

  @property
  def _multi_train_step(self):
    """Returns a multi-step training function based on the single-step one."""

    def _train_multistep(
        train_rng: Array, train_state: S, batches: BatchType, metrics: Metrics
    ) -> tuple[S, Metrics]:
      """Runs multiple training steps via `jax.lax.scan`."""
      rngs = jax.random.split(train_rng, self._num_steps_to_compile)

      def _step(state: tuple[S, Metrics], inputs: tuple[BatchType, Array]):
        batch, rng = inputs
        state, metrics = state
        new_state, step_metrics = self.train_step(state, batch, rng)
        metrics = metrics.merge(step_metrics)
        return (new_state, metrics), None

      (train_state, metrics), _ = jax.lax.scan(
          f=_step, init=(train_state, metrics), xs=(batches, rngs)
      )
      return train_state, metrics

    return _train_multistep

  def _get_super_batch_iter(
      self, batch_iter: Iterator[BatchType], step0: int, rng: Array
  ) -> Iterator[BatchType]:
    """Collects `num_steps_to_compile` batches into a single stacked element."""
    # Moves rng to CPU to avoid CPU<->TPU round-trip.
    rng = jax.device_put(rng, device=jax.local_devices(backend="cpu")[0])

    def add_preproc(batch_iter: Iterator[BatchType]) -> Iterator[BatchType]:
      """Adds preprocessing to the batch iterator."""
      for step, batch in enumerate(batch_iter):
        step_rng = jax.random.fold_in(rng, step)
        yield self.preprocess_train_batch(batch, step0 + step, step_rng)

    def stack_batches(
        super_iter: Iterator[Sequence[BatchType]],
    ) -> Iterator[BatchType]:
      """Stacks batches along a new leading dimension."""
      for super_batch in super_iter:
        stacked_batch = jax.tree.map(lambda *arrs: np.stack(arrs), *super_batch)
        yield stacked_batch

    super_iter = stack_batches(
        itertools.batched(add_preproc(batch_iter), self._num_steps_to_compile)
    )
    return super_iter

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
  def train_step(self) -> Callable[[S, BatchType, Array], tuple[S, Metrics]]:
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
      [BasicTrainState, BatchType, Array], tuple[BasicTrainState, Metrics]
  ]:
    """Returns the one-step training function."""

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
    """Returns the one-step evaluation function."""

    def _eval_step(
        train_state: BasicTrainState, batch: BatchType, rng: Array
    ) -> Metrics:
      """Use model to compute the evaluation metrics."""
      eval_metrics = self.model.eval_fn(
          variables=train_state.model_variables, batch=batch, rng=rng
      )
      return self.EvalMetrics.single_from_model_output(**eval_metrics)

    return _eval_step

  def initialize_train_state(self, rng: Array) -> BasicTrainState:
    """Initializes the model variables and the train state."""
    init_vars = self.model.initialize(rng)
    mutables, params = flax.core.pop(init_vars, "params")
    return train_states.BasicTrainState.create(
        params=params,
        opt_state=self.optimizer.init(params),
        flax_mutables=mutables,
    )


P = jax.sharding.PartitionSpec


def _make_global_from_process_local(
    pytree: PyTree,
    data_sharding: jax.sharding.Sharding,
    batch_axis_index: int,
) -> PyTree:
  """Makes a global PyTree from a process-local PyTree."""

  def _build_global_array(array: np.ndarray):
    if array.ndim <= batch_axis_index:
      raise ValueError(
          f"Array ndim {array.ndim} must be greater than {batch_axis_index=}."
      )
    local_batch_size = array.shape[batch_axis_index]
    global_batch_size = local_batch_size * jax.process_count()
    global_shape = list(array.shape)
    global_shape[batch_axis_index] = global_batch_size
    global_shape = tuple(global_shape)
    return jax.make_array_from_process_local_data(
        data_sharding, array, global_shape
    )

  return jax.tree.map(_build_global_array, pytree)


class BasicDistributedTrainer(BasicTrainer[BasicModel, BasicTrainState]):
  """The data-parallel extension of `BasicTrainer`."""

  is_distributed: bool = True
  batch_axis: str = "batch"

  @functools.cached_property
  def mesh(self) -> jax.sharding.Mesh:
    return jax.sharding.Mesh(jax.devices(), axis_names=(self.batch_axis,))

  def train(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
    """Runs training for a specified number of steps."""
    step0 = self.train_state.int_step
    if num_steps % self._num_steps_to_compile != 0:
      raise ValueError(
          f"{num_steps=} must be an exact multiple of"
          f" {self._num_steps_to_compile=}."
      )

    train_rng, data_rng = self.get_train_rng(step=step0, num=2)
    super_batch_iter = self._get_super_batch_iter(batch_iter, step0, data_rng)

    # Creates an empty metrics object for all devices.
    train_metrics = jax.device_put(
        jax.tree.map(
            lambda x: jnp.broadcast_to(x, (jax.device_count(),) + x.shape),
            self.TrainMetrics.empty(),
        ),
        jax.sharding.NamedSharding(self.mesh, P(self.batch_axis)),
    )

    for cur_step in range(step0, step0 + num_steps, self._num_steps_to_compile):
      with jax.profiler.StepTraceAnnotation("train", step_num=cur_step):
        super_batch = next(super_batch_iter)
        # Abstractly merges the process-local batch into a single global batch
        # so that it can be sharded by shard_map.
        super_global_batch = _make_global_from_process_local(
            super_batch,
            # Shards along the 2nd axis (batch; 1st axis is multi-step).
            jax.sharding.NamedSharding(self.mesh, P(None, self.batch_axis)),
            batch_axis_index=1,
        )
        train_step_rng = jax.random.fold_in(train_rng, cur_step)
        self.train_state, train_metrics = self._compiled_train_step(
            train_step_rng, self.train_state, super_global_batch, train_metrics
        )
    return train_metrics.reduce()

  @functools.cached_property
  def train_step(
      self,
  ) -> Callable[
      [BasicTrainState, BatchType, Array], tuple[BasicTrainState, Metrics]
  ]:
    """Returns the one-step training function."""

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=(P(), P(self.batch_axis), P()),
        out_specs=(P(), P(self.batch_axis)),
    )
    def _train_step(
        train_state: BasicTrainState, batch: BatchType, rng: Array
    ) -> tuple[BasicTrainState, Metrics]:
      """Performs gradient step and compute training metrics."""
      # Note there is no need to fold in process index in the root rng.
      rng = jax.random.fold_in(rng, jax.lax.axis_index(self.batch_axis))
      grad_fn = jax.grad(self.model.loss_fn, argnums=0, has_aux=True)
      grads, (metrics, mutables) = grad_fn(
          train_state.params, batch, rng, train_state.flax_mutables
      )
      # Collective operations like `pmean` are computed over GLOBAL devices
      grads = jax.lax.pmean(grads, axis_name=self.batch_axis)
      mutables = jax.lax.pmean(mutables, axis_name=self.batch_axis)

      # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
      new_state = self.update_train_state(  # pytype: disable=wrong-keyword-args  # always-use-property-annotation
          train_state=train_state, grads=grads, mutables=mutables
      )
      # pylint: enable=no-value-for-parameter, unexpected-keyword-arg
      # Avoids excessive all-gather operations by computing metrics on shards
      # and aggregating later outside.
      metrics_update = self.TrainMetrics.single_from_model_output(**metrics)
      metrics_update = jax.tree.map(
          lambda x: jnp.expand_dims(x, axis=0), metrics_update
      )
      return new_state, metrics_update

    return _train_step

  def eval(self, batch_iter: Iterator[BatchType], num_steps: int) -> Metrics:
    """Runs evaluation for a specified number of steps."""
    data_sharding = jax.sharding.NamedSharding(self.mesh, P(self.batch_axis))
    eval_metrics = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (jax.device_count(),) + x.shape),
        self.EvalMetrics.empty(),
    )
    eval_metrics = jax.device_put(eval_metrics, data_sharding)
    for _ in range(num_steps):
      eval_rng, preproc_rng = self.get_eval_rng(num=2)
      batch = self.preprocess_eval_batch(
          batch_data=next(batch_iter), rng=preproc_rng
      )
      global_batch = _make_global_from_process_local(
          batch, data_sharding, batch_axis_index=0
      )
      metrics_update = self._compiled_eval_step(
          self.train_state, global_batch, eval_rng
      )
      # This runs on CPU to accommodate potential collecting-type metrics.
      eval_metrics = eval_metrics.merge(metrics_update)
    return eval_metrics.reduce()

  @functools.cached_property
  def eval_step(
      self,
  ) -> Callable[[BasicTrainState, BatchType, Array], Metrics]:
    """Returns the one-step evaluation function."""

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=(P(), P(self.batch_axis), P()),
        out_specs=P(self.batch_axis),
    )
    def _eval_step(
        train_state: BasicTrainState, batch: BatchType, rng: Array
    ) -> Metrics:
      """Use model to compute the evaluation metrics."""
      rng = jax.random.fold_in(rng, jax.lax.axis_index(self.batch_axis))
      eval_metrics = self.model.eval_fn(
          variables=train_state.model_variables, batch=batch, rng=rng
      )
      metrics_update = self.EvalMetrics.single_from_model_output(**eval_metrics)
      return jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), metrics_update)

    return _eval_step
