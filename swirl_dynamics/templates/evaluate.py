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

"""Evaluation modules."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
import functools
import time
from typing import Any, Literal, Protocol, Self

from absl import logging
from clu import metric_writers
from clu import metrics as clu_metrics
from clu import periodic_actions
from etils import epath
import flax
import gin
import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.data import zarr_utils
import xarray as xr

filesys = epath.backend.tf_backend

InferenceModel = Any
BatchType = Any
PredType = Any


# Forked from clu.metrics
def check_param(value, *, ndim=None, dtype=jnp.float32):
  """Raises a `ValueError` if `value` does not match ndim/dtype.

  Args:
    value: Value to be tested.
    ndim: Expected dimensions.
    dtype: Expected dtype.

  Raises:
    A `ValueError` if `value` does not match `ndim` or `dtype`, or if `value`
    is not an instance of `jnp.ndarray`.
  """
  if not isinstance(value, (np.ndarray, jnp.ndarray)):
    raise ValueError(f"Expected np.array or jnp.array, got type={type(value)}")
  if ndim is not None and value.ndim != ndim:
    raise ValueError(f"Expected ndim={ndim}, got ndim={value.ndim}")
  if dtype is not None and value.dtype != dtype:
    raise ValueError(f"Expected dtype={dtype}, got dtype={value.dtype}")


# Forked from clu.metrics
def _broadcast_masks(values: jax.Array, mask: jax.Array | None):
  """Checks and broadcasts mask for aggregating values."""
  if values.ndim == 0:
    values = values[None]
  if mask is None:
    mask = jnp.ones_like(values)
  # Leading dimensions of mask and values must match.
  if mask.shape[0] != values.shape[0]:
    raise ValueError(
        "Argument `mask` must have the same leading dimension as `values`. "
        f"Received mask of dimension {mask.shape} "
        f"and values of dimension {values.shape}."
    )
  # Broadcast mask to the same number of dimensions as values.
  if mask.ndim < values.ndim:
    mask = jnp.expand_dims(mask, axis=tuple(np.arange(mask.ndim, values.ndim)))
  mask = mask.astype(bool)
  check_param(mask, dtype=bool, ndim=values.ndim)
  return values, mask


class Benchmark(Protocol):
  """The abstract benchmark task interface.

  This class should be immutable (e.g. frozen dataclass) to facilitate jit
  compiling with jax.
  """

  def run_batch_inference(
      self, inference_fn: InferenceModel, batch: BatchType, rng: jax.Array
  ) -> PredType:
    """Runs the inference task given an inference function and a batch."""
    ...

  def compute_batch_metrics(
      self, pred: PredType, batch: BatchType
  ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Compute metrics given a prediction and a batch.

    This function should generate two types of metrics/results:

      * collect type: a metric/result generated on each test example and
      simply "collected" (without being combined with those of other test
      examples). Examples: conditional sample, the prediction itself.

      * aggregate type: a metric/result first computed on each test example
      and then aggregated (e.g. averaged, taken maximum values, etc.) across all
      test examples. Examples: mean squared error, energy spectra.

    Both types are calculated in this method and returned as dictionaries of
    arrays. They will be collected and aggregated respectively at the evaluator
    level.

    Args:
      pred: The predictions made by a benchmarked model.
      batch: The evaluation batch data containing the groud truth labels.

    Returns:
      A tuple of `(collect, aggregate)` batch evaluation results.
    """
    ...


def TensorAverage(  # pylint: disable=invalid-name
    axis: int | tuple[int, ...] | None = None, rms: bool = False
) -> type[clu_metrics.Average]:
  """Defines a metric that computes the average only along selected axes.

  Extends `clu.metrics.Average`, which *always* averages across all axes.
  Instead, only entries on the selected axes are averaged (with equal weights),
  while the remaining axes stay untouched (which means their dimensions will be
  part of the final result obtained after the aggregating `.compute()` call).

  Args:
    axis: The axis or axes along which the average is computed. If `None`,
      average is taken across all axes.
    rms: Whether to compute the root mean square (RMS) of the input values. If
      `False`, the regular average is computed.

  Returns:
    The metric class.
  """

  @flax.struct.dataclass
  class _TensorAverage(clu_metrics.Average):
    """Tensor Average metric class."""

    @classmethod
    def from_model_output(
        cls, values: jax.Array, mask: jax.Array | None = None, **_
    ) -> clu_metrics.Average:
      """Construct a metric instance given model output values."""
      values = jnp.square(values) if rms else values
      values, mask = _broadcast_masks(values, mask)
      return cls(
          total=jnp.where(mask, values, jnp.zeros_like(values)).sum(axis=axis),
          count=jnp.where(
              mask,
              jnp.ones_like(values, dtype=jnp.int32),
              jnp.zeros_like(values, dtype=jnp.int32),
          ).sum(axis=axis),
      )

    def merge(self, other: clu_metrics.Average) -> clu_metrics.Average:
      """Merges with another metric instance of the same type."""
      return type(self)(
          total=self.total + other.total,
          count=self.count + other.count,
      )

    def compute(self) -> jax.Array:
      res = self.total / self.count
      return jnp.sqrt(res) if rms else res

  return _TensorAverage


def TensorRatio(  # pylint: disable=invalid-name
    axis: int | tuple[int, ...] | None = None,
):
  """Computes the ratio between two aggregated metrics.

  The ratio is performed after full aggregation of numerator and denominator.
  For numerator and denominator, only entries on the selected axes are
  aggregated, while the remaining axes stay untouched (which means their
  dimensions will be part of the final result obtained after the aggregating
  `.compute()` call).

  Args:
    axis: The axis or axes along which numerator and denominator are aggregated.
      If `None`, average is taken across all axes.

  Returns:
    The metric class.
  """

  @flax.struct.dataclass
  class _TensorRatio(clu_metrics.Metric):
    """Ratio of metrics class."""

    numerator: jax.Array
    denominator: jax.Array

    @classmethod
    def from_model_output(
        cls,
        *,
        numerator: jax.Array,
        denominator: jax.Array,
        mask: jax.Array | None = None,
        **_,
    ) -> Self:
      """Construct a metric instance given model output values."""
      numerator, mask = _broadcast_masks(numerator, mask)
      return cls(
          numerator=jnp.where(mask, numerator, jnp.zeros_like(numerator)).sum(
              axis=axis
          ),
          denominator=jnp.where(
              mask, denominator, jnp.zeros_like(denominator)
          ).sum(axis=axis),
      )

    def merge(self, other):
      """Merges with another metric instance of the same type."""
      return type(self)(
          numerator=self.numerator + other.numerator,
          denominator=self.denominator + other.denominator,
      )

    def compute(self) -> jax.Array:
      ratio = self.numerator / self.denominator
      return ratio

    @classmethod
    def empty(cls) -> Self:
      return cls(
          numerator=jnp.array(0, jnp.float32),
          denominator=jnp.array(0, jnp.float32),
      )

    @classmethod
    def from_outputs(cls, numerator_name: str, denominator_name: str):
      """Calls `cls.from_model_output` with model output names.

      Synopsis:

        @flax.struct.dataclass
        class Metrics(Collection):
          loss: TensorRatio(axis=None).from_outputs("foo", "bar")

      Args:
        numerator_name: Name of the model output that should be passed as the
          `numerator` keyword argument to `cls.from_model_output()`.
        denominator_name: Name of the model output that should be passed as the
          `denominator` keyword argument to `cls.from_model_output()`.

      Returns:
        A `Metric` derived from `cls` that calls `.from_model_output()` with the
        the specified numerator and denominator arguments.
      """

      @flax.struct.dataclass
      class FromOutputs(cls):
        """Wrapper Metric class that collects numerator and denominator."""

        @classmethod
        def from_model_output(cls, **model_output):
          numerator = jnp.array(model_output[numerator_name])
          denominator = jnp.array(model_output[denominator_name])
          mask = model_output.get("mask")
          if mask is not None and (numerator.shape or [0])[0] != mask.shape[0]:
            logging.warning(
                "Ignoring mask for model output '%s' because of shape mismatch:"
                " output.shape=%s vs. mask.shape=%s",
                numerator_name,
                numerator.shape,
                mask.shape,
            )
            mask = None
          return super().from_model_output(
              numerator=numerator, denominator=denominator, mask=mask
          )

      return FromOutputs

  return _TensorRatio


class CollectingResults:
  """Object that holds collected evaluation results.

  Results collected from batches are concatenated along the leading axis.
  Results for different evaluated models are stored as a dict (using their names
  as the key) in the `results` field. The model keys and metric fields are
  automatically initialized when `.collect_batch_result()` is called for the
  first time.
  """

  def __init__(self):
    self.metric_cls = None
    self.results = {}

  @property
  def model_keys(self) -> list[str]:
    return list(self.results.keys())

  @property
  def is_empty(self) -> bool:
    return not self.results

  def collect_batch_result(
      self, key: str, batch_result: Mapping[str, jax.Array]
  ) -> None:
    """Collects the results from a single batch and a single model."""
    if self.metric_cls is None:
      self.metric_cls = clu_metrics.CollectingMetric.from_outputs(
          list(batch_result.keys())
      )
    batch_update = self.metric_cls.from_model_output(**batch_result)
    if key not in self.model_keys:
      self.results[key] = batch_update
    else:
      self.results[key] = self.results[key].merge(batch_update)

  def compute(self) -> dict[str, dict[str, jax.Array]]:
    """Returns all the collected results concatenated along axis 0."""
    if self.is_empty:
      raise RuntimeError(
          "Metric class is not initialized. Make sure that"
          " `.collect_batch_result()` is called at least once!"
      )
    return {k: v.compute() for k, v in self.results.items()}


class EvalState(flax.struct.PyTreeNode):
  """Evaluation state that holds the aggregated metrics for all models."""

  step: int
  aggregated_metrics: dict[str, clu_metrics.Collection]

  @property
  def model_keys(self) -> list[str]:
    return list(self.aggregated_metrics.keys())

  def compute_aggregated_metrics(self) -> dict[str, dict[str, jax.Array]]:
    """Computes and returns the aggregated metrics for all models."""
    return {k: v.compute() for k, v in self.aggregated_metrics.items()}

  def aggregate_batch_result(
      self, batch_update: Mapping[str, clu_metrics.Collection]
  ) -> "EvalState":
    """Aggregates batch results."""
    return self.replace(
        step=self.step + 1,
        aggregated_metrics={
            key: self.aggregated_metrics[key].merge(batch_update[key])
            for key in batch_update.keys()
        },
    )

  @classmethod
  def create(
      cls, model_keys: Sequence[str], metric_cls: type[clu_metrics.Collection]
  ) -> "EvalState":
    return cls(
        step=0,
        aggregated_metrics={key: metric_cls.empty() for key in model_keys},
    )


class EndOfDataset(Exception):  # pylint: disable=g-bad-exception-name
  """Exception to signal that the evaluation dataset has run out."""

  def __init__(self, message: str, collected: CollectingResults, step: int):
    super().__init__(message)
    self.collected = collected
    self.step = step


class Evaluator:
  """Base evaluator class.

  This object runs a dictionary of inference models through a `Benchmark`, and
  collects the resulting metrics.

  For use with a specific benchmark, one should subclass and define the
  `AggregatingMetrics` class attribute. The collect-type metrics are
  initialized dynamically at runtime (see `Benchmark.compute_batch_metrics()`
  for an explanation of the distinctions).
  """

  @flax.struct.dataclass
  class AggregatingMetrics(clu_metrics.Collection):
    """Declaration for the aggregated metrics of the benchmark.

    This class should be defined based on the expected output of
    `benchmark.compute_batch_results()`, which is a `(collect, aggregate)`
    tuple. `aggregate` is a dictionary whose entries are used to construct an
    instance of this class. For example, for the following expected schema::

      aggregate_update = {"error": ...}

    one may define::

      @flax.struct.dataclass
      class AggregatingMetrics(clu_metrics.Collection):
        mean_error: TensorAverage(axis=-1).from_output("error")

    which means the `error` key of `aggregate_update` will be used to compute
    metrics named `mean_error`. The type of aggregation follows the type
    declaration of the class attributes, i.e. averaging along the last axis for
    `TensorAverage(axis=-1)`. Instances of this class will be created in
    `.evaluate()` and aggregated in batches.

    (Note that `KeyError` will be thrown if `aggregate_update` does not contain
    a key required per the  definition of `AggregatingMetrics` - in this case,
    "error". If `aggregate_update` contains additional keys, they will simply
    be ignored.)
    """

  def __init__(
      self,
      models: Mapping[str, InferenceModel],
      benchmark: Benchmark,
      rng: jax.Array,
  ) -> None:
    self.benchmark = benchmark
    self.rng = rng
    self.state = EvalState.create(
        model_keys=list(models.keys()), metric_cls=self.AggregatingMetrics
    )
    self._compiled_inf_fns = {
        key: jax.jit(
            functools.partial(self.benchmark.run_batch_inference, model)
        )
        for key, model in models.items()
    }
    self._compiled_metrics_compute = jax.jit(
        self.benchmark.compute_batch_metrics
    )

  def evaluate(
      self, iterator: Iterator[BatchType], num_batches: int
  ) -> CollectingResults:
    """Runs evaluation for a specified number of batches."""
    collected = CollectingResults()
    for i in range(num_batches):
      try:
        batch = next(iterator)
      except StopIteration as e:
        raise EndOfDataset(
            "Reached the end of dataset.", collected=collected, step=i
        ) from e
      else:
        rng = jax.random.fold_in(self.rng, self.state.step)
        batch_agg_update = {}
        for key, inf_fn in self._compiled_inf_fns.items():
          inference_rng = jax.random.fold_in(
              rng, np.int32(hash(key) % (2**31 - 1))  # Prevent overflows.
          )
          pred = inf_fn(batch, inference_rng)
          batch_collect, batch_res = self._compiled_metrics_compute(pred, batch)
          collected.collect_batch_result(key, batch_collect)
          batch_agg_update[key] = (
              self.AggregatingMetrics.single_from_model_output(**batch_res)
          )

        self.state = self.state.aggregate_batch_result(batch_agg_update)
    return collected

  @property
  def scalar_metrics_to_log(self) -> dict[str, jax.Array]:
    """Scalar metrics that will be logged to tensorboard."""
    return {}


PYGRAIN_CHECKPOINTER = checkpoint.Checkpointer(
    pygrain.PyGrainCheckpointHandler()
)  # pytype:disable=wrong-arg-types


def run(
    *,
    evaluator: Evaluator,
    dataloader: Iterable[Any],
    workdir: epath.PathLike,
    num_aggregation_batches: int,
    max_eval_batches: int | None = 10000,
    dump_collected_every_n_groups: int = 0,
    log_gin_config_str: bool = False,
    enable_checkpoints: bool = True,
    data_checkpointer: checkpoint.Checkpointer = PYGRAIN_CHECKPOINTER,
    checkpoint_options: checkpoint.CheckpointManagerOptions | None = None,
    results_format: Literal["hdf5", "zarr"] = "hdf5",
    datacoords: xr.core.coordinates.DatasetCoordinates | None = None,
) -> None:
  """Runs a benchmark evaluation.

  This function runs the benchmark evaluation in bunches, or "groups of
  batches" (where group size = `num_aggregation_batches`). After a group is done
  evaluating, the results are (optionally) saved/checkpointed. At the end of the
  whole evaluation (either by reaching `max_eval_batches` or the dataloader
  raising `StopIteration`), the aggregated metrics are saved.

  Args:
    evaluator: An evaluator object containing the configs and logic for the
      inference task, benchmarked models and metric/results computation.
    dataloader: A dataloader that emits the data batches for evaluation.
    workdir: The working directory where evaluation results and checkpoints are
      saved.
    num_aggregation_batches: The evaluator runs this many batches at a time,
      after which result dumping and checkpointing may take place (based on
      their respective configs).
    max_eval_batches: The maximum of evaluation batches to run. When set to
      `None`, will run until data is exhausted (i.e. `StopIteration` raised by
      dataloader).
    dump_collected_every_n_groups: The frequency at which the collect-type
      evaluation results will be dumped to disc. Dumping may only happen after
      each group of batches finishes evaluating and applies to the entire group.
      For example, if `num_aggregation_batches = 10` and `dump_every_n_groups =
      2`, results for batch 10-20, 30-40, 50-60, ... are dumped, each to a
      separate hdf5 file.
    log_gin_config_str: Whether to log gin config str. If `True`, the operative
      gin config is written as tf events and can be visualized under the "text"
      tab on tensorboard.
    enable_checkpoints: Whether to enable checkpointing of the evaluation
      progress (i.e. state of the data iterator and aggregated results).
    data_checkpointer: The orbax checkpointer for checkpointing the data
      iterator.
    checkpoint_options: The orbax checkpoint options (see
      `orbax.checkpoint.CheckpointManagerOptions`).
    results_format: Writing format for results. It can be `"zarr"` or `"hdf5"`.
    datacoords: The coordinates of the dataset being evaluated, used to annotate
      collected results.
  """
  # **** Initialization ****
  workdir = epath.Path(workdir)
  if not filesys.exists(workdir):
    filesys.makedirs(workdir)

  if dump_collected_every_n_groups < 0:
    raise ValueError(
        "`dump_every_n_groups` must not be negative:"
        f" {dump_collected_every_n_groups}"
    )
  dump_period = dump_collected_every_n_groups * num_aggregation_batches

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  progress_report = periodic_actions.ReportProgress(writer=writer)
  if log_gin_config_str:
    config_str = gin.operative_config_str()
    writer.write_texts(
        0, {"config": gin.markdown(config_str), "raw_config_str": config_str}
    )

  ckpt_manager = None
  if enable_checkpoints:
    ckpt_manager = checkpoint.CheckpointManager(
        directory=workdir / "checkpoints",
        checkpointers=dict(
            iterator=data_checkpointer,
            eval_state=checkpoint.PyTreeCheckpointer(),
        ),
        options=checkpoint_options,
    )

  results_subfolder = workdir / "results"
  if not filesys.exists(results_subfolder):
    filesys.makedirs(results_subfolder)

  # **** Stepping through batch ****
  iterator = iter(dataloader)
  cur_step = 0

  # restore checkpoint state if applicable
  if enable_checkpoints and ckpt_manager.latest_step():  # pytype: disable=attribute-error
    ckpt = ckpt_manager.restore(  # pytype: disable=attribute-error
        ckpt_manager.latest_step(),  # pytype: disable=attribute-error
        items=dict(iterator=iterator, eval_state=evaluator.state),
    )
    evaluator.state = ckpt["eval_state"]
    iterator = ckpt["iterator"]
    cur_step = evaluator.state.step

  max_eval_batches = max_eval_batches or np.inf
  is_finished = cur_step >= max_eval_batches if not max_eval_batches else False
  ran_out = False

  while not is_finished and not ran_out:
    try:
      collected = evaluator.evaluate(iterator, num_aggregation_batches)

    except EndOfDataset as eod:
      cur_step += eod.step
      ran_out = True
      collected = eod.collected

      logging.info("Reached the end of dataset at batch number %d.", cur_step)

    else:
      cur_step += num_aggregation_batches
      if cur_step >= max_eval_batches:
        is_finished = True

    finally:
      progress_report(cur_step, time.time())
      writer.write_scalars(cur_step, evaluator.scalar_metrics_to_log)

      # save checkpoints
      if enable_checkpoints:
        assert ckpt_manager is not None
        ckpt_manager.save(
            step=cur_step,
            items=dict(
                iterator=iterator,
                eval_state=jax.tree.map(np.array, evaluator.state),
            ),
        )

      # dump collected
      if dump_period and (ran_out or cur_step % dump_period == 0):
        if results_format == "hdf5":
          batch_path = results_subfolder / f"batch_{cur_step}.hdf5"
          if not filesys.exists(batch_path):
            hdf5_utils.save_array_dict(batch_path, collected.compute())  # pylint: disable=undefined-variable
        elif results_format == "zarr":
          for key, value in collected.compute().items():  # pylint: disable=undefined-variable
            zarr_utils.collected_metrics_to_zarr(
                value,
                out_dir=results_subfolder,
                basename=f"{key}_collected_metrics",
                append_dim="time",
                coords=datacoords,
                append_slice=slice(
                    cur_step - num_aggregation_batches, cur_step
                ),
            )
        else:
          raise ValueError(f"Unknown results format: {results_format}")

  if results_format == "hdf5":
    # save final metrics to hdf5
    agg_metric_path = results_subfolder / "final_aggregated_metrics.hdf5"
    hdf5_utils.save_array_dict(
        agg_metric_path, evaluator.state.compute_aggregated_metrics()
    )
  elif results_format == "zarr":
    # save final metrics to zarr
    metrics = evaluator.state.compute_aggregated_metrics()
    metrics = jax.tree.map(np.asarray, metrics)
    for key, value in metrics.items():
      zarr_utils.aggregated_metrics_to_zarr(
          value,
          out_dir=results_subfolder,
          basename=f"{key}_aggregated_metrics",
          coords=datacoords,
      )
  else:
    raise ValueError(f"Unknown results format: {results_format}")
