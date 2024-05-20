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

import itertools
import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from clu import metrics as clu_metrics
from etils import epath
import flax
import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint
from swirl_dynamics.data import hdf5_utils
from swirl_dynamics.templates import evaluate
import xarray as xr

FLAGS = flags.FLAGS


class TensorAverageTest(parameterized.TestCase):

  @parameterized.parameters(
      ((1, 2),),
      ((0, 1, 2, 3, 4),),
      (None,),
      (3,),
  )
  def test_from_model_output(self, mean_axis):
    test_shape = (1, 2, 3, 4, 5)
    metric_cls = evaluate.TensorAverage(mean_axis).from_output("abc")
    rng = np.random.default_rng(123)
    test_values = rng.random(test_shape)
    metric = metric_cls.from_model_output(abc=test_values)
    metric_compute = metric.compute()
    np.testing.assert_allclose(
        metric_compute, np.mean(test_values, axis=mean_axis), atol=1e-5
    )

  @parameterized.parameters(
      ((1, 2),),
      ((0, 1, 2, 3, 4),),
      (None,),
      (3,),
  )
  def test_from_model_output_with_mask(self, mean_axis):
    test_shape = (1, 2, 3, 4, 5)
    mask = np.ones(test_shape)
    mask[0, 0, :, 0, 0] = 0
    metric_cls = evaluate.TensorAverage(mean_axis).from_output("abc")
    rng = np.random.default_rng(123)
    test_values = rng.random(test_shape)
    metric = metric_cls.from_model_output(abc=test_values, mask=mask)
    metric_compute = metric.compute()
    # A masked numpy array ignores the masked elements, opposite to our mask.
    expected = np.ma.array(test_values, mask=np.logical_not(mask)).mean(
        axis=mean_axis
    )
    np.testing.assert_allclose(metric_compute, expected, atol=1e-5)

  @parameterized.parameters(
      ((1, 2),),
      ((0, 1, 2, 3, 4),),
      (None,),
      (3,),
  )
  def test_merge(self, mean_axis):
    test_shape = (1, 2, 3, 4, 5)
    metric_cls = evaluate.TensorAverage(mean_axis).from_output("abc")
    metric = metric_cls.empty()
    rng = np.random.default_rng(321)
    test_values1 = rng.random(test_shape)
    metric = metric.merge(metric_cls.from_model_output(abc=test_values1))
    test_values2 = rng.random(test_shape)
    metric = metric.merge(metric_cls.from_model_output(abc=test_values2))
    metric_compute = metric.compute()
    expected = np.mean(
        np.concatenate(
            [test_values1[..., None], test_values2[..., None]], axis=-1
        ),
        axis=mean_axis,
    )
    if mean_axis is not None:
      expected = np.mean(expected, axis=-1)
    np.testing.assert_allclose(metric_compute, expected, atol=1e-5)

  def test_rms(self):
    test_shape = (10,)
    num_batches = 10
    metric_cls = evaluate.TensorAverage(rms=True).from_output("abc")
    metric = metric_cls.empty()
    rng = np.random.default_rng(987)
    test_values = []
    for _ in range(num_batches):
      test_values.append(rng.random(test_shape))
      metric = metric.merge(metric_cls.from_model_output(abc=test_values[-1]))
    metric_compute = metric.compute()
    expected = np.sqrt(np.mean(np.square(np.concatenate(test_values))))
    np.testing.assert_allclose(metric_compute, expected, atol=1e-5)


@parameterized.parameters(
    ((1, 2),),
    (None,),
    (3,),
)
class TensorRatioTest(parameterized.TestCase):

  def test_from_model_output(self, agg_axis):
    test_shape = (1, 2, 3, 4, 5)
    metric_cls = evaluate.TensorRatio(agg_axis).from_outputs("abc", "efg")
    rng = np.random.default_rng(123)
    test_values = rng.random(test_shape)
    expected = 2 * np.ones_like(np.mean(test_values, axis=agg_axis))
    metric = metric_cls.from_model_output(abc=2 * test_values, efg=test_values)
    metric_compute = metric.compute()
    np.testing.assert_allclose(metric_compute, expected, atol=1e-4)

  def test_merge(self, agg_axis):
    test_shape = (1, 2, 3, 4, 5)
    metric_cls = evaluate.TensorRatio(agg_axis).from_outputs("abc", "efg")
    metric = metric_cls.empty()
    rng = np.random.default_rng(321)
    test_values1 = rng.random(test_shape)
    test_values2 = rng.random(test_shape)
    metric = metric.merge(
        metric_cls.from_model_output(abc=test_values1, efg=test_values2)
    )
    test_values3 = rng.random(test_shape)
    test_values4 = rng.random(test_shape)
    metric = metric.merge(
        metric_cls.from_model_output(abc=test_values3, efg=test_values4)
    )
    metric_compute = metric.compute()
    # The expected result is the ratio of the sums of numerators and
    # denominators, which does not commute with the sum of ratios.
    expected_numerator = np.add(
        np.sum(test_values1, axis=agg_axis), np.sum(test_values3, axis=agg_axis)
    )
    expected_denominator = np.add(
        np.sum(test_values2, axis=agg_axis), np.sum(test_values4, axis=agg_axis)
    )
    expected = np.divide(expected_numerator, expected_denominator)
    np.testing.assert_allclose(metric_compute, expected, atol=1e-5)


class CollectingResultsTest(parameterized.TestCase):

  def test_collect_batches_and_compute(self):
    num_batches = 5
    batch_size = 8
    batch_result = {
        "field0": jnp.ones((batch_size, 4)),
        "field1": jnp.ones((batch_size, 4, 4)),
    }

    collection = evaluate.CollectingResults()
    for _ in range(num_batches):
      collection.collect_batch_result(key="model0", batch_result=batch_result)

    collected = collection.compute()
    tot = num_batches * batch_size
    self.assertEqual(collected["model0"]["field0"].shape, (tot, 4))
    self.assertEqual(collected["model0"]["field1"].shape, (tot, 4, 4))

  def test_raise_compute_when_uninitialized(self):
    with self.assertRaises(RuntimeError):
      evaluate.CollectingResults().compute()


@flax.struct.dataclass
class TestAggMetrics(clu_metrics.Collection):
  mean: evaluate.TensorAverage().from_output("abc")
  tensor_mean: evaluate.TensorAverage(axis=-1).from_output("abc")


class EvalStateTest(parameterized.TestCase):

  def test_aggregate_batch_results_and_compute(self):
    test_shape = (1, 2, 3, 4, 5)
    num_batches = 5
    eval_output = {"abc": jnp.ones(test_shape)}
    model_keys = ["model0"]
    state = evaluate.EvalState.create(model_keys, TestAggMetrics)
    for _ in range(num_batches):
      state = state.aggregate_batch_result(
          {
              m: TestAggMetrics.single_from_model_output(**eval_output)
              for m in model_keys
          }
      )

    aggregated = state.compute_aggregated_metrics()
    for m in model_keys:
      self.assertEqual(aggregated[m]["mean"], 1.0)
      np.testing.assert_allclose(
          aggregated[m]["tensor_mean"], np.ones(test_shape[:-1]), atol=1e-5
      )


class DummyBenchmark:

  def run_batch_inference(self, inference_fn, batch, rng):
    del rng
    return inference_fn(batch)

  def compute_batch_metrics(self, pred, batch):
    return {"collect0": batch}, {"abc": pred}


class DummyEvaluator(evaluate.Evaluator):
  AggregatingMetrics = TestAggMetrics  # pylint:disable=invalid-name


class EvaluatorTest(parameterized.TestCase):

  def test_collected_and_aggregated_after_evaluate(self):
    dummy_fn = lambda x: jnp.ones((1, 2, 3))  # aggregated
    dummy_iterator = itertools.repeat(np.ones((1, 2)))  # collected
    model_keys = ["model0", "model1"]
    num_batches = 10
    evaluator = DummyEvaluator(
        models={m: dummy_fn for m in model_keys},
        benchmark=DummyBenchmark(),
        rng=jax.random.PRNGKey(10),
    )
    batch_collected = evaluator.evaluate(
        iterator=dummy_iterator, num_batches=num_batches
    ).compute()
    aggregated = evaluator.state.compute_aggregated_metrics()
    for m in model_keys:
      np.testing.assert_allclose(
          batch_collected[m]["collect0"], np.ones((num_batches, 2))
      )
      np.testing.assert_allclose(aggregated[m]["tensor_mean"], np.ones((1, 2)))
      self.assertEqual(aggregated[m]["mean"], 1.0)


class EvaluateRunTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.model_keys = ["model0", "model1"]
    self.agg_shape = (1, 2, 3)
    self.collect_shape = (1, 2)
    self.evaluator = DummyEvaluator(
        models={m: lambda x: jnp.ones(self.agg_shape) for m in self.model_keys},
        benchmark=DummyBenchmark(),
        rng=jax.random.PRNGKey(10),
    )

  @parameterized.named_parameters(("runs_out", 10), ("doesnt_run_out", 200))
  def test_aggregated_metrics(self, iterator_batches):
    work_dir = self.create_tempdir().full_path
    iterator = itertools.repeat(np.ones(self.collect_shape), iterator_batches)
    evaluate.run(
        evaluator=self.evaluator,
        dataloader=iterator,
        workdir=work_dir,
        num_aggregation_batches=10,
        max_eval_batches=100,
        enable_checkpoints=False,
    )
    final_metrics_path = (
        epath.Path(work_dir) / "results/final_aggregated_metrics.hdf5"
    )
    final_metrics = hdf5_utils.read_all_arrays_as_dict(final_metrics_path)
    for m in self.model_keys:
      np.testing.assert_allclose(
          final_metrics[m]["tensor_mean"], np.ones(self.agg_shape[:-1])
      )
      self.assertEqual(final_metrics[m]["mean"], 1.0)

  @parameterized.named_parameters(("runs_out", 10), ("doesnt_run_out", 200))
  def test_aggregated_metrics_to_zarr(self, iterator_batches):
    work_dir = self.create_tempdir().full_path
    iterator = itertools.repeat(np.ones(self.collect_shape), iterator_batches)
    evaluate.run(
        evaluator=self.evaluator,
        dataloader=iterator,
        workdir=work_dir,
        num_aggregation_batches=10,
        max_eval_batches=100,
        enable_checkpoints=False,
        results_format="zarr",
    )
    final_metrics_path = (
        epath.Path(work_dir) / "results/model0_aggregated_metrics.zarr"
    )
    self.assertTrue(os.path.exists(final_metrics_path))

    final_metrics = xr.open_zarr(final_metrics_path)
    np.testing.assert_allclose(
        final_metrics["tensor_mean"], np.ones(self.agg_shape[:-1])
    )
    self.assertEqual(final_metrics["mean"], 1.0)

  @parameterized.parameters((1,), (2,), (3,))
  def test_dump_collected_results(self, dump_every_n_groups):
    work_dir = self.create_tempdir().full_path
    max_batches = 100
    num_agg_batches = 10
    dump_period = dump_every_n_groups * num_agg_batches
    evaluate.run(
        evaluator=self.evaluator,
        dataloader=itertools.repeat(np.ones(self.collect_shape)),
        workdir=work_dir,
        dump_collected_every_n_groups=dump_every_n_groups,
        num_aggregation_batches=num_agg_batches,
        max_eval_batches=max_batches,
        enable_checkpoints=False,
    )
    self.assertLen(
        os.listdir(epath.Path(work_dir) / "results"),
        # dumped batches plus final aggregated
        np.ceil(max_batches // dump_period) + 1,
    )
    first_batch = hdf5_utils.read_all_arrays_as_dict(
        epath.Path(work_dir) / f"results/batch_{dump_period}.hdf5"
    )
    for m in self.model_keys:
      np.testing.assert_allclose(
          first_batch[m]["collect0"],
          np.ones((num_agg_batches,) + self.collect_shape[1:]),
      )

  @parameterized.parameters((1,), (2,))
  def test_dump_collected_results_to_zarr(self, dump_every_n_groups):
    work_dir = self.create_tempdir().full_path
    max_batches = 100
    num_agg_batches = 10
    evaluate.run(
        evaluator=self.evaluator,
        dataloader=itertools.repeat(np.ones(self.collect_shape)),
        workdir=work_dir,
        dump_collected_every_n_groups=dump_every_n_groups,
        num_aggregation_batches=num_agg_batches,
        max_eval_batches=max_batches,
        enable_checkpoints=False,
        results_format="zarr",
    )
    # collected plus final aggregated per model
    self.assertLen(os.listdir(epath.Path(work_dir) / "results"), 4)

    collected_ds = xr.open_zarr(
        epath.Path(work_dir) / "results/model0_collected_metrics.zarr"
    )
    # Collected zarr contains all dumped batches
    np.testing.assert_allclose(
        collected_ds["collect0"],
        np.ones((max_batches // dump_every_n_groups,) + self.collect_shape[1:]),
    )

  @parameterized.parameters((10,), (20,))
  def test_save_checkpoint(self, period):
    work_dir = self.create_tempdir().full_path
    loader = pygrain.load(pygrain.RangeDataSource(start=0, stop=10, step=1))
    max_batches = 100
    num_agg_batches = 10
    evaluate.run(
        evaluator=self.evaluator,
        dataloader=loader,
        workdir=work_dir,
        num_aggregation_batches=num_agg_batches,
        max_eval_batches=max_batches,
        checkpoint_options=checkpoint.CheckpointManagerOptions(
            save_interval_steps=period
        ),
    )
    self.assertLen(
        os.listdir(epath.Path(work_dir) / "checkpoints"),
        max_batches // period + int(num_agg_batches != period),
    )

  def test_restore_progress_from_checkpoint(self):
    work_dir = self.create_tempdir().full_path
    loader = pygrain.load(pygrain.RangeDataSource(start=0, stop=30, step=1))
    options = checkpoint.CheckpointManagerOptions(save_interval_steps=10)
    # first run a few steps without batch dumps
    steps_without_dumping = 50
    evaluate.run(
        evaluator=self.evaluator,
        dataloader=loader,
        workdir=work_dir,
        num_aggregation_batches=10,
        max_eval_batches=steps_without_dumping,
        checkpoint_options=options,
    )
    ckpt_manager = checkpoint.CheckpointManager(
        directory=epath.Path(work_dir) / "checkpoints",
        checkpointers=dict(
            data=evaluate.PYGRAIN_CHECKPOINTER,
            eval_state=checkpoint.PyTreeCheckpointer(),
        ),
        options=options,
    )
    self.assertEqual(ckpt_manager.latest_step(), steps_without_dumping)
    # then run a few steps with batch dumps
    steps_with_dumping = 60
    evaluate.run(
        evaluator=self.evaluator,
        dataloader=loader,
        workdir=work_dir,
        num_aggregation_batches=10,
        dump_collected_every_n_groups=1,
        max_eval_batches=steps_with_dumping,
        checkpoint_options=options,
    )
    # check that there are only dumps for the second run
    res_files = os.listdir(epath.Path(work_dir) / "results")
    self.assertNotIn(f"batch_{steps_without_dumping}.hdf5", res_files)
    self.assertIn(f"batch_{steps_with_dumping}.hdf5", res_files)
    # check that dumped values reflect iterator state was restored from ckpt
    collected = hdf5_utils.read_all_arrays_as_dict(
        epath.Path(work_dir) / f"results/batch_{steps_with_dumping}.hdf5"
    )
    np.testing.assert_allclose(
        collected["model0"]["collect0"], np.arange(20, 30)
    )


if __name__ == "__main__":
  absltest.main()
