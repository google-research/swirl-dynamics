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

"""Tests for the unrolled (`num_steps_to_compile > 1`) training loop.

In Blaze (`templates/BUILD`), this test target runs with
`XLA_FLAGS=--xla_force_host_platform_device_count=2` so the distributed `pmap`
+ `gather_from_model_output` path is exercised on two CPU devices. Under
multi-file `pytest` runners (such as GitHub Actions CI), it avoids calling
`chex.set_n_cpu_devices` at import time so it does not conflict with other test
modules sharing the same worker process.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
from clu import metrics as clu_metrics
import jax
import jax.numpy as jnp
import numpy as np
import optax
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers


def dummy_iter(batch_sz: int, value: float = 0.5):
  while True:
    yield {"1": np.full(batch_sz, value, dtype=np.float32)}


def step_iter(batch_sz: int):
  """Yields batches whose mean on step `k` (1-indexed) equals `k`."""
  step = 1
  while True:
    # Offsets around zero (`[-1.5, -0.5, 0.5, 1.5]` for `batch_sz=4`) give
    # distinct per-device means while keeping the global batch mean equal to
    # `step`, exercising both last-step selection (`batch(-1)`) and per-device
    # gathering (`gather_from_model_output`).
    offsets = np.arange(batch_sz, dtype=np.float32) - 0.5 * (batch_sz - 1)
    yield {"1": float(step) + offsets}
    step += 1


def train_loss_metric(metrics) -> clu_metrics.Average:
  """Returns the `train_loss` metric of a collection created at runtime."""
  return vars(metrics)["train_loss"]


class DummyModel(models.BaseModel):
  """Minimal model with unit parameter gradients and batch-dependent loss."""

  def __init__(self, params_shape: tuple[int, ...] = (5, 5)):
    self.params_shape = params_shape

  def initialize(self, rng: jax.Array) -> models.PyTree:
    del rng
    return {"params": jnp.zeros(self.params_shape)}

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: jax.Array,
      mutables: models.PyTree,
      **kwargs,
  ) -> models.LossAndAux:
    del rng, kwargs
    batch_loss = jnp.mean(batch["1"])
    return jnp.sum(params) + batch_loss, ({"loss": batch_loss}, mutables)


def get_test_trainer(trainer_cls, **kwargs):
  trainer = trainer_cls(
      model=DummyModel(),
      rng=jax.random.PRNGKey(0),
      optimizer=optax.sgd(learning_rate=1),
      **kwargs,
  )
  trainer.TrainMetrics = clu_metrics.Collection.create(
      train_loss=clu_metrics.Average.from_output("loss")
  )
  return trainer


class UnrolledTrainingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if "--xla_force_host_platform_device_count=2" in os.environ.get(
        "XLA_FLAGS", ""
    ):
      self.assertEqual(jax.local_device_count(), 2)

  @parameterized.named_parameters(
      ("single_device", trainers.BasicTrainer, False),
      ("distributed", trainers.BasicDistributedTrainer, True),
  )
  def test_unrolled_metrics_merged_once_per_block(
      self, trainer_cls, is_distributed
  ):
    """Unrolled training reports the last step's gathered update per block.

    The unrolled loop keeps the metrics of the last step (`batch(-1)`) of every
    compiled block. Those updates must be unreplicated and merged on the host
    exactly like in `_train_without_unrolling`; in particular, metrics that were
    already gathered across devices inside `train_step` must not be reduced a
    second time, which would inflate the counts by the number of devices.
    """
    num_devices = jax.local_device_count() if is_distributed else 1
    num_steps_to_compile = 2
    num_steps = 6
    trainer = get_test_trainer(
        trainer_cls, num_steps_to_compile=num_steps_to_compile
    )
    params_shape = (5, 5)
    train_metrics = trainer.train(
        batch_iter=step_iter(batch_sz=4), num_steps=num_steps
    )

    # The train state advanced by `num_steps` and parameters were updated once
    # per step (sgd with lr=1 and unit gradients: -1 per step).
    state = trainer._maybe_unreplicate(trainer.train_state)
    self.assertEqual(int(state.step), num_steps)
    np.testing.assert_allclose(
        state.params, -num_steps * jnp.ones(params_shape)
    )

    # One metric update per compiled block, each gathered over all devices.
    # For 3 blocks of 2 steps (`(1, 2)`, `(3, 4)`, `(5, 6)`), the last-step
    # losses are `[2.0, 4.0, 6.0]` with mean `4.0` (whereas accidentally
    # returning step 0 of each block would give `mean([1.0, 3.0, 5.0]) = 3.0`).
    num_blocks = num_steps // num_steps_to_compile
    expected_last_step_mean = np.mean([2.0, 4.0, 6.0])
    self.assertEqual(
        int(train_loss_metric(train_metrics).count), num_blocks * num_devices
    )
    np.testing.assert_allclose(
        float(train_metrics.compute()["train_loss"]),
        expected_last_step_mean,
    )

  def test_unrolled_matches_non_unrolled_metrics(self):
    """With a constant loss both loops agree on the merged training metrics."""
    num_steps = 4
    loss = 0.5
    results = {}
    for num_steps_to_compile in (1, 2):
      trainer = get_test_trainer(
          trainers.BasicDistributedTrainer,
          num_steps_to_compile=num_steps_to_compile,
      )
      metrics = trainer.train(
          batch_iter=dummy_iter(batch_sz=4, value=loss), num_steps=num_steps
      )
      results[num_steps_to_compile] = metrics

    # Both paths produce host-side, unreplicated (scalar) metrics.
    for metrics in results.values():
      self.assertEqual(jnp.shape(train_loss_metric(metrics).total), ())
      self.assertEqual(jnp.shape(train_loss_metric(metrics).count), ())
    np.testing.assert_allclose(
        float(results[1].compute()["train_loss"]),
        float(results[2].compute()["train_loss"]),
    )
    # The unrolled loop merges one update per block instead of one per step.
    self.assertEqual(
        int(train_loss_metric(results[2]).count),
        int(train_loss_metric(results[1]).count) // 2,
    )

  def test_unrolled_unreplicates_multi_replica_metrics_without_double_reduce(
      self,
  ):
    """Verifies `_train_unrolled` unreplicates (`[0]`) instead of `.reduce()`."""
    num_steps_to_compile = 2
    num_steps = 6
    num_blocks = num_steps // num_steps_to_compile
    simulated_replicas = 4
    loss = 0.25
    trainer = get_test_trainer(
        trainers.BasicDistributedTrainer,
        num_steps_to_compile=num_steps_to_compile,
    )
    # Simulate the output of `gather_from_model_output` inside a 4-replica pmap:
    # each replica already holds the gathered total and count across all 4
    # replicas (`count = 4`, `total = 4 * loss`).
    gathered_single = trainer.TrainMetrics.single_from_model_output(
        loss=jnp.full((simulated_replicas,), loss)
    )
    replicated_update = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (simulated_replicas,) + x.shape),
        gathered_single,
    )
    step_counter = [0]

    def fake_compiled_inner_loop(train_rng, train_state, stacked_batches):
      del stacked_batches
      step_counter[0] += num_steps_to_compile
      new_state = train_states.BasicTrainState(
          step=jnp.full((simulated_replicas,), step_counter[0]),
          params=train_state.params,
          opt_state=train_state.opt_state,
          flax_mutables=train_state.flax_mutables,
      )
      return new_state, replicated_update, train_rng

    trainer._compiled_train_inner_loop = fake_compiled_inner_loop
    train_metrics = trainer.train(
        batch_iter=dummy_iter(batch_sz=4), num_steps=num_steps
    )
    # Unreplicating takes replica 0 (`count = num_blocks * simulated_replicas`),
    # whereas `.reduce()` would sum across the replica axis a second time
    # (`count = num_blocks * simulated_replicas**2`).
    self.assertEqual(
        int(train_loss_metric(train_metrics).count),
        num_blocks * simulated_replicas,
    )
    np.testing.assert_allclose(
        float(train_metrics.compute()["train_loss"]), loss
    )


if __name__ == "__main__":
  absltest.main()
