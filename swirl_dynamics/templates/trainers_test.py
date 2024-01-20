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

from absl.testing import absltest
from absl.testing import parameterized
from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

mock = absltest.mock


def dummy_iter(batch_sz):
  while True:
    yield {"1": np.ones(batch_sz)}


class TestTrainer(trainers.BaseTrainer):

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    train_loss: clu_metrics.Average.from_output("loss")

  @flax.struct.dataclass
  class EvalMetrics(clu_metrics.Collection):
    eval_accuracy: clu_metrics.Average.from_output("accuracy")

  @property
  def train_step(self):
    pass  # will be mocked

  @property
  def eval_step(self):
    pass  # will be mocked

  def initialize_train_state(self, rng):
    del rng
    return train_states.TrainState.create()


class BaseTrainerTest(absltest.TestCase):

  def test_train(self):
    """Test training loop with mocked step function."""
    num_steps = 5
    rng = np.random.default_rng(42)
    test_train_losses = rng.uniform(size=(num_steps,))

    with mock.patch.object(
        trainers.jax, "jit", autospec=True
    ) as mock_compiled_train_fn:
      # mocks jit-compiled train_step function to return state with
      # increasing step count and mocked metrics
      train_output = [
          (  # pylint: disable=g-complex-comprehension
              train_states.TrainState(step=jnp.array(i + 1)),
              TestTrainer.TrainMetrics.single_from_model_output(loss=l),
          )
          for i, l in enumerate(test_train_losses)
      ]
      mock_compiled_train_fn.return_value = mock.Mock(side_effect=train_output)
      mock_model = mock.Mock(spec=models.BaseModel)
      trainer = TestTrainer(model=mock_model, rng=jax.random.PRNGKey(0))
      train_metrics = trainer.train(
          batch_iter=iter(np.ones(num_steps)), num_steps=num_steps
      ).compute()

    self.assertEqual(trainer.train_state.step, num_steps)
    self.assertTrue(
        jnp.allclose(train_metrics["train_loss"], jnp.mean(test_train_losses))
    )

  def test_eval(self):
    """Test evaluation loop with mocked step function."""
    num_steps = 10
    rng = np.random.default_rng(43)
    test_eval_metrics = rng.uniform(size=(num_steps,))

    with mock.patch.object(
        trainers.jax, "jit", autospec=True
    ) as mock_compiled_eval_fn:
      # mocks jit-compiled eval_step function to return mocked metrics
      mock_compiled_eval_fn.return_value = mock.Mock(
          side_effect=[
              TestTrainer.EvalMetrics.single_from_model_output(accuracy=a)
              for a in test_eval_metrics
          ]
      )
      mock_model = mock.Mock(spec=models.BaseModel)
      trainer = TestTrainer(model=mock_model, rng=jax.random.PRNGKey(0))
      eval_metrics = trainer.eval(
          batch_iter=iter(np.ones(num_steps)), num_steps=num_steps
      ).compute()

    self.assertEqual(trainer.train_state.step, 0)
    self.assertTrue(
        jnp.allclose(eval_metrics["eval_accuracy"], jnp.mean(test_eval_metrics))
    )


def get_test_trainer(trainer_cls):
  params_shape = (5, 5)
  mock_model = mock.Mock(spec=models.BaseModel)
  mock_model.initialize.return_value = flax.core.freeze(
      {"params": jnp.zeros(params_shape)}
  )
  return trainer_cls(
      model=mock_model,
      rng=jax.random.PRNGKey(0),
      optimizer=optax.sgd(learning_rate=1),
  )


class BasicTrainerTest(parameterized.TestCase):

  def test_train_metrics_compute(self):
    """Test train metrics calculation."""
    trainer = get_test_trainer(trainers.BasicTrainer)
    trainer.TrainMetrics = clu_metrics.Collection.create(
        train_loss=clu_metrics.Average.from_output("loss")
    )
    params_shape = trainer.train_state.params.shape

    num_steps = 5
    rng = np.random.default_rng(84)
    mock_losses = rng.uniform(size=(num_steps,))
    # use non-jitted train_step function because although mocked function can be
    # jit-compiled, it cannot be used with `side_effect` to specify multiple
    # return values - the compiled function will always return the first
    trainer._compiled_train_step = trainer.train_step
    with mock.patch.object(trainers.jax, "grad", autospec=True) as mock_grad_fn:
      mock_grad_fn.return_value = mock.Mock(
          side_effect=[
              (jnp.ones(params_shape), ({"loss": l}, {})) for l in mock_losses
          ]
      )
      train_metrics = trainer.train(
          batch_iter=iter(np.ones(num_steps)), num_steps=num_steps
      ).compute()
    mock_grad_fn.assert_called_with(
        trainer.model.loss_fn, argnums=0, has_aux=True
    )
    self.assertTrue(
        jnp.allclose(train_metrics["train_loss"], jnp.mean(mock_losses))
    )

  @parameterized.product(
      trainer_cls=(trainers.BasicTrainer, trainers.BasicDistributedTrainer),
      clip_delta=(0.25, 1.0, 5.0),  # >, = and < grad norm respectively
  )
  def test_grad_clip(self, trainer_cls, clip_delta):
    """Test train step with gradient clipping."""
    trainer = get_test_trainer(trainer_cls)
    # replace optimizer with grad-clipped version and re-initialize
    trainer.optimizer = optax.chain(
        optax.clip(max_delta=clip_delta), optax.sgd(learning_rate=1)
    )
    trainer._train_state = trainer.initialize_train_state(jax.random.PRNGKey(0))
    params_shape = trainer.train_state.params.shape

    num_steps = 5
    # patch `jax.value_and_grad` to return a grad_fn with mocked output
    with mock.patch.object(trainers.jax, "grad", autospec=True) as mock_grad_fn:
      mock_grad_fn.return_value = mock.Mock(
          return_value=(-1 * jnp.ones(params_shape), ({"loss": 0.0}, {}))
      )
      trainer.train(
          batch_iter=dummy_iter(batch_sz=5), num_steps=num_steps
      ).compute()
    mock_grad_fn.assert_called_with(
        trainer.model.loss_fn, argnums=0, has_aux=True
    )
    # expected params should have all entries = num_steps since with sgd(lr=1)
    # and grad = jnp.ones, we are incrementing the params by 1 every step
    update_per_step = min(clip_delta, 1.0)
    self.assertTrue(
        jnp.allclose(
            trainer.train_state.params,
            update_per_step * num_steps * jnp.ones(params_shape),
        )
    )

  def test_eval_step(self):
    """Test eval metrics calculation."""
    trainer = get_test_trainer(trainers.BasicTrainer)
    trainer.EvalMetrics = clu_metrics.Collection.create(
        eval_accuracy=clu_metrics.Average.from_output("accuracy")
    )
    # use non-jitted eval step function to allow mocking changing output
    trainer._compiled_eval_step = trainer.eval_step

    num_steps = 15
    rng = np.random.default_rng(126)
    mock_accuracies = rng.uniform(size=(num_steps,))
    eval_fn_returns = [
        {"accuracy": a, "not_used": 8.0} for a in mock_accuracies
    ]
    trainer.model.eval_fn = mock.Mock(side_effect=eval_fn_returns)
    eval_metrics = trainer.eval(
        batch_iter=iter(np.ones(num_steps)), num_steps=num_steps
    ).compute()
    self.assertTrue(
        jnp.allclose(eval_metrics["eval_accuracy"], jnp.mean(mock_accuracies))
    )

  def test_distributed_train_step(self):
    trainer = get_test_trainer(trainers.BasicDistributedTrainer)
    trainer.TrainMetrics = clu_metrics.Collection.create(
        train_loss=clu_metrics.Average.from_output("loss")
    )
    num_steps = 5
    l = 0.01
    with mock.patch.object(trainers.jax, "grad", autospec=True) as mock_grad_fn:
      mock_grad_fn.return_value = mock.Mock(
          return_value=(
              jnp.ones(trainer.train_state.params.shape),
              ({"loss": l}, {}),
          )
      )
      train_metrics = trainer.train(
          batch_iter=dummy_iter(batch_sz=5), num_steps=num_steps
      ).compute()
    mock_grad_fn.assert_called_with(
        trainer.model.loss_fn, argnums=0, has_aux=True
    )
    self.assertTrue(jnp.allclose(train_metrics["train_loss"], l))

  def test_distributed_eval_step(self):
    trainer = get_test_trainer(trainers.BasicDistributedTrainer)
    trainer.EvalMetrics = clu_metrics.Collection.create(
        eval_accuracy=clu_metrics.Average.from_output("accuracy")
    )
    num_steps = 15
    acc = 0.01
    trainer.model.eval_fn = mock.Mock(
        return_value={"accuracy": acc, "not_used": 8.0}
    )
    eval_metrics = trainer.eval(
        batch_iter=dummy_iter(batch_sz=5), num_steps=num_steps
    ).compute()
    self.assertTrue(jnp.allclose(eval_metrics["eval_accuracy"], acc))


if __name__ == "__main__":
  absltest.main()
