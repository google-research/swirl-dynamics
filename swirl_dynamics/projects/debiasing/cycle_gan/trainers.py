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

"""Trainers for the CycleGAN debiasing models.

Each training step alternates a generator phase, which differentiates
`CycleGANModel.generator_loss_and_samples` with the discriminators frozen, and
a discriminator phase, which runs `n_critic` gradient steps of
`CycleGANModel.discriminator_loss_and_metrics` on the real batch and the
(stop-gradiented) samples produced in the generator phase, as in the reference
implementation. The step counter advances once per alternating step.
"""

from collections.abc import Callable
import functools
from typing import Any, TypeAlias, cast

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
import optax
from swirl_dynamics.projects.debiasing.cycle_gan import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array: TypeAlias = jax.Array
BatchType: TypeAlias = trainers.BatchType
Metrics: TypeAlias = clu_metrics.Collection
PyTree: TypeAlias = Any
VariableDict: TypeAlias = trainers.VariableDict
ParamsTuple: TypeAlias = tuple[
    VariableDict, VariableDict, VariableDict, VariableDict
]

_GENERATOR_METRICS = (
    "loss",
    "loss_gen_a",
    "loss_gen_b",
    "loss_cycle_a",
    "loss_cycle_b",
    "loss_id_a",
    "loss_id_b",
)
_DISCRIMINATOR_METRICS = (
    "loss_dis",
    "loss_dis_a",
    "loss_dis_b",
    "loss_dis_crossed",
)
_SAMPLE_KEYS = ("real_data_a", "fake_data_b", "real_data_b", "fake_data_a")


# Metric collections are created from the model output keys; the trainer
# exposes them through the `TrainMetrics` / `EvalMetrics` properties required
# by the `BaseTrainer` interface.
_TRAIN_METRICS = clu_metrics.Collection.create(
    **{
        key: clu_metrics.Average.from_output(key)
        for key in _GENERATOR_METRICS + _DISCRIMINATOR_METRICS
    },
    loss_std=clu_metrics.Std.from_output("loss"),
)
_EVAL_METRICS = clu_metrics.Collection.create(
    eval_loss=clu_metrics.Average.from_output("loss"),
    eval_loss_std=clu_metrics.Std.from_output("loss"),
    eval_loss_dis_a=clu_metrics.Average.from_output("loss_dis_a"),
    eval_loss_dis_b=clu_metrics.Average.from_output("loss_dis_b"),
    samples=clu_metrics.CollectingMetric.from_outputs(_SAMPLE_KEYS),
)


def _as_arrays(batch: BatchType) -> dict[str, Array]:
  """Converts the batch entries to JAX arrays."""
  return {key: jnp.asarray(value) for key, value in batch.items()}


class CycleGANTrainState(train_states.TrainState):
  """Train state with one parameter group per network and two optimizers.

  Attributes:
    params_gen_a2b: Parameters of the A-to-B generator.
    params_gen_b2a: Parameters of the B-to-A generator.
    params_dis_a: Parameters of the discriminator of A.
    params_dis_b: Parameters of the discriminator of B.
    opt_state_gen: Optimizer state over `(params_gen_a2b, params_gen_b2a)`.
    opt_state_dis: Optimizer state over `(params_dis_a, params_dis_b)`.
    flax_mutables: Mutable variable collections. The CycleGAN networks are
      stateless, so this is empty and passed through the model unchanged.
  """

  params_gen_a2b: VariableDict
  params_gen_b2a: VariableDict
  params_dis_a: VariableDict
  params_dis_b: VariableDict
  opt_state_gen: optax.OptState
  opt_state_dis: optax.OptState
  flax_mutables: VariableDict = train_states.EMPTY_DICT

  @property
  def params(self) -> ParamsTuple:
    """Parameters in the order expected by `CycleGANModel`."""
    return (
        self.params_gen_a2b,
        self.params_gen_b2a,
        self.params_dis_a,
        self.params_dis_b,
    )

  @property
  def model_variables(self) -> tuple[VariableDict, ...]:
    """Variables of the four networks in the format of `eval_fn`."""
    return tuple(
        flax.core.freeze(dict(params=params, **self.flax_mutables))
        for params in self.params
    )


class CycleGANTrainer(
    trainers.BaseTrainer[models.CycleGANModel, CycleGANTrainState]
):
  """Alternating generator / discriminator optimisation.

  Two optimizers are used, one over both generators and one over both
  discriminators, as in the reference implementation. This is identical to
  four per-network optimizers for element-wise transforms with shared
  hyper-parameters (e.g. Adam) but not for cross-parameter transforms such as
  `optax.clip_by_global_norm`, which then act on both generators (or both
  discriminators) jointly; use `optax.multi_transform` for per-network
  behaviour in that case.
  """

  @functools.cached_property
  def TrainMetrics(self):  # pyrefly: ignore[bad-override]
    """Averages of the generator and discriminator losses, std of `loss`."""
    return _TRAIN_METRICS

  @functools.cached_property
  def EvalMetrics(self):  # pyrefly: ignore[bad-override]
    """Evaluation losses and the translated samples of each eval batch."""
    return _EVAL_METRICS

  def __init__(
      self,
      model: models.CycleGANModel,
      rng: Array,
      optimizer_gen: optax.GradientTransformation,
      optimizer_dis: optax.GradientTransformation,
      n_critic: int = 1,
      num_steps_to_compile: int = 1,
  ):
    """Initializes the trainer.

    Args:
      model: The CycleGAN model.
      rng: Random key of the trainer.
      optimizer_gen: Optimizer of the pair `(params_gen_a2b, params_gen_b2a)`.
      optimizer_dis: Optimizer of the pair `(params_dis_a, params_dis_b)`.
      n_critic: Number of discriminator updates per generator update, all on the
        same real batch and generated samples.
      num_steps_to_compile: See `trainers.BaseTrainer`.

    Raises:
      ValueError: If `n_critic` is not positive.
    """
    if n_critic < 1:
      raise ValueError(f"n_critic must be positive, got {n_critic}.")
    self.optimizer_gen = optimizer_gen
    self.optimizer_dis = optimizer_dis
    self.n_critic = n_critic
    super().__init__(
        model=model, rng=rng, num_steps_to_compile=num_steps_to_compile
    )

  def initialize_train_state(self, rng: Array) -> CycleGANTrainState:
    """Initializes the four networks, the two optimizers and the step.

    Args:
      rng: Random key.

    Returns:
      The initial train state.

    Raises:
      ValueError: If a network has mutable variable collections (not supported
        by this trainer).
    """
    init_vars = self.model.initialize(rng)
    params = []
    for variables in init_vars:
      mutables, network_params = flax.core.pop(variables, "params")
      if mutables:
        raise ValueError(
            "CycleGANTrainer does not support networks with mutable variable"
            f" collections, got {list(mutables)}."
        )
      params.append(network_params)
    return CycleGANTrainState.create(
        replicate=self.is_distributed,
        params_gen_a2b=params[0],
        params_gen_b2a=params[1],
        params_dis_a=params[2],
        params_dis_b=params[3],
        opt_state_gen=self.optimizer_gen.init((params[0], params[1])),
        opt_state_dis=self.optimizer_dis.init((params[2], params[3])),
    )

  def _reduce_grads(self, grads: PyTree) -> PyTree:
    """Hook that averages gradients across devices when distributed."""
    return grads

  def _metrics_from_output(self, collection, **outputs) -> Metrics:
    """Hook that builds a metrics collection from per-step outputs."""
    return collection.single_from_model_output(**outputs)

  @property
  def train_step(
      self,
  ) -> Callable[
      [CycleGANTrainState, BatchType, Array],
      tuple[CycleGANTrainState, Metrics],
  ]:
    def _train_step(
        state: CycleGANTrainState, batch: BatchType, rng: Array
    ) -> tuple[CycleGANTrainState, Metrics]:
      """One generator update followed by `n_critic` discriminator updates."""
      batch = _as_arrays(batch)
      rng_gen, rng_dis = jax.random.split(rng)
      params = state.params

      # Phase 1: generators (the model stop-gradients the discriminators).
      grad_fn = jax.value_and_grad(
          self.model.generator_loss_and_samples, has_aux=True
      )
      (_, (gen_metrics, mutables, fakes)), grads = grad_fn(
          params, batch, rng_gen, state.flax_mutables
      )
      params_gen = (params[0], params[1])
      grads_gen = self._reduce_grads((grads[0], grads[1]))
      updates, opt_state_gen = self.optimizer_gen.update(
          grads_gen, state.opt_state_gen, params_gen
      )
      params_gen = cast(
          tuple[VariableDict, VariableDict],
          optax.apply_updates(params_gen, updates),
      )

      # Phase 2: discriminators on the real batch and the generated samples,
      # which are kept fixed (and stop-gradiented) across the critic steps.
      fake_b, _, fake_a, _ = fakes
      dis_batch = dict(
          batch,
          fake_data_a=jax.lax.stop_gradient(fake_a),
          fake_data_b=jax.lax.stop_gradient(fake_b),
      )

      def dis_loss(params_dis, rng_critic):
        full_params = (
            params_gen[0],
            params_gen[1],
            params_dis[0],
            params_dis[1],
        )
        return self.model.discriminator_loss_and_metrics(
            full_params, dis_batch, rng_critic, mutables
        )

      def critic_step(i, carry):
        params_dis, opt_state_dis, acc_metrics = carry
        (_, step_metrics), grads_dis = jax.value_and_grad(
            dis_loss, has_aux=True
        )(params_dis, jax.random.fold_in(rng_dis, i))
        grads_dis = self._reduce_grads(grads_dis)
        updates, opt_state_dis = self.optimizer_dis.update(
            grads_dis, opt_state_dis, params_dis
        )
        params_dis = cast(
            tuple[VariableDict, VariableDict],
            optax.apply_updates(params_dis, updates),
        )
        acc_metrics = jax.tree_util.tree_map(
            lambda acc, m: acc + m / self.n_critic, acc_metrics, step_metrics
        )
        return params_dis, opt_state_dis, acc_metrics

      initial_metrics = {key: jnp.zeros(()) for key in _DISCRIMINATOR_METRICS}
      params_dis, opt_state_dis, dis_metrics = jax.lax.fori_loop(
          0,
          self.n_critic,
          critic_step,
          ((params[2], params[3]), state.opt_state_dis, initial_metrics),
      )

      new_state = state.replace(
          step=state.step + 1,
          params_gen_a2b=params_gen[0],
          params_gen_b2a=params_gen[1],
          params_dis_a=params_dis[0],
          params_dis_b=params_dis[1],
          opt_state_gen=opt_state_gen,
          opt_state_dis=opt_state_dis,
          flax_mutables=mutables,
      )
      metrics_update = self._metrics_from_output(
          self.TrainMetrics, **gen_metrics, **dis_metrics
      )
      return new_state, metrics_update

    return _train_step

  @property
  def eval_step(
      self,
  ) -> Callable[[CycleGANTrainState, BatchType, Array], Metrics]:
    def _eval_step(
        state: CycleGANTrainState, batch: BatchType, rng: Array
    ) -> Metrics:
      """Scalar evaluation metrics plus translated samples for inspection."""
      batch = _as_arrays(batch)
      variables = state.model_variables
      eval_metrics, (fake_b, _, fake_a, _) = (
          self.model.eval_metrics_and_samples(
              variables=variables, batch=batch, rng=rng
          )
      )
      return self._metrics_from_output(
          self.EvalMetrics,
          **eval_metrics,
          real_data_a=batch["real_data_a"],
          fake_data_b=fake_b,
          real_data_b=batch["real_data_b"],
          fake_data_a=fake_a,
      )

    return _eval_step

  @staticmethod
  def inference_fn_from_state_dict(
      state: CycleGANTrainState,
      model: models.CycleGANModel,
      direction: str = "a2b",
  ) -> Callable[[Array], Array]:
    """Builds a deterministic generator map from a (restored) train state.

    Args:
      state: A `CycleGANTrainState`, or the object returned by
        `CycleGANTrainState.restore_from_orbax_ckpt(ref_state=None)`, whose
        parameter fields are plain dictionaries.
      model: The model whose generators are applied.
      direction: `"a2b"` for `generator_a2b`, `"b2a"` for `generator_b2a`.

    Returns:
      A function mapping a batch of samples to their translations.

    Raises:
      ValueError: For an unknown `direction`.
    """
    if direction == "a2b":
      params, generator = state.params_gen_a2b, model.cycle_gan.generator_a2b
    elif direction == "b2a":
      params, generator = state.params_gen_b2a, model.cycle_gan.generator_b2a
    else:
      raise ValueError(f"direction must be 'a2b' or 'b2a', got {direction!r}.")
    return model.inference_fn({"params": params}, generator)


class DistributedCycleGANTrainer(CycleGANTrainer):
  """Multi-device (pmap) variant of `CycleGANTrainer`."""

  is_distributed: bool = True

  def _reduce_grads(self, grads: PyTree) -> PyTree:
    """See base class."""
    return jax.lax.pmean(grads, axis_name=trainers.PMAP_AXIS_NAME)

  def _metrics_from_output(self, collection, **outputs) -> Metrics:
    """See base class."""
    return collection.gather_from_model_output(
        axis_name=trainers.PMAP_AXIS_NAME, **outputs
    )
