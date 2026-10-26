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

Two optional stabilisers, both off by default, can be enabled through the
trainer constructor: a replay pool of generated samples for the discriminators
(Shrivastava et al., 2017; `pool_size`) and an exponential moving average of
the generator parameters for inference (`ema_decay`).
"""

from collections.abc import Callable
import functools
from typing import Any, TypeAlias, cast

from clu import metrics as clu_metrics
import flax
import jax
import jax.numpy as jnp
import optax
from swirl_dynamics.projects.debiasing.cycle_gan import loss_functions
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
    "loss_gp_a",
    "loss_gp_b",
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


class ImagePoolState(flax.struct.PyTreeNode):
  """Replay buffer of generated samples shown to a discriminator.

  Attributes:
    buffer: Stored samples, of shape `(pool_size, *sample_shape)`.
    count: Number of filled slots, an int32 scalar in `[0, pool_size]`.
  """

  buffer: Array
  count: Array


def create_image_pool(
    pool_size: int,
    sample_shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike = jnp.float32,
) -> ImagePoolState:
  """Creates an empty pool of `pool_size` samples of shape `sample_shape`.

  Args:
    pool_size: Number of samples the pool can hold.
    sample_shape: Shape of one sample (without the batch axis).
    dtype: Data type of the stored samples.

  Returns:
    An empty pool (`count == 0`).
  """
  return ImagePoolState(
      buffer=jnp.zeros((pool_size, *sample_shape), dtype),
      count=jnp.zeros((), jnp.int32),
  )


def query_image_pool(
    state: ImagePoolState, rng: Array, images: Array, p_swap: float = 0.5
) -> tuple[ImagePoolState, Array]:
  """Returns the samples to show to the discriminator and the updated pool.

  The batch is processed sequentially. While the pool is not full, each
  sample is stored and returned unchanged. Once it is full, each sample is,
  with probability `p_swap`, exchanged against a uniformly chosen stored
  sample (which is returned, the new sample taking its slot) and otherwise
  returned unchanged without touching the pool.

  Args:
    state: The pool.
    rng: Random key.
    images: Batch of generated samples, of shape `(batch, *sample_shape)`.
    p_swap: Probability of returning a stored sample once the pool is full.

  Returns:
    The updated pool and a batch of the same shape as `images`.

  Raises:
    ValueError: If `p_swap` is not in `[0.0, 1.0]`.
  """
  if not 0.0 <= p_swap <= 1.0:
    raise ValueError(f"p_swap must be in [0.0, 1.0], got {p_swap}.")
  pool_size = state.buffer.shape[0]

  def _process_sample(carry, inputs):
    buffer, count = carry
    image, rng_sample = inputs
    rng_swap, rng_slot = jax.random.split(rng_sample)
    is_full = count >= pool_size
    swap = jnp.logical_and(is_full, jax.random.uniform(rng_swap) < p_swap)
    slot = jnp.where(
        is_full, jax.random.randint(rng_slot, (), 0, pool_size), count
    )
    stored = buffer[slot]
    writes = jnp.logical_or(jnp.logical_not(is_full), swap)
    buffer = jnp.where(writes, buffer.at[slot].set(image), buffer)
    count = jnp.minimum(count + 1, pool_size)
    return (buffer, count), jnp.where(swap, stored, image)

  rngs = jax.random.split(rng, images.shape[0])
  (buffer, count), outputs = jax.lax.scan(
      _process_sample, (state.buffer, state.count), (images, rngs)
  )
  return ImagePoolState(buffer=buffer, count=count), outputs


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
    pool_a: Replay pool of generated samples of A, or `None` when disabled.
    pool_b: Replay pool of generated samples of B, or `None` when disabled.
    ema_gen: Exponential moving average of `(params_gen_a2b, params_gen_b2a)`
      (accessed through `ema_gen.ema`), or `None` when disabled.
  """

  params_gen_a2b: VariableDict
  params_gen_b2a: VariableDict
  params_dis_a: VariableDict
  params_dis_b: VariableDict
  opt_state_gen: optax.OptState
  opt_state_dis: optax.OptState
  flax_mutables: VariableDict = train_states.EMPTY_DICT
  pool_a: ImagePoolState | None = None
  pool_b: ImagePoolState | None = None
  ema_gen: optax.EmaState | None = None

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

  @property
  def eval_variables(self) -> tuple[VariableDict, ...]:
    """Variables for evaluation, using EMA generators when enabled."""
    params: ParamsTuple = self.params
    if self.ema_gen is not None:
      ema_a2b, ema_b2a = cast(
          tuple[VariableDict, VariableDict], self.ema_gen.ema
      )
      params = (ema_a2b, ema_b2a, self.params_dis_a, self.params_dis_b)
    return tuple(
        flax.core.freeze(dict(params=p, **self.flax_mutables)) for p in params
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
      pool_size: int = 0,
      ema_decay: float | None = None,
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
      pool_size: Size of the replay pools of generated samples shown to the
        discriminators; `0` disables them. The pools are also disabled for the
        Wasserstein critic (`GanMode.WGANGP`), whose gradient penalty needs
        on-policy samples.
      ema_decay: Decay of an exponential moving average of the generator
        parameters (for inference, see `inference_fn_from_state_dict`); `None`
        disables it.
      num_steps_to_compile: See `trainers.BaseTrainer`.

    Raises:
      ValueError: If `n_critic` is not positive, `pool_size` is negative, or
        `ema_decay` is outside `[0.0, 1.0]`.
    """
    if n_critic < 1:
      raise ValueError(f"n_critic must be positive, got {n_critic}.")
    if pool_size < 0:
      raise ValueError(f"pool_size must be non-negative, got {pool_size}.")
    if ema_decay is not None and not 0.0 <= ema_decay <= 1.0:
      raise ValueError(f"ema_decay must be in [0.0, 1.0], got {ema_decay}.")
    self.optimizer_gen = optimizer_gen
    self.optimizer_dis = optimizer_dis
    self.n_critic = n_critic
    is_wgan = model.cycle_gan.config.gan_mode == loss_functions.GanMode.WGANGP
    self.pool_size = 0 if is_wgan else pool_size
    self.ema = None if ema_decay is None else optax.ema(ema_decay)
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
    pool_a = pool_b = None
    if self.pool_size > 0:
      shape_a, shape_b = self.model.dim_inputs
      pool_a = create_image_pool(self.pool_size, tuple(shape_a[1:]))
      pool_b = create_image_pool(self.pool_size, tuple(shape_b[1:]))
    ema_gen = None
    if self.ema is not None:
      ema_gen = self.ema.init((params[0], params[1]))
    return CycleGANTrainState.create(
        replicate=self.is_distributed,
        params_gen_a2b=params[0],
        params_gen_b2a=params[1],
        params_dis_a=params[2],
        params_dis_b=params[3],
        opt_state_gen=self.optimizer_gen.init((params[0], params[1])),
        opt_state_dis=self.optimizer_dis.init((params[2], params[3])),
        pool_a=pool_a,
        pool_b=pool_b,
        ema_gen=ema_gen,
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
      rng_gen, rng_dis, rng_pool_a, rng_pool_b = jax.random.split(rng, 4)
      params = state.params

      # Phase 1: generators, differentiated with respect to their parameters
      # only (the model additionally stop-gradients the discriminators).
      def gen_loss(params_gen):
        return self.model.generator_loss_and_samples(
            params_gen + params[2:], batch, rng_gen, state.flax_mutables
        )

      params_gen = (params[0], params[1])
      (_, (gen_metrics, mutables, fakes)), grads_gen = jax.value_and_grad(
          gen_loss, has_aux=True
      )(params_gen)
      grads_gen = self._reduce_grads(grads_gen)
      updates, opt_state_gen = self.optimizer_gen.update(
          grads_gen, state.opt_state_gen, params_gen
      )
      params_gen = cast(
          tuple[VariableDict, VariableDict],
          optax.apply_updates(params_gen, updates),
      )
      ema_gen = state.ema_gen
      if self.ema is not None and ema_gen is not None:
        _, ema_gen = self.ema.update(params_gen, ema_gen)

      # Phase 2: discriminators on the real batch and the generated samples,
      # which are kept fixed (and stop-gradiented) across the critic steps.
      fake_b, _, fake_a, _ = fakes
      fake_a = jax.lax.stop_gradient(fake_a)
      fake_b = jax.lax.stop_gradient(fake_b)
      pool_a, pool_b = state.pool_a, state.pool_b
      if pool_a is not None and pool_b is not None:
        pool_a, fake_a = query_image_pool(pool_a, rng_pool_a, fake_a)
        pool_b, fake_b = query_image_pool(pool_b, rng_pool_b, fake_b)
      dis_batch = dict(batch, fake_data_a=fake_a, fake_data_b=fake_b)

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

      # The loop carry must match the metrics' shapes and dtypes exactly.
      initial_metrics = jax.tree_util.tree_map(
          lambda leaf: jnp.zeros(leaf.shape, leaf.dtype),
          jax.eval_shape(dis_loss, (params[2], params[3]), rng_dis)[1],
      )
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
          pool_a=pool_a,
          pool_b=pool_b,
          ema_gen=ema_gen,
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
      variables = state.eval_variables
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
      use_ema: bool = False,
  ) -> Callable[[Array], Array]:
    """Builds a deterministic generator map from a (restored) train state.

    Args:
      state: A `CycleGANTrainState`, or the object returned by
        `CycleGANTrainState.restore_from_orbax_ckpt(ref_state=None)`, whose
        fields are plain dictionaries and lists.
      model: The model whose generators are applied.
      direction: `"a2b"` for `generator_a2b`, `"b2a"` for `generator_b2a`.
      use_ema: Whether to use the EMA generator parameters instead of the
        trained ones.

    Returns:
      A function mapping a batch of samples to their translations.

    Raises:
      ValueError: For an unknown `direction`, or if `use_ema` is set but the
        state has no EMA parameters.
    """
    if direction == "a2b":
      index, generator = 0, model.cycle_gan.generator_a2b
    elif direction == "b2a":
      index, generator = 1, model.cycle_gan.generator_b2a
    else:
      raise ValueError(f"direction must be 'a2b' or 'b2a', got {direction!r}.")
    if use_ema:
      # A disabled EMA is `None` in a live state and an empty dictionary in
      # a raw orbax restore.
      if not state.ema_gen:
        raise ValueError("The train state has no EMA generator parameters.")
      ema_state = state.ema_gen
      # Raw orbax restores hold the EMA state as a dictionary.
      ema = ema_state["ema"] if isinstance(ema_state, dict) else ema_state.ema
      params = cast(tuple[VariableDict, VariableDict], ema)[index]
    else:
      params = state.params[index]
    return model.inference_fn({"params": params}, generator)


class DistributedCycleGANTrainer(CycleGANTrainer):
  """Multi-device (pmap) variant of `CycleGANTrainer`.

  When `pool_size > 0`, each device replica maintains its own independent replay
  pool (`pool_a`, `pool_b` are initialized identically and diverge as per-device
  `rng` splits arrive), and `unreplicated_train_state` checkpoints replica 0's
  pool.
  """

  is_distributed: bool = True

  def _reduce_grads(self, grads: PyTree) -> PyTree:
    """See base class."""
    return jax.lax.pmean(grads, axis_name=trainers.PMAP_AXIS_NAME)

  def _metrics_from_output(self, collection, **outputs) -> Metrics:
    """See base class."""
    return collection.gather_from_model_output(
        axis_name=trainers.PMAP_AXIS_NAME, **outputs
    )
