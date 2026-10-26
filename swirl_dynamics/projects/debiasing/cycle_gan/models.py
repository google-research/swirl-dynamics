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

"""CycleGAN module.

References:
[1] Zhu, J.Y., Park, T., Isola, P. and Efros, A.A., "Unpaired image-to-image
    translation using cycle-consistent adversarial networks". In Proceedings of
    the IEEE international conference on computer vision (pp. 2223-2232) 2017.
[2] Reference implementation (loss conventions):
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

from collections.abc import Callable
import dataclasses
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.projects.debiasing.cycle_gan import loss_functions
from swirl_dynamics.templates import models as swirl_models


PyTree = Any
Array = jax.Array
Initializer = nn.initializers.Initializer


@dataclasses.dataclass(kw_only=True)
class CycleGANModelConfig:
  """Config used by cycleGAN model.

  Attributes:
    lambda_a: Weight of the cycle-consistency loss of samples from A, and (times
      `lambda_id`) of the identity loss of `generator_b2a`.
    lambda_b: Weight of the cycle-consistency loss of samples from B, and (times
      `lambda_id`) of the identity loss of `generator_a2b`.
    lambda_id: Relative weight of the identity losses as defined in [1]. The
      identity forward passes are skipped when it is not positive.
    gan_mode: Type of adversarial loss: least squares GAN ("lsgan"), the vanilla
      GAN with BCE loss ("bce_with_logits"), or the Wasserstein critic loss
      ("wgangp").
    use_cycle_loss: Option to use the cycle loss as defined in [1].
    same_dimension: Whether samples of A and B have the same shape, which is
      required by the identity losses.
    use_crossed_dis_loss: Additionally train each discriminator to classify real
      samples of the *other* domain as fake, so that the discriminators also
      tell A from B.

  The defaults `lambda_a = lambda_b = lambda_id = 1` differ from the paper
  defaults (`lambda_a = lambda_b = 10`, `lambda_id = 0.5`).
  """

  lambda_a: float = 1.0
  lambda_b: float = 1.0
  lambda_id: float = 1.0
  gan_mode: loss_functions.GanMode | str = loss_functions.GanMode.LSGAN
  use_cycle_loss: bool = True
  same_dimension: bool = True
  use_crossed_dis_loss: bool = False

  def __post_init__(self):
    self.gan_mode = loss_functions.GanMode(self.gan_mode)


def _apply(module: nn.Module, params: PyTree, *args, **kwargs) -> Array:
  """Applies a stateless parameter-only `module` with `{"params": params}`."""
  return jnp.asarray(module.apply({"params": params}, *args, **kwargs))


def _split_rngs(rngs: dict[str, Array], num: int) -> list[dict[str, Array]]:
  """Splits every key of `rngs` into `num` independent keys.

  Args:
    rngs: Random keys per collection, e.g. `{"dropout": key}`.
    num: Number of independent copies.

  Returns:
    `num` dictionaries with the same collections and independent keys, so that
    successive network applications do not share e.g. dropout masks.
  """
  split = {name: jax.random.split(key, num) for name, key in rngs.items()}
  return [{name: keys[i] for name, keys in split.items()} for i in range(num)]


@dataclasses.dataclass(frozen=True)
class CycleGAN:
  """Container of the four CycleGAN networks and their loss computations.

  Sub-networks must be stateless parameter-only Flax modules (no mutable state
  collections such as `batch_stats`), as forward passes wrap parameters directly
  as `{"params": params}`.

  Attributes:
    config: Loss weights and options, see `CycleGANModelConfig`.
    generator_a2b: Generator for samples from A to B.
    generator_b2a: Generator for samples from B to A.
    discriminator_a: Discriminator of samples in A.
    discriminator_b: Discriminator of samples in B.
  """

  config: CycleGANModelConfig
  generator_a2b: nn.Module
  generator_b2a: nn.Module
  discriminator_a: nn.Module
  discriminator_b: nn.Module

  @property
  def dis_a(self) -> nn.Module:
    """Alias of `discriminator_a`."""
    return self.discriminator_a

  @property
  def dis_b(self) -> nn.Module:
    """Alias of `discriminator_b`."""
    return self.discriminator_b

  @property
  def criterion_gan(self) -> Callable[[Array, bool], Array]:
    """Adversarial criterion `loss(prediction, target_is_real)`."""
    return loss_functions.GanLoss(gan_mode=self.config.gan_mode)()

  @property
  def criterion_cycle(self) -> Callable[[Array, Array], Array]:
    """Cycle-consistency criterion."""
    return loss_functions.l1_loss

  @property
  def criterion_id(self) -> Callable[[Array, Array], Array]:
    """Identity criterion."""
    return loss_functions.l1_loss

  @property
  def lambda_a(self) -> float:
    """Weight of the cycle (and identity) loss of samples from A."""
    return self.config.lambda_a

  @property
  def lambda_b(self) -> float:
    """Weight of the cycle (and identity) loss of samples from B."""
    return self.config.lambda_b

  @property
  def lambda_id(self) -> float:
    """Relative weight of the identity losses."""
    return self.config.lambda_id

  @property
  def use_crossed_dis_loss(self) -> bool:
    """Whether the discriminators are also trained to tell A from B."""
    return self.config.use_crossed_dis_loss

  def init(
      self,
      rng: Array,
      dummy_input_a: jax.typing.ArrayLike,
      dummy_input_b: jax.typing.ArrayLike,
  ) -> tuple[PyTree, PyTree, PyTree, PyTree]:
    """Initializes the variables of the four networks.

    Args:
      rng: Random key.
      dummy_input_a: Batch of samples of A used to infer the shapes.
      dummy_input_b: Batch of samples of B used to infer the shapes.

    Returns:
      The variables of `(generator_a2b, generator_b2a, discriminator_a,
      discriminator_b)`.
    """
    rng_array = jax.random.split(rng, num=4)

    params_a2b = self.generator_a2b.init(
        rng_array[0], dummy_input_a, is_training=False
    )
    params_b2a = self.generator_b2a.init(
        rng_array[1], dummy_input_b, is_training=False
    )
    params_dis_a = self.discriminator_a.init(rng_array[2], dummy_input_a)
    params_dis_b = self.discriminator_b.init(rng_array[3], dummy_input_b)

    return params_a2b, params_b2a, params_dis_a, params_dis_b

  def run_generator_forward(
      self,
      rngs: dict[str, Array],
      params_gen: tuple[PyTree, PyTree],
      real_data: tuple[Array, Array],
      is_training: bool = True,
  ) -> tuple[Array, Array, Array, Array]:
    """Runs both generators forward, including the cycle reconstructions.

    Args:
      rngs: Random keys of the dropout layers, `{"dropout": ...}`.
      params_gen: Parameters of the generators, `(params_gen_a2b,
        params_gen_b2a)`.
      real_data: Batches of real samples, `(real_a, real_b)`.
      is_training: Whether dropout is active.

    Returns:
      The generated samples `(fake_b, recover_a, fake_a, recover_b)`.
    """
    # TODO: perhaps use dictionaries instead of positional tuples.
    params_gen_a2b, params_gen_b2a = params_gen
    real_a, real_b = real_data
    # Independent keys, so that the two applications of each generator do not
    # share dropout masks.
    rngs_fake_b, rngs_recover_a, rngs_fake_a, rngs_recover_b = _split_rngs(
        rngs, 4
    )

    # fake_b = gen_a2b(u_a).
    fake_b = _apply(
        self.generator_a2b,
        params_gen_a2b,
        real_a,
        is_training=is_training,
        rngs=rngs_fake_b,
    )
    # recover_a = gen_b2a(gen_a2b(u_a)).
    recover_a = _apply(
        self.generator_b2a,
        params_gen_b2a,
        fake_b,
        is_training=is_training,
        rngs=rngs_recover_a,
    )
    # fake_a = gen_b2a(u_b).
    fake_a = _apply(
        self.generator_b2a,
        params_gen_b2a,
        real_b,
        is_training=is_training,
        rngs=rngs_fake_a,
    )
    # recover_b = gen_a2b(gen_b2a(u_b)).
    recover_b = _apply(
        self.generator_a2b,
        params_gen_a2b,
        fake_a,
        is_training=is_training,
        rngs=rngs_recover_b,
    )

    return fake_b, recover_a, fake_a, recover_b

  def run_generator_forward_a2b(
      self,
      rngs: dict[str, Array],
      params_gen_a2b: PyTree,
      real_data_a: Array,
      is_training: bool = False,
  ) -> Array:
    """Translates samples from A to B.

    Args:
      rngs: Random keys of the dropout layers, `{"dropout": ...}`.
      params_gen_a2b: Parameters of the A-to-B generator.
      real_data_a: Batch of samples of A.
      is_training: Whether dropout is active.

    Returns:
      The generated samples `gen_a2b(u_a)`.
    """
    return _apply(
        self.generator_a2b,
        params_gen_a2b,
        real_data_a,
        is_training=is_training,
        rngs=rngs,
    )

  def run_generator_backward(
      self,
      rngs: dict[str, Array],
      params: tuple[PyTree, ...],
      generated_data: tuple[Array, Array, Array, Array],
      real_data: tuple[Array, Array],
      is_training: bool = True,
  ) -> dict[str, Array]:
    """Computes the generator losses.

    Args:
      rngs: Random keys of the dropout layers, `{"dropout": ...}`.
      params: `(params_g_a2b, params_g_b2a, params_d_a, params_d_b)`.
      generated_data: `(fake_b, recover_a, fake_a, recover_b)`.
      real_data: `(real_a, real_b)`.
      is_training: Whether dropout is active in the identity passes.

    Returns:
      A dictionary with the adversarial (`loss_gen_a`, `loss_gen_b`), cycle
      (`loss_cycle_a`, `loss_cycle_b`) and identity (`loss_id_a`, `loss_id_b`)
      losses; disabled terms are zero.
    """
    params_g_a2b, params_g_b2a, params_d_a, params_d_b = params
    fake_b, recover_a, fake_a, recover_b = generated_data
    real_a, real_b = real_data

    # Adversarial loss of the fakes in A: dis_a(gen_b2a(u_b)).
    disc_a_fake_a = _apply(self.discriminator_a, params_d_a, fake_a)
    loss_gen_a = self.criterion_gan(disc_a_fake_a, True)

    # Adversarial loss of the fakes in B: dis_b(gen_a2b(u_a)).
    disc_b_fake_b = _apply(self.discriminator_b, params_d_b, fake_b)
    loss_gen_b = self.criterion_gan(disc_b_fake_b, True)

    if self.config.use_cycle_loss:
      # Cycle loss ||gen_b2a(gen_a2b(u_a)) - u_a||.
      loss_cycle_a = self.criterion_cycle(recover_a, real_a) * self.lambda_a
      # Cycle loss ||gen_a2b(gen_b2a(u_b)) - u_b||.
      loss_cycle_b = self.criterion_cycle(recover_b, real_b) * self.lambda_b
    else:
      loss_cycle_a = loss_cycle_b = jnp.zeros(())

    # Identity losses need samples of A and B with the same shape, and cost two
    # generator passes, which are skipped when their weight vanishes.
    if self.config.same_dimension and self.lambda_id > 0.0:
      rngs_id_a, rngs_id_b = _split_rngs(rngs, 2)
      # gen_a2b should be the identity on samples of B: ||gen_a2b(u_b) - u_b||.
      id_a = _apply(
          self.generator_a2b,
          params_g_a2b,
          real_b,
          is_training=is_training,
          rngs=rngs_id_a,
      )
      # gen_b2a should be the identity on samples of A: ||gen_b2a(u_a) - u_a||.
      id_b = _apply(
          self.generator_b2a,
          params_g_b2a,
          real_a,
          is_training=is_training,
          rngs=rngs_id_b,
      )
      weight_id_a = self.lambda_b * self.lambda_id
      weight_id_b = self.lambda_a * self.lambda_id
      loss_id_a = weight_id_a * self.criterion_id(id_a, real_b)
      loss_id_b = weight_id_b * self.criterion_id(id_b, real_a)
    else:
      loss_id_a = loss_id_b = jnp.zeros(())

    return dict(
        loss_gen_a=loss_gen_a,
        loss_gen_b=loss_gen_b,
        loss_cycle_a=loss_cycle_a,
        loss_cycle_b=loss_cycle_b,
        loss_id_a=loss_id_a,
        loss_id_b=loss_id_b,
    )

  def run_discriminator_backward_a(
      self, params: PyTree, real_a: Array, fake_a: Array
  ) -> Array:
    """Loss of the discriminator of A on real and generated samples.

    Args:
      params: Parameters of `discriminator_a`.
      real_a: Batch of real samples of A.
      fake_a: Batch of generated samples of A (gradients are stopped).

    Returns:
      The average of the real and fake losses of the discriminator of A.
    """
    pred_real = _apply(self.discriminator_a, params, real_a)
    loss_dis_real = self.criterion_gan(pred_real, True)

    # The generator is not updated through the discriminator loss.
    pred_fake = _apply(
        self.discriminator_a, params, jax.lax.stop_gradient(fake_a)
    )
    loss_dis_fake = self.criterion_gan(pred_fake, False)

    return (loss_dis_real + loss_dis_fake) * 0.5

  def run_discriminator_backward_b(
      self, params: PyTree, real_b: Array, fake_b: Array
  ) -> Array:
    """Loss of the discriminator of B on real and generated samples.

    Args:
      params: Parameters of `discriminator_b`.
      real_b: Batch of real samples of B.
      fake_b: Batch of generated samples of B (gradients are stopped).

    Returns:
      The average of the real and fake losses of the discriminator of B.
    """
    pred_real = _apply(self.discriminator_b, params, real_b)
    loss_dis_real = self.criterion_gan(pred_real, True)

    # The generator is not updated through the discriminator loss.
    pred_fake = _apply(
        self.discriminator_b, params, jax.lax.stop_gradient(fake_b)
    )
    loss_dis_fake = self.criterion_gan(pred_fake, False)

    return (loss_dis_real + loss_dis_fake) * 0.5

  def run_discriminators_crossed(
      self, params: tuple[PyTree, PyTree], real_a: Array, real_b: Array
  ) -> Array:
    """Loss of the discriminators on real samples of the other domain.

    Args:
      params: `(params_dis_a, params_dis_b)`.
      real_a: Batch of real samples of A.
      real_b: Batch of real samples of B.

    Returns:
      The average of the losses classifying the other domain as fake.
    """
    params_dis_a, params_dis_b = params

    pred_a_on_b = _apply(self.discriminator_a, params_dis_a, real_b)
    loss_dis_a = self.criterion_gan(pred_a_on_b, False)

    pred_b_on_a = _apply(self.discriminator_b, params_dis_b, real_a)
    loss_dis_b = self.criterion_gan(pred_b_on_a, False)

    return (loss_dis_a + loss_dis_b) * 0.5


@dataclasses.dataclass(kw_only=True)
class CycleGANModel(swirl_models.BaseModel):
  """Model wrapping a `CycleGAN` for the swirl_dynamics trainers.

  Attributes:
    cycle_gan: Container of the four networks and the loss computations.
    dim_inputs: The shapes (including the batch axis) of samples from A and from
      B, used to initialise the networks.
  """

  cycle_gan: CycleGAN
  dim_inputs: tuple[tuple[int, ...], tuple[int, ...]]

  def initialize(self, rng: Array) -> tuple[PyTree, PyTree, PyTree, PyTree]:
    # Initialize the parameters.
    # Here we need the input dimension for both A and B, and the output should
    # be a dictionary with 4 fields.
    init_input = self.dim_inputs
    dummy_a = jnp.ones(init_input[0])
    dummy_b = jnp.ones(init_input[1])

    return self.cycle_gan.init(rng, dummy_a, dummy_b)

  def __post_init__(self):
    # Adding the generator functions.
    self.run_generator_forward = self.cycle_gan.run_generator_forward
    self.run_generator_backward = self.cycle_gan.run_generator_backward
    self.run_discriminator_backward_a = (
        self.cycle_gan.run_discriminator_backward_a
    )
    self.run_discriminator_backward_b = (
        self.cycle_gan.run_discriminator_backward_b
    )

  def generator_loss_and_samples(
      self,
      params: tuple[PyTree, ...],
      batch: swirl_models.BatchType,
      rng: Array,
      mutables: PyTree,
  ) -> tuple[Array, tuple[swirl_models.ArrayDict, PyTree, tuple[Array, ...]]]:
    """Generator loss together with the generated samples.

    Trainers differentiate this function in the generator phase and feed the
    returned samples to the discriminator phase, which avoids a second
    generator forward pass.

    Args:
      params: Parameters of the four networks, in the order `(gen_a2b, gen_b2a,
        dis_a, dis_b)`. The discriminator parameters are stop-gradiented.
      batch: Dictionary with keys `"real_data_a"` and `"real_data_b"`.
      rng: Random key for dropout.
      mutables: Mutable variable collections, returned unchanged.

    Returns:
      `(loss, (metrics, mutables, generated_data))` where `metrics` contains
      the total `loss` and its components (see `run_generator_backward`) and
      `generated_data` is `(fake_b, recover_a, fake_a, recover_b)`.
    """
    params_gen_a2b = params[0]
    params_gen_b2a = params[1]
    params_dis_a = params[2]
    params_dis_b = params[3]

    real_data = (
        jnp.asarray(batch["real_data_a"]),
        jnp.asarray(batch["real_data_b"]),
    )

    rng_forward, rng_backward = jax.random.split(rng)

    # Run the model forward
    generated_data = self.run_generator_forward(
        {"dropout": rng_forward},
        (params_gen_a2b, params_gen_b2a),
        real_data,
        is_training=True,
    )
    backward_params = (
        params_gen_a2b,
        params_gen_b2a,
        jax.lax.stop_gradient(params_dis_a),
        jax.lax.stop_gradient(params_dis_b),
    )

    loss_dict = self.run_generator_backward(
        {"dropout": rng_backward},
        backward_params,
        generated_data,
        real_data,
        is_training=True,
    )

    loss = sum(loss_dict.values(), start=jnp.zeros(()))

    # Add all the metrics from run generator backwards.
    metrics = dict(loss=loss, **loss_dict)

    return loss, (metrics, mutables, generated_data)

  def loss_fn(
      self,
      params: tuple[PyTree, ...],
      batch: swirl_models.BatchType,
      rng: Array,
      mutables: PyTree,
      **kwargs,
  ) -> swirl_models.LossAndAux:
    """Generator loss in the `BaseModel` format `(loss, (metrics, mutables))`.

    The generators are updated by generating data and letting the (frozen)
    discriminators critique it: the loss decreases when a discriminator
    wrongly predicts the generated data to be real.

    Args:
      params: See `generator_loss_and_samples`.
      batch: See `generator_loss_and_samples`.
      rng: See `generator_loss_and_samples`.
      mutables: See `generator_loss_and_samples`.
      **kwargs: Unused additional keyword arguments.

    Returns:
      The loss and the `(metrics, mutables)` auxiliary pair.
    """
    del kwargs  # Unused.
    loss, (metrics, mutables, _) = self.generator_loss_and_samples(
        params, batch, rng, mutables
    )
    return loss, (metrics, mutables)

  @staticmethod
  def inference_fn(variables: PyTree, generator: nn.Module):  # pyrefly: ignore[bad-override]
    """Returns the deterministic generator map `x -> generator(x)`.

    Args:
      variables: Variables of `generator`, e.g. `{"params": ...}`.
      generator: One of the two generators of the model.

    Returns:
      A function mapping a batch of samples to their translations, with
      dropout disabled.
    """

    def _translate(x: Array):
      return generator.apply(variables, x, is_training=False)

    return _translate

  def discriminator_loss_and_metrics(
      self,
      params: tuple[PyTree, ...],
      batch: swirl_models.BatchType,
      rng: Array,
      mutables: PyTree,
  ) -> tuple[Array, swirl_models.ArrayDict]:
    """Discriminator loss together with its per-discriminator terms.

    The discriminators are updated by critiquing both real and generated data;
    the loss decreases as they correctly tell real samples from generated ones.

    Args:
      params: Parameters of the four networks, in the order `(gen_a2b, gen_b2a,
        dis_a, dis_b)`; only the discriminator parameters are used.
      batch: Dictionary with keys `"real_data_a"`, `"real_data_b"`,
        `"fake_data_a"` and `"fake_data_b"`.
      rng: Random key, unused by the current criteria.
      mutables: Mutable variable collections, unused by the current networks.

    Returns:
      `(loss, metrics)` where `loss = loss_dis_a + loss_dis_b +
      loss_dis_crossed`
      (`loss_dis_crossed` is 0 when `use_crossed_dis_loss` is `False`), and
      `metrics` holds `loss_dis`, `loss_dis_a`, `loss_dis_b` and
      `loss_dis_crossed`.
    """
    del rng, mutables  # Unused.
    real_data_a = jnp.asarray(batch["real_data_a"])
    real_data_b = jnp.asarray(batch["real_data_b"])

    fake_data_a = jnp.asarray(batch["fake_data_a"])
    fake_data_b = jnp.asarray(batch["fake_data_b"])

    params_dis_a = params[2]
    params_dis_b = params[3]

    # Loss for the discriminator of A.
    loss_a = self.cycle_gan.run_discriminator_backward_a(
        params_dis_a, real_data_a, fake_data_a
    )

    # Loss for the discriminator of B.
    loss_b = self.cycle_gan.run_discriminator_backward_b(
        params_dis_b, real_data_b, fake_data_b
    )

    # Training discriminators to discriminate samples from A and B.
    if self.cycle_gan.use_crossed_dis_loss:
      loss_crossed = self.cycle_gan.run_discriminators_crossed(
          (params_dis_a, params_dis_b), real_data_a, real_data_b
      )
    else:
      loss_crossed = jnp.zeros_like(loss_a)
    loss = loss_a + loss_b + loss_crossed

    return loss, dict(
        loss_dis=loss,
        loss_dis_a=loss_a,
        loss_dis_b=loss_b,
        loss_dis_crossed=loss_crossed,
    )

  def loss_fn_discriminator(
      self,
      params: tuple[PyTree, ...],
      batch: swirl_models.BatchType,
      rng: Array,
      mutables: PyTree,
  ) -> Array:
    """Total discriminator loss, see `discriminator_loss_and_metrics`.

    Args:
      params: See `discriminator_loss_and_metrics`.
      batch: See `discriminator_loss_and_metrics`.
      rng: See `discriminator_loss_and_metrics`.
      mutables: See `discriminator_loss_and_metrics`.

    Returns:
      The total loss of the discriminators.
    """
    return self.discriminator_loss_and_metrics(params, batch, rng, mutables)[0]

  def eval_metrics_and_samples(
      self,
      variables: tuple[PyTree, ...],
      batch: swirl_models.BatchType,
      rng: Array,
  ) -> tuple[swirl_models.ArrayDict, tuple[Array, Array, Array, Array]]:
    """Evaluation metrics together with the deterministic generator outputs.

    Args:
      variables: Variables of the four networks, in the order `(gen_a2b,
        gen_b2a, dis_a, dis_b)`; only their `"params"` collections are used.
      batch: Dictionary with keys `"real_data_a"` and `"real_data_b"`.
      rng: Random key (dropout is disabled, so it only seeds the split).

    Returns:
      `(metrics, generated_data)` where `metrics` holds the generator losses
      (`loss` and its components, see `run_generator_backward`) and the
      discriminator losses `loss_dis_a` and `loss_dis_b` (the optional crossed
      discriminator term is a training-only objective and is not reported here),
      and `generated_data` is `(fake_b, recover_a, fake_a, recover_b)`.
    """
    params_gen_a2b = variables[0]["params"]
    params_gen_b2a = variables[1]["params"]
    params_dis_a = variables[2]["params"]
    params_dis_b = variables[3]["params"]

    real_data = (
        jnp.asarray(batch["real_data_a"]),
        jnp.asarray(batch["real_data_b"]),
    )

    rng_forward, rng_backward = jax.random.split(rng)

    # Run the model forward.
    generated_data = self.run_generator_forward(
        {"dropout": rng_forward},
        (params_gen_a2b, params_gen_b2a),
        real_data,
        is_training=False,
    )

    backward_params = (
        params_gen_a2b,
        params_gen_b2a,
        jax.lax.stop_gradient(params_dis_a),
        jax.lax.stop_gradient(params_dis_b),
    )

    loss_dict = self.run_generator_backward(
        {"dropout": rng_backward},
        backward_params,
        generated_data,
        real_data,
        is_training=False,
    )

    loss = sum(loss_dict.values(), start=jnp.zeros(()))

    real_data_a = real_data[0]
    real_data_b = real_data[1]

    fake_data_a = generated_data[2]
    fake_data_b = generated_data[0]

    # Loss for the discriminator of A.
    loss_a = self.cycle_gan.run_discriminator_backward_a(
        params_dis_a, real_data_a, fake_data_a
    )

    # Loss for the discriminator of B.
    loss_b = self.cycle_gan.run_discriminator_backward_b(
        params_dis_b, real_data_b, fake_data_b
    )

    metrics = dict(loss=loss, loss_dis_a=loss_a, loss_dis_b=loss_b, **loss_dict)
    return metrics, generated_data

  def eval_fn(
      self,
      variables: tuple[PyTree, ...],
      batch: swirl_models.BatchType,
      rng: Array,
      **kwargs,
  ) -> swirl_models.ArrayDict:
    """Evaluation metrics of the generators and discriminators.

    Args:
      variables: See `eval_metrics_and_samples`.
      batch: See `eval_metrics_and_samples`.
      rng: See `eval_metrics_and_samples`.
      **kwargs: Unused additional keyword arguments.

    Returns:
      See `eval_metrics_and_samples`.
    """
    del kwargs  # Unused.
    return self.eval_metrics_and_samples(variables, batch, rng)[0]
