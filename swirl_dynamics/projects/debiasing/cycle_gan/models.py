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

"""CycleGAN module.

References:
[1] Zhu, J.Y., Park, T., Isola, P. and Efros, A.A., "Unpaired image-to-image
    translation using cycle-consistent adversarial networks". In Proceedings of
    the IEEE international conference on computer vision (pp. 2223-2232) 2017.
[2] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190#issuecomment-358546675  # pylint: disable=unused-argument: disable=line-too-long
"""

import dataclasses
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.networks import cycle_gan
from swirl_dynamics.projects.debiasing.cycle_gan import loss_functions
from swirl_dynamics.templates import models as swirl_models


PyTree = Any
Array = jax.Array
Initializer = nn.initializers.Initializer


@dataclasses.dataclass(kw_only=True)
class CycleGANModelConfig:
  """Config used by cycleGAN model.

  Attributes:
    lambda_a: Weight for the generator/discriminator loss for A.
    lambda_b: Weight for the generator/discriminator loss for B.
    lambda_id: Weight for the identiy loss as defined in [1].
    gan_mode: Type of loss, it currently accepts 3 types, least squares gan
      ("lsgan"), the vanilla gan with BCE loss ("bce_with_logits"), and
      Wasserstein gan ("wganp").
    use_cycle_loss: Option to use the cycle loss as defined in [1].
    same_dimension: Option so the dimensions of elements of A and B matches.
    use_crossed_dis_loss: Use a crossed loss. Basically
    internal_shape_dims: Shape dimensions in which the Cycle-GAN networks
      operates. This is necessary when we interpolate the inputs/outputs.
  """

  lambda_a: float = 1.0
  lambda_b: float = 1.0
  lambda_id: float = 1.0
  gan_mode: loss_functions.GanMode = loss_functions.GanMode.LSGAN
  use_cycle_loss: bool = True
  same_dimension: bool = True
  use_crossed_dis_loss: bool = False
  internal_shape_dims: tuple[int, ...] | None = None


class CycleGAN(nn.Module):
  """Module encapsulating the CycleGAN framework.

  Attributes:
    config: Configuration file.
    generator_a2b: Generator for samples from A to B.
    generator_b2a: Generator for samples from B to A.
    dis_a: discriminator for samples in A.
    dis_b: Discriminator for samples in B.
    lambda_a: weight for losses with respect to A.
    lambda_b: weight for losses with respect to B.
    lambda_id: weight for the identity losses.
  """

  config: CycleGANModelConfig

  def __init__(self, config: CycleGANModelConfig,
               generator_a2b: nn.Module,
               generator_b2a: nn.Module,
               discriminator_a: nn.Module,
               discriminator_b: nn.Module):
    """Initializes the CycleGAN module.

    Args:
      config: Configuration dictionary.
      generator_a2b: Generator from distribution A to distribution B.
      generator_b2a: Generator from distribution B to distribution A.
      discriminator_a: Network to gauge if a sample is in distribution a.
      discriminator_b: Network to gauge if a sample is in distribution b.

    Returns:
      A class of the CycleGAN

    Config needs to have at least neural networks in the config dictionary.
    Namely two for the generators A -> B and B -> A, and two for the
    discriminators, one for A, and the other for B. We assume that the
    dimensions of elements in A and B are the same.
    """

    self.config = config

    self.generator_a2b = generator_a2b
    self.generator_b2a = generator_b2a

    self.dis_a = discriminator_a
    self.dis_b = discriminator_b

    self.criterion_gan = loss_functions.GanLoss(gan_mode=config.gan_mode)()
    self.criterion_cycle = loss_functions.l1_loss
    self.criterion_id = loss_functions.l1_loss

    self.lambda_a = config.lambda_a  # weight of loss on inputs from set A
    self.lambda_b = config.lambda_b  # weight of loss on inputs from set B
    self.lambda_id = config.lambda_id  # weight of identity loss

    # Using cross loss for the discriminant so they can differentiate A from B.
    self.use_crossed_dis_loss = config.use_crossed_dis_loss

  def init(
      self, rng: Array, *args, **kwargs
  ) -> tuple[PyTree, PyTree, PyTree, PyTree]:
    """Initialize the weights of all the networks.

    Args:
      rng: Random generator key.
      *args: The dimension of samples in a and b.
      **kwargs: Extra key arguments.

    Returns:
      The weights of the given network.
    """

    rng_array = jax.random.split(rng, num=4)

    dummy_input_a, dummy_input_b = args

    params_a2b = self.generator_a2b.init(
        rng_array[0], dummy_input_a, is_training=False
    )
    params_b2a = self.generator_a2b.init(
        rng_array[1], dummy_input_b, is_training=False
    )

    params_dist_a = self.dis_a.init(rng_array[2], dummy_input_a)
    params_dist_b = self.dis_b.init(rng_array[3], dummy_input_b)

    return params_a2b, params_b2a, params_dist_a, params_dist_b

  def run_generator_forward(
      self,
      rngs: dict[str, Array],
      params_gen: tuple[PyTree, PyTree],
      real_data: tuple[Array, Array],
      is_training: bool = True,
  ) -> tuple[Array, Array, Array, Array]:
    """We run the generator forward.

    Args:
      rngs: Random seeds for the dropout layers within a dictionary following
        '{"dropout": ...}'.
      params_gen: Parameters of the generators only, following
        '(params_gen_a2b, params_gen_b2a)'.
      real_data: Data to generate samples from A to B and B to A. The data
        follows '(real_a, real_b)'
      is_training: toggle that affects dropout layer.

    Returns:
      A tuple containing the generated samples.
    """

    # TODO: perhaps use dictionaries instead of positional tuples.
    params_gen_a2b = params_gen[0]
    params_gen_b2a = params_gen[1]

    real_a = real_data[0]
    real_b = real_data[1]

    # Forward through the generator.
    # fake_b = gen_a2b(u_a).
    fake_b = self.generator_a2b.apply(
        {"params": params_gen_a2b}, real_a, is_training=is_training, rngs=rngs
    )

    # recover_a = gen_b2a(gen_a2b(u_a)).
    recover_a = self.generator_b2a.apply(
        {"params": params_gen_b2a}, fake_b, is_training=is_training, rngs=rngs
    )

    # fake_a = gen_b2a(u_b).
    fake_a = self.generator_b2a.apply(
        {"params": params_gen_b2a}, real_b, is_training=is_training, rngs=rngs
    )

    # recover_b = gen_a2b(gen_b2a(u_b)).
    recover_b = self.generator_a2b.apply(
        {"params": params_gen_a2b}, fake_a, is_training=is_training, rngs=rngs
    )

    return (fake_b, recover_a, fake_a, recover_b)

  def run_generator_forward_a2b(
      self,
      rngs: dict[str, Array],
      params_gen_a2b: PyTree,
      real_data_a: Array,
      is_training: bool = False,
  ) -> Array:
    """We run the generator forward to translate samples from A to B.

    Args:
      rngs: {"dropout": ...}
      params_gen_a2b: Parameters of for the a2b generator.
      real_data_a: Data from distribution a.
      is_training: toggle that affects dropout layer

    Returns:
      A tuple containing the generated samples.
    """

    # Forward through the generator
    # fake_b = gen_a2b(u_a)
    fake_b = self.generator_a2b.apply({"params": params_gen_a2b},
                                      real_data_a,
                                      is_training=is_training,
                                      rngs=rngs)

    return fake_b

  def run_generator_backward(
      self,
      rngs: dict[str, Array],
      params: tuple[PyTree, ...],
      generated_data: tuple[Array, Array, Array, Array],
      real_data: tuple[Array, Array],
      is_training: bool = True,
  ) -> dict[str, Array]:
    """Run the generator backwards.

    Args:
      rngs: Random seed for the dropout functions.
      params: (params_g_a2b, params_g_b2a, params_d_a, params_d_b)
      generated_data: (fake_b, recover_a, fake_a, recover_b)
      real_data: (real_a, real_b)
      is_training: if the applicaton of the networks is supposed to be used for
        training.

    Returns:
      A tuple containing the different losses.
    """
    # Parameters for the generators.
    params_g_a2b = params[0]
    params_g_b2a = params[1]
    # Parameters for the discriminators.
    params_d_a = params[2]
    params_d_b = params[3]

    fake_b = generated_data[0]
    recover_a = generated_data[1]
    fake_a = generated_data[2]
    recover_b = generated_data[3]

    real_a = real_data[0]
    real_b = real_data[1]

    # Compute 3-criteria loss function.

    # Generator loss dis_a(gen_a2b(u_a)).
    disc_a_gen_a2b = self.dis_a.apply({"params": params_d_a}, fake_a)
    loss_gen_a = self.criterion_gan(disc_a_gen_a2b, True)

    # Generator loss dis_b(gen_b2a(u_b)).
    disc_b_gen_b2a = self.dis_b.apply({"params": params_d_b}, fake_b)
    loss_gen_b = self.criterion_gan(disc_b_gen_b2a, True)

    if self.config.use_cycle_loss:
      # Cycle loss ||gen_b2a(gen_a2b(u_a)) - u_a||.
      loss_cycle_a = self.criterion_cycle(recover_a, real_a) * self.lambda_a
      # Cycle loss ||gen_a2b(gen_b2a(u_b)) - u_b||.
      loss_cycle_b = self.criterion_cycle(recover_b, real_b) * self.lambda_b
    else:
      loss_cycle_a, loss_cycle_b = 0., 0.

    # Identity losses only if the dimensions of a and b are the same.
    if self.config.same_dimension:
      # G_A should be identity if real_b is fed: ||gen_a2b(u_b) - u_b||.
      id_a = self.generator_a2b.apply(
          {"params": params_g_a2b}, real_b, is_training=is_training, rngs=rngs
      )
      # G_B should be identity if real_a is fed: ||gen_b2a(u_a) - u_a||.
      id_b = self.generator_b2a.apply(
          {"params": params_g_b2a}, real_a, is_training=is_training, rngs=rngs
      )

      loss_id_a = (
          self.lambda_b * self.lambda_id * self.criterion_id(id_a, real_b)
      )
      loss_id_b = (
          self.lambda_a * self.lambda_id * self.criterion_id(id_b, real_a)
      )

    else:

      loss_id_a, loss_id_b = 0., 0.

    return dict(loss_gen_a=jnp.array(loss_gen_a),
                loss_gen_b=jnp.array(loss_gen_b),
                loss_cycle_a=jnp.array(loss_cycle_a),
                loss_cycle_b=jnp.array(loss_cycle_b),
                loss_id_a=jnp.array(loss_id_a),
                loss_id_b=jnp.array(loss_id_b))

  def run_discriminator_backward_a(self,
                                   params: PyTree,
                                   real_a: Array,
                                   fake_a: Array) -> Array:
    """Running the disctriminator loss on samples from A.

    Args:
        params: Parameters for one discriminator for samples from A.
        real_a: Real image from dataset A.
        fake_a: Generated image from the generator.

    Returns:
      The average of the losses from the discriminator in A.
    """
    # Real data being discriminated.
    pred_real = self.dis_a.apply({"params": params}, real_a)
    loss_dis_real = self.criterion_gan(pred_real, True)

    # Fake data, i.e, generated by generator_b2a being discriminated.
    # Here we stop the gradient to we don't update the generator_b2a.
    pred_fake = self.dis_a.apply({"params": params},
                                 jax.lax.stop_gradient(fake_a))
    loss_dis_fake = self.criterion_gan(pred_fake, False)

    # Combined loss to calculate gradients.
    loss = (loss_dis_real + loss_dis_fake) * 0.5

    return loss

  def run_discriminator_backward_b(self, params: PyTree,
                                   real_b: Array,
                                   fake_b: Array) -> Array:
    """Running the disctriminator backwards on samples from B.

    Args:
        params: Parameters for the discriminator for samples from B.
        real_b: Real image from dataset B.
        fake_b: G.nerated image from the generator.

    Returns:
      The average of the losses from the discriminator in B.
    """
    # Real data being discriminated.
    pred_real = self.dis_b.apply({"params": params}, real_b)
    loss_dis_real = self.criterion_gan(pred_real, True)

    # Fake data, i.e, generated by generator_a2b being discriminated.
    # Here we stop the gradient to we don't update the generator_a2b.
    pred_fake = self.dis_b.apply({"params": params},
                                 jax.lax.stop_gradient(fake_b))
    loss_dis_fake = self.criterion_gan(pred_fake, False)

    # Combined loss to calculate gradients.
    loss = (loss_dis_real + loss_dis_fake) * 0.5

    return jnp.array(loss)

  def run_discriminators_crossed(self, params, real_a, real_b) -> Array:
    """Running the disctriminators backwards on samples from A and B.

    Args:
        params: Parameters for one discriminator.
        real_a: Real image from dataset A.
        real_b: Real image from dataset B.

    Returns:
      The average of the losses using the wrong datasets for each discriminator.
    """

    params_dis_a = params[0]
    params_dis_b = params[1]

    pred_real_a = self.dis_a.apply({"params": params_dis_a}, real_b)
    loss_dis_real_a = self.criterion_gan(pred_real_a, False)

    pred_real_b = self.dis_b.apply({"params": params_dis_b}, real_a)
    loss_dis_real_b = self.criterion_gan(pred_real_b, False)

    # Combined loss to calculate gradients.
    loss = (loss_dis_real_a + loss_dis_real_b) * 0.5

    return loss


@dataclasses.dataclass(kw_only=True)
class CycleGANConfig:
  """Config used by cycleGAN models."""

  generator_a2b: cycle_gan.Generator
  generator_b2a: cycle_gan.Generator

  dims_a: tuple[int, ...]
  dims_b: tuple[int, ...]

  discriminator_a: cycle_gan.Discriminator
  discriminator_b: cycle_gan.Discriminator

  lambda_a: float
  lambda_b: float
  lambda_id: float


@dataclasses.dataclass(kw_only=True)
class CycleGANModel(swirl_models.BaseModel):
  """Model used storing cycleGan information.

  Attributes:
    cycle_gan: Flax Module containing all the functionatly for CycleGAN.
    dim_inputs: The dimension of the inputs for both samples from A and B.
      These shapes may de different from the ones inside the model due to an
      interpolation step.
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

  def loss_fn(
      self,
      params: tuple[PyTree, ...],  # this would be a tuple of parameters.
      batch: swirl_models.BatchType,
      rng: Array,
      mutables: PyTree,
  ) -> tuple[
      jax.Array, tuple[swirl_models.ArrayDict, PyTree, tuple[Array, ...]]
  ]:  # pytype: disable=signature-mismatch
    """Loss function for the generator.

    Args:
      params: Parameters for the neural networks within the CycleGAN module.
      batch: Batch dictionary with containing the real data.
      rng: Random seed.
      mutables: The rest of the mutables in the flax modules.

    Returns:
      The loss and auxiliary fields. In this case we add an extra field that
      stores the values of the generated samples, which will be used later for
      computing the discriminators loss.

    The generator is updated by generating data and letting the discriminator
    critique it. It's loss goes down if the discriminator wrongly predicts it to
    to be real data.
    """
    # Split the States.
    # TODO: specify how to split the parameters.

    params_gen_a2b = params[0]
    params_gen_b2a = params[1]
    params_dis_a = params[2]
    params_dis_b = params[3]

    real_data = (batch["real_data_a"], batch["real_data_b"])

    rng_forward, rng_backward = jax.random.split(rng)

    # Run the model forward
    generated_data = self.run_generator_forward(
        {"dropout": rng_forward},
        (params_gen_a2b, params_gen_b2a),
        real_data,
        is_training=True
    )
    backward_params = (params[0], params[1],
                       jax.lax.stop_gradient(params_dis_a),
                       jax.lax.stop_gradient(params_dis_b))

    loss_dict = self.run_generator_backward(
        {"dropout": rng_backward},
        backward_params,
        generated_data,
        real_data,
        is_training=True,
    )

    loss = jnp.sum(jnp.array([loss_dict[key] for key in loss_dict.keys()]))

    # Add all the metrics from run generator backwards.
    metrics = dict(
        loss=loss,
        **loss_dict
    )

    return jnp.array(loss), (metrics, mutables, generated_data)

  def loss_fn_discriminator(
      self,
      params: tuple[PyTree, ...],
      batch: swirl_models.BatchType,
      rng: Array,  # pylint: disable=unused-argument
      mutables: PyTree,  # pylint: disable=unused-argument
  ) ->  Array:
    """Loss function for the discriminator.

    Args:
      params: Parameters of the networks within the cyclaGAN class, here they
        are a tuple of PyTrees.
      batch: Data for training, as dictionary-like object with fields
        '["real_data", "fake_data"]'.
      rng: Random seed for the random number generator.
      mutables: Set of mutables for the networks. In this case this is a dummy
        variable, necessary due to the interface.

    Returns:
      The total loss for the discriminators.

      The discriminator is updated by critiquing both real and generated data.
      The value of the loss decreased as it predicts correctly if images are
      real (part of the training set) or generated by the neural network.
    """
    real_data_a = batch["real_data_a"]
    real_data_b = batch["real_data_b"]

    fake_data_a = batch["fake_data_a"]
    fake_data_b = batch["fake_data_b"]

    params_dis_a = params[2]
    params_dis_b = params[3]

    # Loss for the discriminator of A.
    loss_a = self.cycle_gan.run_discriminator_backward_a(
        params_dis_a, real_data_a, fake_data_a
    )

    # Step for D_B
    loss_b = self.cycle_gan.run_discriminator_backward_b(
        params_dis_b, real_data_b, fake_data_b
    )

    # Training discriminators to discriminate samples from A and B.
    if self.cycle_gan.use_crossed_dis_loss:
      loss_crossed = self.cycle_gan.run_discriminators_crossed(
          (params_dis_a, params_dis_b), real_data_a, real_data_b
      )
    else:
      loss_crossed = 0

    return jnp.array(loss_a + loss_b + loss_crossed)

  def eval_fn(
      self,
      variables: tuple[PyTree, ...],
      batch: swirl_models.BatchType,
      rng: Array,
      **kwargs,
  ) -> swirl_models.ArrayDict:
    """Eval function for the generator and discriminator.

    Args:
      variables: Parameters of the different neural networks within the cycle-
        gan class.
      batch: Data as a dictionary with keys  ['real_data_a', 'real_data_b',
      'fake_data_a', 'fake_data_b'].
      rng: Random seed.
      **kwargs: Additional keyed arguments.

    Returns:
      A dictionary with the evaluation metrics.
    """

    # The variables is a tuple of dicts containing the params and the mutables
    # we are not using the mutables, so we only extract the parameters.
    params_gen_a2b = variables[0]["params"]
    params_gen_b2a = variables[1]["params"]
    params_dis_a = variables[2]["params"]
    params_dis_b = variables[3]["params"]

    real_data = (batch["real_data_a"], batch["real_data_b"])

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

    loss = jnp.sum(jnp.array([loss_dict[key] for key in loss_dict.keys()]))

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

    return dict(loss=loss,
                loss_dis_a=loss_a,
                loss_dis_b=loss_b,
                u_lf=real_data_a,
                u_hf=fake_data_b,
                **loss_dict)
  # pytype: enable=bad-return-type
