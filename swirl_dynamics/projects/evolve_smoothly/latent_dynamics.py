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

"""LatentDynamics model and trainer."""

from collections.abc import Callable
import dataclasses
import functools
from typing import Any

from clu import metrics as clu_metrics
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import metrics
from swirl_dynamics.lib.networks import hyper_unet
from swirl_dynamics.lib.networks import nonlinear_fourier
from swirl_dynamics.lib.solvers import ode
from swirl_dynamics.projects.evolve_smoothly import ansatzes
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states
from swirl_dynamics.templates import trainers

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
EncodingFn = Callable[[Array], Array]
PyTree = Any


# Helpers from instantiating latent dynamical models
def create_mlp_dynamics_model(
    ansatz: ansatzes.Ansatz, features: tuple[int, ...], **kwargs
) -> nn.Module:
  return nonlinear_fourier.MLP(
      features=features + (ansatz.num_params,), **kwargs
  )


def create_hyperunet_dynamics_model(
    ansatz: ansatzes.Ansatz, **kwargs
) -> nn.Module:
  return hyper_unet.HyperUnet(
      flat_layer_shapes=ansatz.num_params_in_layers, **kwargs
  )


# Model and trainer
@dataclasses.dataclass(frozen=True, kw_only=True)
class LatentDynamics(models.BaseModel):
  """Latent dynamics training."""

  encoder: EncodingFn
  ansatz: ansatzes.Ansatz
  latent_dynamics_model: nn.Module
  integrator: ode.OdeSolver
  reconstruction_weight: float = 0.0
  latent_weight: float = 1.0
  consistency_weight: float = 1.0

  def initialize(self, rng: Array) -> models.ModelVariable:
    """Initializes the variables of the dynamics model."""
    sample_input = jnp.ones((1, self.ansatz.num_params))
    out, variables = self.latent_dynamics_model.init_with_output(
        rng, sample_input
    )
    assert out.shape[-1] == self.ansatz.num_params
    return variables

  def loss_fn(
      self,
      params: PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: PyTree,
  ) -> models.LossAndAux:
    """Computes the reconstruction and consistency loss."""
    ambient_ic, ambient_target = batch["u"][:, 0], batch["u"]
    tspan = batch["t"]
    grid = batch["x"]
    inference_fn = self.inference_fn(
        variables=dict(params=params, **mutables),
        encoder=self.encoder,
        ansatz=self.ansatz,
        latent_dynamics_model=self.latent_dynamics_model,
        integrator=self.integrator,
        return_latents=True,
    )
    reconstruction, latent_rollout = inference_fn(ambient_ic, tspan, grid)
    reencoding = jax.vmap(self.encoder)(reconstruction)
    latent_target = jax.vmap(self.encoder)(ambient_target)
    reconstruction = jnp.reshape(reconstruction, ambient_target.shape)

    reconstruction_loss = jnp.mean(jnp.square(reconstruction - ambient_target))
    latent_loss = jnp.mean(jnp.square(latent_rollout - latent_target))
    consistency_loss = jnp.mean(jnp.square(reencoding - latent_rollout))
    loss = (
        self.reconstruction_weight * reconstruction_loss
        + self.latent_weight * latent_loss
        + self.consistency_weight * consistency_loss
    )
    metric = dict(
        loss=loss,
        reconstruction_loss=reconstruction_loss,
        latent_loss=latent_loss,
        consistency_loss=consistency_loss,
    )
    return loss, (metric, mutables)

  def eval_fn(
      self,
      variables: PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Evaluates mean, worst-case and std relative l2 errors."""
    ambient_ic, ambient_target = batch["u"][:, 0], batch["u"]
    tspan = batch["t"]
    grid = batch["x"]  # grid is single vector that applies to all time stamps
    inference_fn = self.inference_fn(
        variables=variables,
        encoder=self.encoder,
        ansatz=self.ansatz,
        latent_dynamics_model=self.latent_dynamics_model,
        integrator=self.integrator,
        return_latents=True,
    )
    reconstruction, latent_rollout = inference_fn(ambient_ic, tspan, grid)
    reencoding = jax.vmap(self.encoder)(reconstruction)
    latent_target = jax.vmap(self.encoder)(ambient_target)
    reconstruction = jnp.reshape(reconstruction, ambient_target.shape)
    rrmse = functools.partial(
        metrics.mean_squared_error,
        sum_axes=(-1, -2),
        relative=True,
        squared=False,
    )
    return dict(
        reconstruction_rel_l2=rrmse(pred=reconstruction, true=ambient_target),
        latent_rel_l2=rrmse(pred=latent_rollout, true=latent_target),
        consistency_rel_l2=rrmse(pred=reencoding, true=latent_rollout),
    )

  @staticmethod
  def inference_fn(
      variables: PyTree,
      encoder: EncodingFn,
      ansatz: ansatzes.Ansatz,
      latent_dynamics_model: nn.Module,
      integrator: ode.OdeSolver,
      return_latents: bool = False,
  ) -> Callable[[ArrayLike, ArrayLike, ArrayLike], Array | tuple[Array, Array]]:
    """Returns an encoder inference function."""
    latent_dynamics_fn = ode.nn_module_to_dynamics(latent_dynamics_model)
    integrate_fn = functools.partial(
        integrator, latent_dynamics_fn, params=variables
    )

    def dynamical_model(
        u0: ArrayLike, tspan: ArrayLike, grid: ArrayLike
    ) -> Array | tuple[Array, Array]:
      """Evolves an initial condition and evaluates on a fixed grid.

      Args:
        u0: initial condition in ambient space. shape ~ (nbatch, ngrid, 1)
        tspan: time stamps for integration. shape ~ (nbatch, ntime)
        grid: grid on which to evaluate evolved latent states. shape ~ (ngrid,
          1)

      Returns:
        Ambient space trajectories of shape ~ (nbatch, ntime, ngrid, 1) and
        optionally latent trajectores of shape ~ (nbatch, ntime, latent_dim).
      """
      v0 = encoder(u0)
      v = jax.vmap(integrate_fn, in_axes=(0, 0))(v0, tspan)
      u = jax.vmap(
          jax.vmap(ansatz.batch_evaluate, in_axes=(0, None)), in_axes=(0, 0)
      )(v, grid)
      return (u, v) if return_latents else u

    return dynamical_model


TrainState = train_states.BasicTrainState


class LatentDynamicsTrainer(trainers.BasicTrainer[LatentDynamics, TrainState]):
  """Trainer for the LatentDynamics model."""

  @flax.struct.dataclass
  class TrainMetrics(clu_metrics.Collection):
    """Training metrics declaration for `LatentDynamics` model."""

    train_loss: clu_metrics.Average.from_output("loss")
    train_reconstruction_loss: clu_metrics.Average.from_output(
        "reconstruction_loss"
    )
    train_latent_loss: clu_metrics.Average.from_output("latent_loss")
    train_consistency_loss: clu_metrics.Average.from_output("consistency_loss")

  @flax.struct.dataclass
  class EvalMetrics(clu_metrics.Collection):
    """Evaluation metrics declaration for `LatentDynamics` model."""

    eval_reconstruction_rel_l2: clu_metrics.Average.from_output(
        "reconstruction_rel_l2"
    )
    eval_latent_rel_l2: clu_metrics.Average.from_output("latent_rel_l2")
    eval_consistency_rel_l2: clu_metrics.Average.from_output(
        "consistency_rel_l2"
    )

  @staticmethod
  def build_inference_fn(
      state: TrainState, **kwargs
  ) -> Callable[[ArrayLike, ArrayLike, ArrayLike], Array | tuple[Array, Array]]:
    """Builds an inference dynamical model from a train state."""
    return LatentDynamics.inference_fn(state.model_variables, **kwargs)
