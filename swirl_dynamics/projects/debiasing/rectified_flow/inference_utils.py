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

"""Utility functions for inference."""

import functools

import jax
import jax.numpy as jnp
from swirl_dynamics.lib.solvers import ode as ode_solvers
from swirl_dynamics.projects.debiasing.rectified_flow import models as reflow_models
from swirl_dynamics.templates import models
from swirl_dynamics.templates import train_states


def move_time_to_channel(
    input_array: jax.Array, time_chunk_size: int, time_to_channel: bool = True
) -> jax.Array:
  """Moves the time dimension to a new or the channel dimension.

  Args:
    input_array: The input array to move the time dimension.
    time_chunk_size: The size of the time chunks in the batch.
    time_to_channel: Whether to move the time dimension to the channel
      dimension or add it to another dimension between the batch and the spatial
      dimensions.

  Returns:
    The input array with the time dimension moved to the channel dimension or a
    new dimension between the batch and the spatial dimensions.
  """
  if input_array.shape[0] % time_chunk_size != 0:
    raise ValueError("Batch dimension should be multiple a time chunk size.")

  new_chunk = input_array.shape[0] // time_chunk_size
  new_channel = input_array.shape[-1] * time_chunk_size

  # Shape: (new_chunk, time_chunk_size, lon, lat, channel)
  input_new = jnp.reshape(
      input_array, (new_chunk, time_chunk_size, *input_array.shape[1:])
  )

  if time_to_channel:
    # Shape: (new_chunk, lon, lat, channel, time_chunk_size)
    input_new = jnp.moveaxis(input_new, 1, -1)
    # Shape: (new_chunk, lon, lat, new_channel)
    input_new = jnp.reshape(input_new, (*input_new.shape[:-2], new_channel))

  return input_new


def move_channel_to_time(
    input_array: jax.Array, time_chunk_size: int, time_to_channel: bool = True
) -> jax.Array:
  """Moves the time dimension back to the batch dimension.

  Args:
    input_array: The input array to move the time dimension.
    time_chunk_size: The size of the time chunks in the batch.
    time_to_channel: Whether to move the time dimension back from the channel
      dimension or to squeeze the first two dimensions together.

  Returns:
    The input array with the time dimension moved back to the batch dimension.
    The output of this function is the inverse of the move_time_to_channel
    function.
  """
  new_chunk = input_array.shape[0] * time_chunk_size
  new_channel = input_array.shape[-1] // time_chunk_size

  if time_to_channel:
    # Shape: (batch, lon, lat, new_channel, time_chunk_size)
    input_new = jnp.reshape(
        input_array, (*input_array.shape[:-1], new_channel, time_chunk_size)
    )
    # Shape: (batch, time_chunk_size, lon, lat, new_channel)
    input_new = jnp.moveaxis(input_new, -1, 1)
    # Shape: (new_batch, lon, lat, new_channel)
    input_new = jnp.reshape(input_new, (new_chunk, *input_new.shape[2:]))
  else:
    input_new = jnp.reshape(input_array, (new_chunk, *input_array.shape[2:]))
  return input_new


def sampling_from_batch(
    batch: models.BatchType,
    model: reflow_models.ReFlowModel | reflow_models.ConditionalReFlowModel,
    trained_state: train_states.BasicTrainState,
    num_sampling_steps: int,
    time_chunk_size: int,
    time_to_channel: bool = True,
    reverse_flow: bool = False,
) -> jax.Array:
  """Sampling from a batch using a reflow model.

  Here the batch is expected to have the following keys:
    - x_0: The initial state of the ODE.
    - channel:mean: The mean of the noise added to the initial state.
    - channel:std: The standard deviation of the noise added to the initial
      state.
    - output_mean: The mean of the output distribution.
    - output_std: The standard deviation of the output distribution.

  The output is expected to have the same shape as the input given by
  batch["x_0"].

  Args:
    batch: The batch to sample from.
    model: The model that encapsulates the flow model.
    trained_state: The trained state of the model.
    num_sampling_steps: The number of sampling steps for solving the ODE.
    time_chunk_size: The size of the time chunks in the batch.
    time_to_channel: Whether to move the time dimension to the channel
      dimension. Or just another dimension between the batch and the spatial
      dimensions.
    reverse_flow: Whether to use the reverse flow, i.e., integrate from 1 to 0,
      instead of 0 to 1.

  Returns:
    The sampled output.
  """
  # Checks the keys in the batch.
  if "channel:mean" not in batch or "channel:std" not in batch:
    raise ValueError("Batch does not contain the channel:mean or channel:std.")
  if "output_mean" not in batch or "output_std" not in batch:
    raise ValueError("Batch does not contain the output_mean or output_std.")
  if "x_0" not in batch:
    raise ValueError("Batch does not contain the x_0 field.")

  # Setting up the conditional ODE for the sampling.
  cond = {
      "channel:mean": move_time_to_channel(
          batch["channel:mean"], time_chunk_size, time_to_channel
      ),
      "channel:std": move_time_to_channel(
          batch["channel:std"], time_chunk_size, time_to_channel
      ),
  }

  latent_dynamics_fn = ode_solvers.nn_module_to_dynamics(
      model.flow_model,
      autonomous=False,
      cond=cond,  # to be added here.
      is_training=False,
  )

  # If we want to use the reverse flow, we need to change the sign of the
  # dynamics function and the evaluation time.
  if reverse_flow:
    latent_dynamics_fn = (
        lambda x, t, params: -latent_dynamics_fn(x, 1.0 - t, params),
    )

  integrator = ode_solvers.RungeKutta4()
  integrate_fn = functools.partial(
      integrator,
      latent_dynamics_fn,
      tspan=jnp.arange(0.0, 1.0, 1.0 / num_sampling_steps),
      params=trained_state.model_variables,
  )

  # Running the integration. Then take the last state.
  out = integrate_fn(
      move_time_to_channel(batch["x_0"], time_chunk_size, time_to_channel)
  )[-1, :]

  # Denormalize the output according to output (ERA5) Climatology.
  out = out * move_time_to_channel(
      batch["output_std"], time_chunk_size, time_to_channel
  ) + move_time_to_channel(
      batch["output_mean"], time_chunk_size, time_to_channel
  )

  # Move the channel dimension to the time dimension.
  out = move_channel_to_time(out, time_chunk_size, time_to_channel)

  return out
