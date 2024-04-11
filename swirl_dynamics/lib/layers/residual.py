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

"""Residual layer modules."""
import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
PrecisionLike = (
    None
    | str
    | jax.lax.Precision
    | tuple[str, str]
    | tuple[jax.lax.Precision, jax.lax.Precision]
)


class CombineResidualWithSkip(nn.Module):
  """Combine residual and skip connections.

  Attributes:
    project_skip: Whether to add a linear projection layer to the skip
      connections. Mandatory if the number of channels are different between
      skip and residual values.
  """

  project_skip: bool = False
  precision: PrecisionLike = None
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, *, residual: Array, skip: Array) -> Array:
    if self.project_skip:
      skip = nn.Dense(
          residual.shape[-1],
          kernel_init=nn.initializers.variance_scaling(
              scale=1.0, mode="fan_avg", distribution="uniform"
          ),
          precision=self.precision,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
      )(skip)
    return (skip + residual) / jnp.sqrt(2.0)
