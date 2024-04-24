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

"""Axial attention modules."""

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


class AddAxialPositionEmbedding(nn.Module):
  """Adds trainable axial position embeddings to the inputs."""

  position_axis: int
  initializer: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    pos_axis = self.position_axis
    pos_axis = pos_axis if pos_axis >= 0 else pos_axis + inputs.ndim

    if not 0 <= pos_axis < inputs.ndim:
      raise ValueError(
          f"Invalid position ({self.position_axis}) or feature axis"
          f" ({self.feature_axis})!"
      )

    feat_axis = inputs.ndim - 1
    if pos_axis == feat_axis:
      raise ValueError(
          f"Position axis ({self.position_axis}) must not coincide with feature"
          f" axis ({feat_axis})!"
      )

    unsqueeze_axes = tuple(set(range(inputs.ndim)) - {pos_axis, feat_axis})
    embedding = self.param(
        "embedding",
        self.initializer,
        (inputs.shape[pos_axis], inputs.shape[feat_axis]),
    )
    return inputs + jnp.expand_dims(embedding, axis=unsqueeze_axes)


class AxialSelfAttention(nn.Module):
  """Axial self-attention for multidimensional inputs."""

  num_heads: int
  attention_axis: int = -2
  kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
  deterministic: bool = True
  precision: PrecisionLike = None
  normalize_qk: bool = False
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies axial self-attention to the inputs."""

    if self.attention_axis == -1 or self.attention_axis == inputs.ndim - 1:
      raise ValueError(
          f"Attention axis ({self.attention_axis}) cannot be the last axis,"
          " which is treated as the features!"
      )

    inputs = jnp.swapaxes(inputs, self.attention_axis, -2)
    inputs_q = jnp.reshape(inputs, (-1, *inputs.shape[-2:]))

    out = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.kernel_init,
        deterministic=self.deterministic,
        normalize_qk=self.normalize_qk,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
    )(inputs_q=inputs_q)

    out = jnp.reshape(out, inputs.shape)
    out = jnp.swapaxes(out, -2, self.attention_axis)

    return out
