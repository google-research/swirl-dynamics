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

"""Hyper-Unet for learning the latent space dynamics in [1].

Implementation of the Hyper Unet in Appendix A.3 in [1]. This is used as an
ansatz to learn the underlying latent-space dynamics. For a diagram of the
network see Fig. 2, right in [1].

References:
[1] Z. Y. Wan, L. Zepeda-Núñez, A. Boral and F.Sha, Evolve Smoothly, Fit
Consistently: Learning Smooth Latent Dynamics For Advection-Dominated Systems,
submitted to ICLR2023. "https://openreview.net/forum?id=Z4s73sJYQM"
"""

from collections.abc import Callable
from typing import Any

from flax import linen as nn
import jax.numpy as jnp
import numpy as np

Scalar = Any
Array = Any
Dtype = jnp.dtype


class HyperUnet(nn.Module):
  """Unet that maps NN weights to the same dimension.

  Attributes:
    flat_layer_shapes: The dimension of each of layers, in which the weights and
      biases are flattened.
    embed_dims: # (weight, layer, global) The embedding dimensions for each
      level from finest to coarsest. In a nutshell, at each level we compress
      the information to the given embedding dimension, and it is processed in
      compressed form to decrease the number of parameters.
    act_fn: Activation function.
    use_layernorm: Boolean for using layer normalization. This helps to make the
      time evolution more stable.
    dtype: Data type for the input and parameters.
  """

  flat_layer_shapes: tuple[int, ...]
  embed_dims: tuple[int, int, int]
  act_fn: Callable[[Array], Array] = nn.swish
  use_layernorm: bool = False
  dtype: Dtype = jnp.float32

  def slice_inputs(self, inputs: Array, axis: int = -1) -> list[Array]:
    """Slice into layers based on weight shapes.

    We make the assumption that the layer weights are in contiguous chunks.

    Args:
      inputs: The weights flattened.
      axis: The axis on which the slicing happens.

    Returns:
      The sliced inputs along the specified axis.
    """
    sections = np.cumsum(self.flat_layer_shapes)
    return jnp.split(inputs, sections, axis)[:-1]

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    weights_emb_dims, layer_emb_dims, global_emb_dims = self.embed_dims
    # Adding an extra dimension for the convolutional layers.
    # shape : batch_shape[:-1] + (nw, 1)
    inputs = jnp.expand_dims(inputs, axis=-1)

    inputs0 = nn.ConvLocal(
        features=weights_emb_dims,
        kernel_size=(1,),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
    )(inputs)
    # Each "layer" is basically flattened (W, b) where W and b are weights and
    # biases for the same layer.
    # shape: [(..., np, weights_emb_dims)].
    w_layer = self.slice_inputs(inputs0, axis=-2)
    s_list = []
    for i, s in enumerate(w_layer):
      # Downsampling with respect to the layer.
      s_list.append(
          nn.DenseGeneral(
              features=layer_emb_dims,
              axis=(-2, -1),
              use_bias=False,
              dtype=self.dtype,
              param_dtype=self.dtype,
              name=f'w2l_down_{i}',
          )(s)
      )
    # shape : (..., nlayer, layer_emb_dims)
    s = jnp.stack(s_list, axis=-2)

    # Mixing the information within each layer.
    ds = nn.ConvLocal(
        features=layer_emb_dims,
        kernel_size=(1,),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='layer_down_mix1',
    )(s)
    ds = self.act_fn(ds)
    ds = nn.ConvLocal(
        features=layer_emb_dims,
        kernel_size=(1,),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='layer_down_mix2',
    )(ds)
    # Using a skip connection.
    # shape : (..., nlayer, layer_emb_dims)
    s += ds
    if self.use_layernorm:
      s = nn.LayerNorm()(s)

    # Downsampling to the coarsest (i.e., global) level.
    # shape : (..., global_emb_dims)
    h = nn.DenseGeneral(
        features=global_emb_dims,
        axis=(-2, -1),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='l2g_down',
    )(s)

    # Mixing/processing information a the coarsest level.
    dh = nn.DenseGeneral(
        features=global_emb_dims,
        axis=-1,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='glob_mix1',
    )(h)
    dh = self.act_fn(dh)
    dh = nn.DenseGeneral(
        features=global_emb_dims,
        axis=-1,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='glob_mix2',
    )(dh)
    # shape : (..., global_emb_dims)
    h2 = h + dh

    if self.use_layernorm:
      h2 = nn.LayerNorm()(h2)

    # Upsampling to the Layer level, and concatenate with the downsampled data.
    # shape :  (..., nlayer, layer_emb_dims)
    s2 = nn.DenseGeneral(
        features=s.shape[-2:],
        axis=-1,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='g2l_up',
    )(h2)
    # shape : (..., nlayer, 2*layer_emb_dims)
    s2 = jnp.concatenate([s, s2], axis=-1)

    # Mixing within layer.
    ds2 = nn.ConvLocal(
        features=s2.shape[-1],
        kernel_size=(1,),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='layer_up_mix1',
    )(s2)
    ds2 = self.act_fn(ds2)
    ds2 = nn.ConvLocal(
        features=s2.shape[-1],
        kernel_size=(1,),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='layer_up_mix2',
    )(ds2)
    # shape : (..., nlayer, 2*layer_emb_dims)
    s2 += ds2

    if self.use_layernorm:
      s2 = nn.LayerNorm()(s2)

    w2_list = []
    for i in range(s2.shape[-2]):
      # Upsampling to the individual weights level
      # shape : (..., np_i, weights_emb_dims)
      w2 = nn.DenseGeneral(
          features=w_layer[i].shape[-2:],
          axis=-1,
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.dtype,
          name=f'l2w_up_{i}',
      )(s2[..., i, :])
      # shape : (..., np_i, 2*weights_emb_dims)
      w2 = jnp.concatenate([w_layer[i], w2], axis=-1)
      w2_list.append(w2)

    # shape : (..., nw, 2*weights_emb_dims)
    w2 = jnp.concatenate(w2_list, axis=-2)
    # Mixing at the weights level.
    dw2 = nn.ConvLocal(
        features=w2.shape[-1],
        kernel_size=(1,),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='weight_mix1',
    )(w2)
    dw2 = self.act_fn(dw2)
    dw2 = nn.ConvLocal(
        features=w2.shape[-1],
        kernel_size=(1,),
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.dtype,
        name='weight_mix2',
    )(dw2)
    w2 += dw2
    if self.use_layernorm:
      w2 = nn.LayerNorm()(w2)

    out = nn.ConvLocal(
        features=1,
        kernel_size=(1,),
        name='final_proj',
        dtype=self.dtype,
        param_dtype=self.dtype,
    )(w2)

    return jnp.squeeze(out, axis=-1)
