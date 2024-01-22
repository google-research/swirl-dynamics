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

# coding=utf-8
"""Nonlinear Fourier Networks.

Implementation of the ansatz used in [1], in which coefficients of a low-order
Fourier series are modulated by a periodic MLP.

References:
[1] Z. Y. Wan, L. Zepeda-Núñez, A. Boral and F. Sha, Evolve Smoothly, Fit
Consistently: Learning Smooth Latent Dynamics For Advection-Dominated Systems,
accepted at ICLR2023. "https://openreview.net/forum?id=Z4s73sJYQM"
"""

from collections.abc import Callable
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

Scalar = Any
Array = Any
ModuleDef = Any
Dtype = jnp.dtype


class MLP(nn.Module):
  """Simple multi-layer perceptron.

  Attributes:
    features: The size of the layers in the MLP.
    act: The activation function for the MLP.
    dtype: The type of inputs expected, by default is fp32. This is also is the
      same type of the parameters.
    layer_norm: Boolean indicating if layer norms are used after each dense
      linear layer.
  """

  features: tuple[int, ...]
  act_fn: Callable[[Array], Array] = nn.relu
  dtype: Dtype = jnp.float32
  layer_norm: bool = False
  use_bias: bool = True

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    # Forcing the inputs to be of the correct type.
    x = jnp.asarray(inputs, self.dtype)

    # Applying different layers followed by an activation function.
    for feat in self.features[:-1]:
      x = self.act_fn(
          nn.Dense(feat, use_bias=self.use_bias, param_dtype=self.dtype)(x)
      )
      if self.layer_norm:
        x = nn.LayerNorm()(x)

    # The last layer is a linear layer.
    x = nn.Dense(
        self.features[-1], use_bias=self.use_bias, param_dtype=self.dtype
    )(x)
    return x


class NonLinearFourier(nn.Module):
  r"""Multi-layer perceptron with periodic features.

  This function implements Eq. (10) in [1].
  We consider an ansatz of the form:

  Ψ(x) = ∑ ϕₖ(x) sin(ωₖ x + aₖ) + ϕ₋ₖ(x) cos(ωₖ x + a₋ₖ). (1)

  In this case {ϕₖ(x)}_{k=-num_freqs}^{k=num_freqs} are the output of an
  MLP fed with oscillatory and periodic features given by the shifted Fourier
  basis, i.e.,:

  {sin(ωₖ x + aₖ), cos(ωₖ x + a₋ₖ)}_{k=1}^num_freqs. (2)

  Attributes:
    features: the size of the layers in the MLP.
    num_freqs: number of frequencies.
    act_fn: activation function for the layers in the MLP.
    dyadic: boolean for dyadic expansion of the frequencies.
    zero_freq: boolean to add a constant accounting for the zero frequency.
    train_freqs: boolean to indicate if the frequencies are trainable.
    dtype: type of the inputs and the parameters.
  """

  features: tuple[int, ...]
  num_freqs: int = 3
  act_fn: Callable[[Array], Array] = nn.relu
  dyadic: bool = False
  zero_freq: bool = False
  train_freqs: bool = False
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Evaluation of the Periodic Multi-Layer Perceptron.

    Args:
      inputs: Array containing a scalar for the evaluation. This will be wrapped
        in a vmap for vectorization.

    Returns:
      The evaluation of the network at a given point.
    """

    features = self.features + (2 * self.num_freqs,)
    x_i = jnp.asarray(inputs, self.dtype)

    # Phases for the sinusoidal functions.
    a = self.param(
        'phases', nn.initializers.zeros, (2 * self.num_freqs,), self.dtype
    ).reshape((2, -1))

    if self.dyadic:
      # Dyadic partition of the frequencies [1, 2, 4, 8,... ].
      freq = jnp.power(
          2,
          jnp.linspace(0, self.num_freqs - 1, self.num_freqs, dtype=self.dtype),
      )
    else:
      # Frequencies in a equispace grid [1, 2, 3, 4 ...].
      freq = jnp.linspace(1, self.num_freqs, self.num_freqs, dtype=self.dtype)
    omega_init = jnp.pi * jnp.repeat(freq, 2).reshape((-1, 2)).T

    if self.train_freqs:
      # If frequencies are trainable, define initializer.
      omega_init_fn = jax.nn.initializers.constant(omega_init)
      omega = self.param(
          'frequencies', omega_init_fn, (2, self.num_freqs), self.dtype
      )
    else:
      omega = omega_init

    # Building the shifted Fourier basis in (2).
    y = omega * (x_i + a)
    fourier_basis = jnp.concatenate(
        [jnp.sin(y[0, :]), jnp.cos(y[1, :])], axis=-1
    )

    phi = MLP(features=features, act_fn=self.act_fn, dtype=self.dtype)(
        fourier_basis
    )

    # Output following (1).
    psi = jnp.sum(phi * fourier_basis, axis=-1, keepdims=True)

    if self.zero_freq:
      phi_0 = self.param('const', nn.initializers.zeros, (1,), self.dtype)
      psi = psi + phi_0

    return psi


class NonLinearFourier2D(nn.Module):
  r"""Multi-layer perceptron with periodic featuress.

  This model is the 2D version of NonLinearFourier model, in which the features,
  and basis function are the tensorized version of the ones in the 1D model.
  We consider the features given by the tensor product of Fourier
  exponentials,

  ψ(x,y)ⱼₖ = exp(i (ωⱼ x + ωₖ y) + aⱼₖ), (3)

  in particular we consider that we have the real and imaginary part using
  different channels.  These periodic features are then fed to a vanilla
  MLP, whose output is used to weight the periodic features.
  Namely

  Ψ(x,y) = ∑ Re( ϕ(x,y)ⱼₖ ⋅ ψ(x,y)ⱼₖ), (4)

  where each of the Fourier coefficients ϕ, depend on the periodic features
  given by:

  { Re( ψ(x,y)ⱼₖ ) }_{j,k}. (5)

  Attributes:
    features: The size of the layers in the MLP.
    num_freqs: number of frequencies.
    act_fn: activation function for the layers in the MLP.
    dyadic: boolean for dyadic expansion of the frequencies.
    zero_freq: boolean to add a constant accounting for the zero frequency.
  """

  features: tuple[int, ...]
  num_freqs: int = 3
  act_fn: Callable[[Array], Array] = nn.relu
  dyadic: bool = False
  zero_freq: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Evaluation of the Periodic Multi-Layer Perceptron in 2D.

    Args:
      inputs: Array containing a two-dimensional vecrot for the evaluation. This
        will be wrapped in a vmap for vectorization.

    Returns:
      The evaluation of the network at a given point.
    """

    features = self.features + ((2 * self.num_freqs + 1) ** 2,)

    assert inputs.shape == (2,), 'The shape of the inputs should be (2,).'
    x_i = inputs.astype(self.dtype)

    # shape : (2, num_freqs, 2).
    a = self.param(
        'phases', nn.initializers.zeros, (2 * 2 * self.num_freqs,), self.dtype
    ).reshape((2, self.num_freqs, 2))

    if self.dyadic:
      # Dyadic partition of the frequencies [1, 2, 4, 8,... ].
      # shape : (1, num_freqs, 2).
      omega = jnp.pi * jnp.repeat(
          2
          ** jnp.linspace(
              0, self.num_freqs - 1, self.num_freqs, dtype=self.dtype
          ),
          2,
      ).reshape((1, self.num_freqs, 2))
    else:
      # shape : (1, num_freqs, 2).
      omega = jnp.pi * jnp.repeat(
          jnp.linspace(1, self.num_freqs, self.num_freqs, dtype=self.dtype), 2
      ).reshape((1, self.num_freqs, 2))

    # shape : (2, num_freqs, 2) for sin-cos, \omega, and x-y.
    y = omega * (x_i.reshape((1, 1, 2)) + a)

    # TODO: create a funcion that creates the periodic features.
    # Applying the trigonometric functions, which can be written as:
    # [[1,           1],
    #  [sin(ω₁ x), sin(ω₁ y)],
    #  [sin(ω₂ x), sin(ω₂ y)],
    #       ⋮         ⋮
    #  [sin(ωₙ x), sin(ωₙ y)],
    #  [cos(ω₁ x), cos(ω₁ y)],
    #  [cos(ω₂ x), cos(ω₂ y)],
    #       ⋮         ⋮
    #  [cos(ωₙ x), cos(ωₙ y)]].
    # where n is equal to the number of frequencies.
    # shape :  (2*num_freqs + 1, 2).
    fourier_1d = jnp.concatenate(
        [jnp.ones((1, 2)), jnp.sin(y[0, :, :]), jnp.cos(y[1, :, :])], axis=0
    )

    # Multiplication of (∑ ( sin(ωᵢ x) + cos(ωᵢ x) )) times
    #                   (∑ ( sin(ωⱼ y) + cos(ωⱼ y) )).
    # To create the Fourier basis in Eq.(5).
    fourier_basis = jnp.einsum('i,j-> ij', fourier_1d[:, 0], fourier_1d[:, 1])
    fourier_basis = fourier_basis.reshape(((2 * self.num_freqs + 1) ** 2,))

    # shape : (2 * num_freqs + 1, 2).
    phi = MLP(features=features, act_fn=self.act_fn, dtype=self.dtype)(
        fourier_basis
    )

    # shape : (1,), this is Eq. (4).
    psi = jnp.sum(phi * fourier_basis, axis=-1, keepdims=True)

    if self.zero_freq:
      a_0 = self.param('const', nn.initializers.zeros, (1,))
      psi = psi + a_0

    return psi
