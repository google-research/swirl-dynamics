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

"""WENO NN scheme written in Jax.

We implement the Neural Network for computing interpolation weights in [1].

Refs:

[1] D. Bezgin, S.J. Schmidt and N.A. Klaus, "WENO3-NN: A maximum-order
three-point data-driven weighted essentially non-oscillatory scheme",
Journal of Computational Physics, Volume 452, Issue C, Mar 2022.

[2] M. Castro, B. Costa, and W. S. Don, "High order weighted essentially
non-oscillatory WENO-Z schemes for hyperbolic conservation laws", Journal of
Computational Physics, Volume 230, 2011.
"""

import functools
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib.networks import rational_networks

Array = jax.Array
PyTree = Any


def _delta_layer(u_bar: Array) -> tuple[Array, Array, Array, Array]:
  """Helper function computing an unnormalized delta layer.

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).

  Returns:
    A tuple containing the absolute value of the first and second order
    finite-differences of u_bar.
  """

  delta_1 = jnp.abs(u_bar[1] - u_bar[0])
  delta_2 = jnp.abs(u_bar[2] - u_bar[1])
  delta_3 = jnp.abs(u_bar[2] - u_bar[0])
  delta_4 = jnp.abs(u_bar[2] - 2 * u_bar[1] + u_bar[0])

  return (delta_1, delta_2, delta_3, delta_4)


def _delta_layer_weno5(u_bar: Array) -> tuple[Array, ...]:
  """Helper function computing an unnormalized delta layer for WENO5.

  Args:
    u_bar: Array containing (u_{n-2}, u_{n-1}, u_n, u_{n+1}, u_{n+2}).

  Returns:
    A tuple containing the absolute value of the first and second order
    finite-differences of u_bar with difference stencils.
  """

  delta_1 = jnp.abs(u_bar[0] - 2.0 * u_bar[1]+ u_bar[2])
  delta_2 = jnp.abs(u_bar[0] - 4.0 * u_bar[1]+ 3 * u_bar[2])
  delta_3 = jnp.abs(u_bar[1] - 2.0 * u_bar[2]+ u_bar[3])
  delta_4 = jnp.abs(u_bar[1] - u_bar[3])
  delta_5 = 1./12.*jnp.abs(- u_bar[0] + 8.0 * u_bar[1] - 30. * u_bar[2]
                           + 8.0 * u_bar[3] - 1.0 * u_bar[4])
  delta_6 = jnp.abs(u_bar[2] - 2.0 * u_bar[3]+ u_bar[4])
  delta_7 = jnp.abs(3 * u_bar[2] - 4.0 * u_bar[3] + u_bar[4])

  return (delta_1, delta_2, delta_3, delta_4, delta_5, delta_6, delta_7)


def delta_layer(
    u_bar: Array,
    global_norm: jnp.float64 | None = None,
    eps: jnp.float64 = 1e-15,
) -> Array:
  """Implementation of Delta layer that outputs the features of the network.

  Implementation of Eqs. (14) and (15) in [1].

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    global_norm: Possible global normalization instead of using the input
      dependent normalization in Eq. (15) of [1].
    eps: Small number to avoid divisions by zero, following [1].

  Returns:
    Absolute value of the first and second order finite-differences of u_bar,
    normalized with the maximum absolute value between the forward and backward
    first order finite differences.
  """

  (delta_1, delta_2, delta_3, delta_4) = _delta_layer(u_bar)

  # Initializing a global normalization constant if provided.
  if not global_norm:
    re_norm = jnp.clip(jnp.maximum(delta_1, delta_2), a_min=eps)
  else:
    re_norm = global_norm

  return jnp.stack([delta_1, delta_2, delta_3, delta_4]) / re_norm


def weno_z_layer(
    u_bar: Array,
    q: int = 2,
    eps: jnp.float64 = 1e-15,
) -> Array:
  """Delta layer mimimick the WENO-Z features.

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    q: Order of the rational function controlling the ratio between the
      smoothness indicators as in [2]
    eps: Small number to avoid divisions by zero, following [2].

  Returns:
    Four features based on fine differences of the input values with different
    weights.
  """

  (delta_1, delta_2, delta_3, delta_4) = _delta_layer(u_bar)

  # Initializing a global normalization constant if provided.
  alpha_1 = 1 + jnp.power(delta_4 / (delta_1 + eps), q)
  alpha_2 = 1 + jnp.power(delta_4 / (delta_2 + eps), q)
  alpha_3 = 1 + jnp.power(delta_4 / (delta_3 + eps), q)

  re_norm_4 = jnp.clip(jnp.maximum(delta_1, delta_2), a_min=eps)
  alpha_4 = 1 + jnp.power(delta_4 / (re_norm_4 + eps), q)

  norm = alpha_1 + alpha_2 + alpha_3 + alpha_4

  return jnp.stack([alpha_1, alpha_2, alpha_3, alpha_4]) / norm


class FeaturesRationalLayer(nn.Module):
  """Fully rational layer for the input features.

  Attributes:
    dtype: Data type for the rational network.
    cutoff: Shift for the thresholding.
  """

  dtype: jnp.dtype = jnp.float64
  cutoff: Optional[jnp.float64] = None

  @nn.compact
  def __call__(self, u_bar: Array) -> Array:
    """Application of the of rational feature layer.

    Args:
      u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).

    Returns:
      Four normalized features (between 0 and 1) for the input to the network.
    """

    (delta_1, delta_2, delta_3, delta_4) = _delta_layer(u_bar)

    delta = jnp.stack([delta_1, delta_2, delta_3, delta_4])
    output = rational_networks.UnsharedRationalLayer(
        dtype=self.dtype, cutoff=self.cutoff
    )(delta)

    norm = jnp.linalg.norm(output)
    if self.cutoff:
      norm = hard_thresholding(
          norm, threshold_value=self.cutoff, cutoff=self.cutoff
      )

    return output / norm


class FeaturesRationalLayerDescentered(nn.Module):
  """Implementation of rational layer with descentered stencils.

  Attributes:
    dtype: Data type for the rational network.
    cutoff: Shift for the thresholding.
  """

  dtype: jnp.dtype = jnp.float64
  cutoff: Optional[jnp.float64] = None

  @nn.compact
  def __call__(self, u_bar: Array) -> Array:
    """Application of the of rational feature layer.

    Args:
      u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).

    Returns:
      Four normalized features (between 0 and 1) for the input to the network.
    """

    (delta_1, delta_2, delta_3, delta_4) = _delta_layer(u_bar)

    delta_5 = jnp.abs(-0.5 * u_bar[2] + 2 * u_bar[1] - 1.5 * u_bar[0])
    delta_6 = jnp.abs(-1.5 * u_bar[2] + 2 * u_bar[1] - 0.5 * u_bar[0])

    delta = jnp.stack([delta_1, delta_2, delta_3, delta_4, delta_5, delta_6])

    output = rational_networks.UnsharedRationalLayer(dtype=self.dtype)(delta)

    norm = jnp.linalg.norm(output)
    if self.cutoff:
      norm = hard_thresholding(
          norm, threshold_value=self.cutoff, cutoff=self.cutoff
      )

    return output / norm


def hard_thresholding(
    x: jnp.float64,
    threshold_value: jnp.float64,
    cutoff: jnp.float64 = 2e-4,
) -> jnp.float64:
  """Simple implementation of hard thresholding in Eq. (16) of [1].

  Args:
    x: Number to be thresholded.
    threshold_value: Value used if x<cutoff
    cutoff: Shift for the thresholding.

  Returns:
    The input, which we assume is a scalar, hard-thresholded by cutoff. Namely
    x if x > cutoff, `threshold_value` otherwise.
  """
  return jax.lax.cond(x < cutoff, lambda x: threshold_value, lambda x: x, x)


def eno_layer(omega: Array, cutoff: jnp.float64 = 2e-4) -> Array:
  """Implementation of the ENO_layer that thresholds and normalizes the weights.

  Args:
    omega: Array with two elements containing the output of the network.
    cutoff: Cutoff for the hard thresholding.

  Returns:
    The hard thresholded and re-weighted version of omega.
  """
  omega_tilde = jax.vmap(hard_thresholding, in_axes=(0, None, None))(
      omega, 0.0, cutoff
  )
  norm_omega = jnp.sum(omega_tilde)
  omega_tilde = omega_tilde / norm_omega

  return omega_tilde


def gamma(u_bar: Array, epsilon_gamma: jnp.float64 = 1e-15) -> Array:
  """Computation of gamma in Eq. (22) of [1].

  Args:
    u_bar: Array containing (u_{n-1}, u_n, u_{n+1}).
    epsilon_gamma: Regularized to avoid division by zero.

  Returns:
    Estimators of the smoothness of the function using the ratio between
    approximations of the second and first derivatives.
  """
  return jnp.abs(u_bar[0] - 2 * u_bar[1] + u_bar[2]) / (
      jnp.abs(u_bar[1] - u_bar[0])
      + jnp.abs(u_bar[2] - u_bar[1])
      + epsilon_gamma
  )


class OmegaNN(nn.Module):
  """Layer for computing the weights of the interpolants.

  Attributes:
    features: Number of neurons for the hidden layers.
    order: Order of the weno scheme. Default to 3.
    features_fun: Function that computes the input features to the MLP based
      on the input to the network. Options are the delta layer in [1], and
      features created using rational networks.
    act_fun: Activation function for the hidden layers.
    act_fun_out: Activation function for the last (output) layer.
    dtype: Type of input/outputs and parameters.
    global_norm: If non-zero, it becomes a global normalization constant for the
      delta layers, instead of local normalization used by default.
    eno_layer_cutoff: Cutoff for the hard thresholding inside the ENO layer. The
      ENO layer should only be used during inference.
  """

  features: tuple[jnp.int64, ...]
  order: int = 3
  features_fun: Callable[[Array], Array] = functools.partial(
      delta_layer, global_norm=None, eps=1e-15
  )
  act_fun: Callable[[Array], Array] | str = nn.swish
  act_fun_out: Callable[[Array], Array] = nn.softmax
  dtype: jnp.dtype = jnp.float64
  global_norm: jnp.float64 | None = None
  eno_layer_cutoff: jnp.float64 = 2e-4

  @nn.compact
  def __call__(self, u_bar: Array, test: bool = False) -> Array:
    """Computation of the weights for the interpolation polynomials.

    Args:
      u_bar: the average of u a within the cells, [u_{i-1}, u_i, u_{i+1}].
      test: flag for change between training and testing. For the latter the ENO
        layer is active.

    Returns:
      The WENO_NN weights, which are the weights for interpolation.
    """
    delta = self.features_fun(u_bar)

    # Forcing the output to be consistent with the WENO order.
    features = self.features[:] + (self.order - 1,)

    for feats in features[:-1]:
      delta = nn.Dense(
          features=feats, param_dtype=self.dtype, dtype=self.dtype
      )(delta)

      # Apply the activation function.
      if self.act_fun == "rational_act_fun":
        delta = rational_networks.RationalLayer()(delta)
      elif self.act_fun == "unshared_rational_act_fun":
        delta = rational_networks.UnsharedRationalLayer()(delta)
      elif self.act_fun == "GeGLU":
        delta = nn.GeGLU()(delta)
      else:
        delta = self.act_fun(delta)

    # Following [1], the last layer has a different activation function.
    omega_out = self.act_fun_out(
        nn.Dense(
            features=features[-1], param_dtype=self.dtype, dtype=self.dtype
        )(delta)
    )

    # Using the ENO layer during inference.
    if test:
      omega_out = eno_layer(omega_out, self.eno_layer_cutoff)

    return omega_out
