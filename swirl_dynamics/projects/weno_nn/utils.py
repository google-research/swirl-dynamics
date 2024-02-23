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

"""Several simple utilities."""

from collections.abc import Callable
from typing import Any, Literal

import flax.linen as nn
import jax
from swirl_dynamics.lib.networks import rational_networks
from swirl_dynamics.projects.weno_nn import weno_nn


PyTree = Any
Array = jax.Array


def flat_dim(params: PyTree) -> int:
  """Computes the total number of scalar elements in a `PyTree`.

  Args:
    params: PyTree containing the parameters/scalar values.

  Returns:
    Total number of scalars within all the leaves of the PyTree.
  """
  flat_params, _ = jax.tree_util.tree_flatten(params)
  return sum([p.size for p in flat_params])


def get_feature_func(
    func_name: Literal[
        'z_layer', 'rational', 'rational_descentered', 'delta_layer'
    ]
) -> Callable[[Array], Array] | None:
  """Returns the feature function for the given function name.

  Args:
    func_name: Name of the function.

  Returns:
    The feature function for the given function name.
  """

  if func_name == 'z_layer':
    func = weno_nn.weno_z_layer
  elif func_name == 'rational':
    func = weno_nn.FeaturesRationalLayer()
  elif func_name == 'rational_descentered':
    func = weno_nn.FeaturesRationalLayerDescentered()
  elif func_name == 'delta_layer':
    func = weno_nn.delta_layer
  else:
    func = None

  return func


def get_act_func(
    func_name: Literal[
        'relu',
        'gelu',
        'selu',
        'rational',
        'rational_unshared',
        'swish',
        'rational_act_fun',
        'unshared_rational_act_fun',
        'GeGLU',
    ]
) -> Callable[[Array], Array] | str | None:
  """Returns the activation function for the given function name.

  Args:
    func_name: Name of the function.

  Returns:
    The activation function for the given function name, or the string for
    defining the model inside the OmegaNN models.
  """

  if func_name == 'relu':
    func = nn.relu
  elif func_name == 'gelu':
    func = nn.gelu
  elif func_name == 'selu':
    func = nn.selu
  elif func_name == 'rational':
    func = rational_networks.RationalLayer()
  elif func_name == 'rational_unshared':
    func = rational_networks.UnsharedRationalLayer()
  elif func_name == 'swish':
    func = nn.swish
  elif func_name == 'rational_act_fun':
    func = 'rational_act_fun'
  elif func_name == 'unshared_rational_act_fun':
    func = 'unshared_rational_act_fun'
  elif func_name == 'GeGLU':
    func = 'GeGLU'
  else:
    func = None

  return func
