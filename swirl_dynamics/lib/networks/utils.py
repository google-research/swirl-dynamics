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

"""Utilities for handling parameters.

This file contains three utilities that are useful for transforming parameters
from trees to vectors and vice versa. They are thoroughly used for
hypernetworks.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = Any
PyTreeDef = jax.tree_util.PyTreeDef
PyTree = Any


def flatten_params(
    params: PyTree, from_axis: int = 0
) -> tuple[Array, list[Array], PyTreeDef]:
  """Function to flatten the dictionary containing the parameters.

  Args:
    params: pytree containing the parameters of the network.
    from_axis: axis from which the flatten is applied. All dimensions before
      axis, will be kept. Thus, the shape of the leaves of the PyTree need to be
      the same.

  Returns:
    Tuple containing:
    i) Array with all the data in the leaves of the Pytree concatenated, where
       we preserve the original dimensions, and we flatten everything after
       from_axis. By default (from_axis=0) everything is flattened to a vector.
    ii) List of arrays with the shapes of the leaves in the Pytree.
    iii) PyTreeDef, containing the names and labes of the nodes in the Pytree.
  """
  flat_params, tree_def = jax.tree_util.tree_flatten(params)
  shapes = [p.shape for p in flat_params]

  flattened = []

  for p in flat_params:
    d = np.int64(p.size // (np.prod(p.shape[:from_axis])))
    flattened.append(jnp.reshape(p, p.shape[:from_axis] + (d,)))

  return jnp.concatenate(flattened, axis=-1), shapes, tree_def


def unflatten_params(
    flattened: Array, shapes: list[Array], tree_def: PyTreeDef
) -> PyTree:
  """Function to unflatten the parameters contained in an Array to a Dict.

  Args:
    flattened: array containing the parameters of the network.
    shapes: shapes of the different Arrays within the dictionary.
    tree_def: the definition of the fields within the dictionary (PyTreeDef).

  Returns:
    a Pytree containing the information in the input array, following the
    dimensions of each leaf contained in shapes, and following the labels of
    nodes of the PyTreecontained in tree_def. This function is the "inverse" of
    flatten_params, and it is used to recover the original Pytree from
    its internal representation and data.
  """
  sections = np.cumsum([np.prod(s) for s in shapes])
  segments = jnp.split(flattened, sections)[:-1]
  flat_params = [x.reshape(shape) for x, shape in zip(segments, shapes)]

  return jax.tree_util.tree_unflatten(tree_def, flat_params)


def flat_dim(params: PyTree) -> int:
  """Computing the total number of scalar elements in a PyTree.

  Args:
    params: PyTree containing the parameters/scalar values.

  Returns:
    Total number of scalars within all the leaves of the PyTree.
  """
  flat_params, _ = jax.tree_util.tree_flatten(params)
  return sum([p.size for p in flat_params])


def vmean(xs: PyTree, axis: int = 0) -> PyTree:
  """Average a vmapped tree along specified axis."""
  return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=axis), xs)
