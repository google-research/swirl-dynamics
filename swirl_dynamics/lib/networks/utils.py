# Copyright 2026 The swirl_dynamics Authors.
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

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
PyTreeDef = jax.tree_util.PyTreeDef
PyTree = Any


def flatten_params(
    params: PyTree, from_axis: int = 0
) -> tuple[Array, list[tuple[int, ...]], PyTreeDef]:
  """Function to flatten the dictionary containing the parameters.

  Args:
    params: pytree containing the parameters of the network.
    from_axis: non-negative axis from which the flatten is applied. All
      dimensions before `from_axis` are preserved across leaves.

  Returns:
    Tuple containing:
    i) Array with all the data in the leaves of the Pytree concatenated, where
       we preserve the leading batch dimensions before `from_axis` and flatten
       everything from `from_axis` onward.
    ii) List of shape tuples of the leaves in the Pytree.
    iii) PyTreeDef containing the names and labels of the nodes in the Pytree.
  """
  if from_axis < 0:
    raise ValueError(f"`from_axis` must be non-negative, got {from_axis}.")

  flat_params, tree_def = jax.tree_util.tree_flatten(params)
  shapes = [p.shape for p in flat_params]

  flattened = []

  for p in flat_params:
    d = np.prod(p.shape[from_axis:], dtype=int)
    flattened.append(jnp.reshape(p, p.shape[:from_axis] + (d,)))

  return jnp.concatenate(flattened, axis=-1), shapes, tree_def


def unflatten_params(
    flattened: Array, shapes: Sequence[Sequence[int]], tree_def: PyTreeDef
) -> PyTree:
  """Function to unflatten the parameters contained in an Array to a Dict.

  Args:
    flattened: array containing the parameters of the network, with any
      preserved batch dimensions preceding the final parameter axis.
    shapes: shapes of the different Arrays within the dictionary.
    tree_def: the definition of the fields within the dictionary (PyTreeDef).

  Returns:
    a Pytree containing the information in the input array, following the
    dimensions of each leaf contained in shapes, and following the labels of
    nodes of the PyTreecontained in tree_def. This function is the "inverse" of
    flatten_params, and it is used to recover the original Pytree from
    its internal representation and data.
  """
  from_axis = flattened.ndim - 1
  sections = np.cumsum([np.prod(s[from_axis:], dtype=int) for s in shapes])
  segments = jnp.split(flattened, sections, axis=-1)[:-1]
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
