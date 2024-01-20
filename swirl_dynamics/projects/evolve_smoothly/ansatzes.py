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

"""Wrappers for ansatz modules that provide standardized interfaces."""

import dataclasses
from typing import ClassVar

from flax import linen as nn
import jax
import numpy as np
from swirl_dynamics.lib.networks import nonlinear_fourier
from swirl_dynamics.lib.networks import utils

Array = jax.Array


@dataclasses.dataclass
class Ansatz:
  """Base Ansatz class.

  This class wraps around an `nn.Module` object and provides interfaces for
  easily querying some static properties related to the parameters of the Module
  (e.g. shapes, PyTree definition). It also provides an interface for flexibly
  evaluating the wrapped model using PyTree or flattened parameters.

  Subclasses should implement/overwrite how to initialize the required
  attributes (those with `init=False`) given a Module instance, via
  `._get_params_meta()` and `._get_num_params_in_layers()`.

  Attributes:
    input_dim: the input dimension of the ansatz.
    output_dim: the output dimension of the ansatz.
    model: the ansatz model instance.
    num_params: the total number of model parameters.
    param_shapes: the tensor shapes of the parameters.
    param_treedef: the PyTree definition of the ansatz parameters.
    num_params_in_layers: the number of parameters in all layers. For example,
      `num_params_in_layers = (1, 2, 3)` means the model has 3 layers with 1, 2
      and 3 parameters respectively (sum to `num_params`). The order should also
      be consistent with the structure of the flattened params (i.e.
      `flattened[0:1]`, `flattened[1:3]`, `flattened[3:6]` should give the layer
      parameters respectively).
  """

  input_dim: ClassVar[int]
  output_dim: ClassVar[int]

  model: nn.Module
  num_params: int = dataclasses.field(init=False)
  param_shapes: list[tuple[int, ...]] = dataclasses.field(init=False)
  param_treedef: jax.tree_util.PyTreeDef = dataclasses.field(init=False)
  num_params_in_layers: tuple[int, ...] = dataclasses.field(init=False)

  def __post_init__(self):
    self.num_params, self.param_shapes, self.param_treedef = (
        self._get_params_meta()
    )
    self.num_params_in_layers = self._get_num_params_in_layers()

  def batch_evaluate(
      self, params: nn.module.FrozenVariableDict | Array, x: Array
  ) -> Array:
    """Evaluates the ansatz on a batch of collocation points."""
    if isinstance(params, Array):
      params = self.unflatten_to_pytree(params)
    return jax.vmap(self.model.apply, in_axes=(None, 0))(params, x)

  def unflatten_to_pytree(self, params: Array) -> nn.module.FrozenVariableDict:
    """Converts flattened params to PyTree format."""
    return utils.unflatten_params(params, self.param_shapes, self.param_treedef)

  def _get_num_params_in_layers(self) -> tuple[int, ...]:
    """Computes the number of parameters in every layer."""
    raise NotImplementedError

  def _get_params_meta(
      self,
  ) -> tuple[int, list[tuple[int, ...]], jax.tree_util.PyTreeDef]:
    """Computes the length, shapes and treedef of the ansatz params."""
    sample_input = np.ones((1, self.input_dim))
    # `sample_params` is only used to extract static information, so it's fine
    # to init deterministically
    sample_params = self.model.init(jax.random.PRNGKey(0), sample_input)
    _, shapes, tree_def = utils.flatten_params(sample_params)
    num_params = utils.flat_dim(sample_params)
    return num_params, shapes, tree_def


@dataclasses.dataclass
class NonLinearFourier(Ansatz):
  """The Nonlinear Fourier ansatz (NFA)."""

  input_dim: ClassVar[int] = 1
  output_dim: ClassVar[int] = 1
  model: nonlinear_fourier.NonLinearFourier

  def _get_num_params_in_layers(self) -> tuple[int, ...]:
    """Computes the number of parameters in the NFA layers.

    Flattened NFA parameters has the following order

      [bias_0, weights_0, ..., bias_N, weights_N, phases]

    Each (bias, weights) pair (from the MLP) constitutes one layer. The phases
    of the sin terms all together constitute another. Refer to
    `nonlinear_fourier.NonLinearFourier` for the computation logic.

    Returns:
      Number of parameters in MLP layers + phases.
    """
    i = 0
    shapes = []
    while i < len(self.param_shapes):
      if i < len(self.param_shapes) - 1:
        shapes.append(
            np.prod(self.param_shapes[i]) + np.prod(self.param_shapes[i + 1])
        )
        i += 2
      else:
        shapes.append(np.prod(self.param_shapes[i]))
        break
    return tuple(shapes)
