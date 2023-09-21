# Copyright 2023 The swirl_dynamics Authors.
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

"""Utility functions for the template."""

import collections
from collections.abc import Mapping, Sequence
import os
from typing import Any

from clu import values
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

Scalar = Any


def load_scalars_from_tfevents(
    logdir: str,
) -> Mapping[int, Mapping[str, Scalar]]:
  """Loads scalar summaries from events in a logdir."""
  paths = tf.io.gfile.glob(os.path.join(logdir, "events.out.tfevents.*"))
  data = collections.defaultdict(dict)
  for path in paths:
    for event in tf.compat.v1.train.summary_iterator(path):
      for value in event.summary.value:
        data[event.step][value.tag] = jnp.asarray(
            tf.make_ndarray(value.tensor).flat[0]
        )

  return data


def is_scalar(value: Any) -> bool:
  """Checks if a given value is a scalar."""
  if isinstance(value, values.Scalar) or isinstance(
      value, (int, float, np.number)
  ):
    return True
  if isinstance(value, (np.ndarray, jnp.ndarray)):
    return value.ndim == 0 or value.size <= 1
  return False


def optax_chain(
    transformations: Sequence[optax.GradientTransformation],
) -> optax.GradientTransformation:
  """Wraps `optax.chain` to allow keyword arguments (for gin config)."""
  return optax.chain(*transformations)
