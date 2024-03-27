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

"""Utility functions for the template."""

import collections
from collections.abc import Callable, Mapping, Sequence
import functools
import os
from typing import Any

from clu import values
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

Scalar = Any


def primary_process_only(cls: type[Any]) -> type[Any]:
  """Class decorator that modifies all methods to run on primary host only."""

  def wrap_method(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
      if jax.process_index() == 0:
        return method(self, *args, **kwargs)
      else:
        return None

    return wrapper

  for attr_name, attr_value in cls.__dict__.items():
    if callable(attr_value) and not attr_name.startswith("__"):
      setattr(cls, attr_name, wrap_method(attr_value))

  return cls


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


def create_slice(
    start: int | None = None, end: int | None = None, step: int | None = None
) -> slice:
  """Wraps the python `slice` to allow keyword arguments (for gin config)."""
  return slice(start, end, step)
