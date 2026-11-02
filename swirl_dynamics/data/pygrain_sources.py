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

"""Module for reusable PyGrain data sources."""

import operator
from typing import Any, SupportsIndex

import grain.python as pygrain
import numpy as np


def _read_only_view(array: Any) -> np.ndarray:
  """Returns a read-only view of `array` (which itself is left untouched)."""
  view = np.asarray(array).view()
  view.flags.writeable = False
  return view


def in_memory_read_options() -> pygrain.ReadOptions:
  """Returns the PyGrain read options for in-memory sources.

  Reading records from memory (e.g. from an `ArrayDataSource`) is cheap, so it
  is done in the main thread where supported: PyGrain's default pool of reader
  threads contends for the GIL and makes the data loading several times slower.
  """
  # Older Grain releases (e.g. grain-nightly 0.0.x) require reader threads and
  # read nothing with `prefetch_buffer_size=0`, so open-source builds keep the
  # default options.
  read_options = pygrain.ReadOptions()
  return read_options


class ArrayDataSource(pygrain.RandomAccessDataSource):
  """A random-access data source over named in-memory arrays.

  The arrays are indexed along their leading axis, which must have the same
  length for all of them: record `i` is `{name: array[i] for each array}`. A
  typical use is to wrap the trajectories read from an HDF5 file, e.g.

    snapshots, tspan = hdf5_utils.read_arrays_as_tuple(path, ("u", "t"))
    source = ArrayDataSource(u=snapshots, t=tspan)

  The source holds read-only views of the arrays, so the records (which are
  views as well, i.e. no data is copied) cannot be modified in place by
  downstream transformations, which would otherwise silently corrupt the
  in-memory data. The arrays passed to the constructor remain writable.
  """

  def __init__(self, **arrays: np.ndarray):
    """Initializes the data source.

    Args:
      **arrays: The arrays to serve, keyed by feature name. They must have at
        least one dimension and the same length along the leading axis (e.g. the
        number of trajectories).
    """
    if not arrays:
      raise ValueError("At least one array must be provided.")
    self._arrays = {
        name: _read_only_view(array) for name, array in arrays.items()
    }
    shapes = {name: array.shape for name, array in self._arrays.items()}
    if any(not shape for shape in shapes.values()):
      raise ValueError(f"All arrays must have at least one dimension: {shapes}")
    self._length = next(iter(shapes.values()))[0]
    if any(shape[0] != self._length for shape in shapes.values()):
      raise ValueError(
          f"All arrays must have the same leading dimension, got: {shapes}"
      )

  def __len__(self) -> int:
    """Returns the number of records in the data source."""
    return self._length

  def __getitem__(self, record_key: SupportsIndex) -> dict[str, np.ndarray]:
    """Returns the record at a given index, as a dictionary of arrays."""
    idx = operator.index(record_key)
    return {name: array[idx] for name, array in self._arrays.items()}

  def __repr__(self) -> str:
    # Deterministic, data and keyword-order independent: PyGrain uses it to
    # validate checkpointed iterator states.
    specs = ", ".join(
        f"{name}={array.dtype}{list(array.shape)}"
        for name, array in sorted(self._arrays.items())
    )
    return f"ArrayDataSource(len={self._length}, {specs})"

  def __setstate__(self, state: dict[str, Any]) -> None:
    # Arrays are writable again after unpickling (e.g. in the PyGrain worker
    # processes), so the read-only views are restored.
    self.__dict__.update(state)
    self._arrays = {
        name: _read_only_view(array) for name, array in self._arrays.items()
    }
