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

"""Utility functions for hdf5 file reading and writing."""

from collections.abc import Mapping, Sequence
import io
from typing import Any

from etils import epath
import h5py
import numpy as np

filesys = epath.backend.tf_backend


def read_arrays_as_tuple(
    file_path: epath.PathLike, keys: Sequence[str], dtype: Any = np.float32
) -> tuple[np.ndarray, ...]:
  """Reads specified fields from a given file as numpy arrays."""
  if not filesys.exists(file_path):
    raise FileNotFoundError(f"No data file found at {file_path}")

  with filesys.gfile.GFile(file_path, "rb") as f:
    with h5py.File(f, "r") as hf:
      data = tuple(np.asarray(hf[key], dtype=dtype) for key in keys)
  return data


def read_single_array(
    file_path: epath.PathLike, key: str, dtype: Any = np.float32
) -> np.ndarray:
  """Reads a single field from a given file."""
  return read_arrays_as_tuple(file_path, [key], dtype)[0]


def _read_group(
    group: h5py.Group, array_dtype: Any = np.float32
) -> Mapping[str, Any]:
  """Recursively reads a hdf5 group."""
  out = {}
  for key in group.keys():
    if isinstance(group[key], h5py.Group):
      out[key] = _read_group(group[key])
    elif isinstance(group[key], h5py.Dataset):
      if group[key].shape:  # pytype: disable=attribute-error
        out[key] = np.asarray(group[key], dtype=array_dtype)
      else:
        out[key] = group[key][()]
    else:
      raise ValueError(f"Unknown type for key {key}")
  return out


def read_all_arrays_as_dict(
    file_path: epath.PathLike, array_dtype: Any = np.float32
) -> Mapping[str, Any]:
  """Reads the entire contents of a file as a (possibly nested) dictionary."""
  if not filesys.exists(file_path):
    raise FileNotFoundError(f"No data file found at {file_path}")

  with filesys.gfile.GFile(file_path, "rb") as f:
    with h5py.File(f, "r") as hf:
      return _read_group(hf, array_dtype)


def _save_array_dict(group: h5py.Group, data: Mapping[str, Any]) -> None:
  """Saves a nested python dictionary to hdf5 groups recursively."""
  for key, value in data.items():
    if isinstance(value, dict):
      subgroup = group.create_group(key)
      _save_array_dict(subgroup, value)
    else:
      group.create_dataset(key, data=value)


def save_array_dict(save_path: epath.PathLike, data: Mapping[str, Any]) -> None:
  """Saves a dictionary (possibly nested) to hdf5 file."""
  bio = io.BytesIO()
  with h5py.File(bio, "w") as f:
    _save_array_dict(f, data)

  with filesys.gfile.GFile(save_path, "w") as f:
    f.write(bio.getvalue())
