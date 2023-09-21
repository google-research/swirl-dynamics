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

"""Some utility functions for data i/o and processing."""

from collections.abc import Mapping
import io
from typing import Any

from etils import epath
import h5py
import numpy as np
import tensorflow as tf

gfile = tf.io.gfile
exists = tf.io.gfile.exists


def read_nparray_from_hdf5(
    file_path: epath.PathLike, *keys
) -> tuple[np.ndarray, ...]:
  """Read specified fields from a given hdf5 file as numpy arrays."""
  if not exists(file_path):
    raise FileNotFoundError(f"No data file found at {file_path}")

  with gfile.GFile(file_path, "rb") as f:
    with h5py.File(f, "r") as hf:
      data = tuple(np.asarray(hf[key]) for key in keys)
  return data


def _save_dict_to_hdf5(group: h5py.Group, data: Mapping[str, Any]) -> None:
  """Save fields to hdf5 groups recursively."""
  for key, value in data.items():
    if isinstance(value, dict):
      subgroup = group.create_group(key)
      _save_dict_to_hdf5(subgroup, value)
    else:
      group.create_dataset(key, data=value)


def save_dict_to_hdf5(
    save_path: epath.PathLike, data: Mapping[str, Any]
) -> None:
  """Save a (possibly nested) dictionary to hdf5 file."""
  bio = io.BytesIO()
  with h5py.File(bio, "w") as f:
    _save_dict_to_hdf5(f, data)

  with gfile.GFile(save_path, "w") as f:
    f.write(bio.getvalue())
