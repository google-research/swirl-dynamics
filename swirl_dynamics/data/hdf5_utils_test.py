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

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from jax import tree_util
import numpy as np
from swirl_dynamics.data import hdf5_utils


class Hdf5UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("float64", np.float64, np.float64),
      ("int64", np.int64, np.int64),
      ("preserve_float64", np.float64, None),
      ("preserve_int64", np.int64, None),
  )
  def test_requested_dtype_is_preserved_in_nested_groups(
      self, stored_dtype, requested_dtype
  ):
    if np.issubdtype(stored_dtype, np.integer):
      values = np.array([2**40 + 1, 2**40 + 3], dtype=stored_dtype)
    else:
      values = np.array([1.0 + 2**-40, 2.0 + 2**-40], dtype=stored_dtype)
    save_path = os.path.join(self.create_tempdir().full_path, "nested.hdf5")
    hdf5_utils.save_array_dict(
        save_path,
        {
            "root": values,
            "group": {"values": values, "nested": {"values": values}},
        },
    )

    restored = hdf5_utils.read_all_arrays_as_dict(
        save_path, array_dtype=requested_dtype
    )
    for actual in (
        restored["root"],
        restored["group"]["values"],
        restored["group"]["nested"]["values"],
    ):
      self.assertEqual(actual.dtype, stored_dtype)
      np.testing.assert_array_equal(actual, values)

  def test_nested_empty_array_uses_requested_dtype(self):
    save_path = os.path.join(self.create_tempdir().full_path, "empty.hdf5")
    hdf5_utils.save_array_dict(
        save_path, {"group": {"empty": np.empty((0, 2), dtype=np.float32)}}
    )
    restored = hdf5_utils.read_all_arrays_as_dict(
        save_path, array_dtype=np.float64
    )
    self.assertEqual(restored["group"]["empty"].dtype, np.float64)
    self.assertEqual(restored["group"]["empty"].shape, (0, 2))

  def test_default_array_dtype_and_scalar_values_are_unchanged(self):
    save_path = os.path.join(self.create_tempdir().full_path, "default.hdf5")
    hdf5_utils.save_array_dict(
        save_path,
        {
            "group": {
                "values": np.ones(2, dtype=np.float64),
                "scalar": 2**40 + 1,
            }
        },
    )
    restored = hdf5_utils.read_all_arrays_as_dict(save_path)
    self.assertEqual(restored["group"]["values"].dtype, np.float32)
    self.assertEqual(restored["group"]["scalar"], 2**40 + 1)

  @parameterized.parameters(
      ({"a": 0, "b": 1},),
      ({"a": {"b": np.ones((10,)), "c": 2}},),
      ({"a": {"b": np.ones((10,)), "c": 2.0 * np.ones((3, 3))}, "d": 8},),
  )
  def test_save_and_load_whole_dicts(self, test_input):
    tmp_dir = self.create_tempdir().full_path
    save_path = os.path.join(tmp_dir, "test.hdf5")
    hdf5_utils.save_array_dict(save_path, test_input)
    self.assertTrue(os.path.exists(save_path))

    restored = hdf5_utils.read_all_arrays_as_dict(save_path)
    self.assertEqual(
        tree_util.tree_flatten(test_input)[1],
        tree_util.tree_flatten(restored)[1],
    )
    self.assertTrue(
        np.all(
            tree_util.tree_flatten(
                tree_util.tree_map(np.array_equal, test_input, restored)
            )[0]
        )
    )

  @parameterized.parameters(
      ({"a": 0, "b": 1}, {"a": 0, "b": 1}),
      ({"a": {"b": 1, "c": 2}}, {"a/b": 1, "a/c": 2}),
  )
  def test_save_and_load_nparrays(self, test_input, check_items):
    tmp_dir = self.create_tempdir().full_path
    save_path = os.path.join(tmp_dir, "test.hdf5")
    hdf5_utils.save_array_dict(save_path, test_input)
    self.assertTrue(os.path.exists(save_path))
    for key, value in check_items.items():
      saved = hdf5_utils.read_single_array(save_path, key)
      self.assertEqual(saved, value)


if __name__ == "__main__":
  flags.FLAGS.mark_as_parsed()
  absltest.main()
