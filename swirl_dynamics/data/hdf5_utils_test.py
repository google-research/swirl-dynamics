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

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from jax import tree_util
import numpy as np
from swirl_dynamics.data import hdf5_utils


class Hdf5UtilsTest(parameterized.TestCase):

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
