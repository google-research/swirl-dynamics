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

import os

from absl.testing import absltest
from absl.testing import parameterized
from swirl_dynamics.data import utils


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      ({"a": 0, "b": 1}, {"a": 0, "b": 1}),
      ({"a": {"b": 1, "c": 2}}, {"a/b": 1, "a/c": 2}),
  )
  def test_save_and_load_nparrays_from_hdf5(self, test_input, check_items):
    tmp_dir = self.create_tempdir().full_path
    save_path = os.path.join(tmp_dir, "test.hdf5")
    utils.save_dict_to_hdf5(save_path, test_input)
    self.assertTrue(os.path.exists(save_path))
    for key, value in check_items.items():
      (saved,) = utils.read_nparray_from_hdf5(save_path, key)
      self.assertEqual(saved, value)


if __name__ == "__main__":
  absltest.main()
