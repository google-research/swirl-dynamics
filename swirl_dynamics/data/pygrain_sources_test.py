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

import copy

from absl.testing import absltest
from absl.testing import parameterized
import grain.python as pygrain
import numpy as np
from swirl_dynamics.data import pygrain_sources


class ArrayDataSourceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.u = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
    self.t = np.arange(15, dtype=np.float32).reshape(5, 3)
    self.source = pygrain_sources.ArrayDataSource(u=self.u, t=self.t)

  def test_length_and_records(self):
    self.assertLen(self.source, 5)
    for idx in (0, 3, np.int64(4)):
      record = self.source[idx]
      self.assertSameElements(record.keys(), ("u", "t"))
      np.testing.assert_array_equal(record["u"], self.u[idx])
      np.testing.assert_array_equal(record["t"], self.t[idx])

  def test_raises_out_of_range_index(self):
    with self.assertRaises(IndexError):
      _ = self.source[5]

  def test_raises_non_integer_index(self):
    with self.assertRaises(TypeError):
      _ = self.source[1.0]  # pyrefly: ignore[bad-index]

  def test_raises_no_arrays(self):
    with self.assertRaisesRegex(ValueError, "At least one array"):
      pygrain_sources.ArrayDataSource()

  def test_raises_mismatched_lengths(self):
    with self.assertRaisesRegex(ValueError, "same leading dimension"):
      pygrain_sources.ArrayDataSource(u=np.zeros((5, 3)), t=np.zeros((4, 3)))

  def test_raises_scalar_array(self):
    with self.assertRaisesRegex(ValueError, "at least one dimension"):
      pygrain_sources.ArrayDataSource(u=np.zeros((5, 3)), t=np.asarray(1.0))

  def test_records_are_read_only_views(self):
    record = self.source[1]
    self.assertTrue(np.shares_memory(record["u"], self.u))
    with self.assertRaises(ValueError):
      record["u"] += 1.0
    # The original arrays are neither modified nor made read-only.
    np.testing.assert_array_equal(
        self.u, np.arange(30, dtype=np.float32).reshape(5, 3, 2)
    )
    self.assertTrue(self.u.flags.writeable)

  def test_repr_is_deterministic_and_data_independent(self):
    other = pygrain_sources.ArrayDataSource(
        t=np.ones((5, 3), dtype=np.float32),
        u=np.ones((5, 3, 2), dtype=np.float32),
    )
    # Same specs with different data and keyword order.
    self.assertEqual(repr(self.source), repr(other))
    self.assertEqual(
        repr(self.source),
        "ArrayDataSource(len=5, t=float32[5, 3], u=float32[5, 3, 2])",
    )

  def test_setstate_restores_read_only_views(self):
    restored = copy.deepcopy(self.source)
    self.assertLen(restored, 5)
    self.assertEqual(repr(restored), repr(self.source))
    record = restored[2]
    np.testing.assert_array_equal(record["u"], self.u[2])
    with self.assertRaises(ValueError):
      record["t"] *= 2.0

  def test_dataloader_reads_all_records(self):
    # The read options must work with the installed Grain version.
    read_options = pygrain_sources.in_memory_read_options()
    loader = pygrain.DataLoader(
        data_source=self.source,
        sampler=pygrain.IndexSampler(
            num_records=len(self.source),
            shard_options=pygrain.NoSharding(),
            shuffle=False,
            num_epochs=1,
        ),
        operations=[pygrain.Batch(batch_size=2, drop_remainder=False)],
        read_options=read_options,
    )
    batches = list(loader)
    self.assertLen(batches, 3)
    np.testing.assert_array_equal(
        np.concatenate([batch["u"] for batch in batches]), self.u
    )
    np.testing.assert_array_equal(
        np.concatenate([batch["t"] for batch in batches]), self.t
    )


if __name__ == "__main__":
  absltest.main()
