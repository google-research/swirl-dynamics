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

"""Tests for utils."""

import tempfile

from absl.testing import absltest
from clu import metric_writers
import numpy as np
from swirl_dynamics.templates import utils

abs_tol = 1e-6


class LoadTFeventsTest(absltest.TestCase):

  def test_load_scalars(self):
    steps = 10
    rng = np.random.default_rng(42)
    loss = rng.uniform(size=(steps,))
    with tempfile.TemporaryDirectory() as temp_dir:
      writer = metric_writers.create_default_writer(temp_dir)
      for s, l in enumerate(loss):
        writer.write_scalars(s, {"loss": l})

      writer.flush()
      loaded = utils.load_scalars_from_tfevents(temp_dir)
      loaded = [loaded[s]["loss"] for s in range(steps)]
      self.assertSequenceAlmostEqual(loaded, loss)


if __name__ == "__main__":
  absltest.main()
