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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from swirl_dynamics.templates import schedules


class SchedulesTest(parameterized.TestCase):

  def test_warmup_hold_cosine_schedule(self):
    sched = schedules.warmup_hold_cosine_schedule(
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=10000,
        hold_steps=40000,
        decay_steps=50000,
        end_value=1e-6,
    )
    self.assertAlmostEqual(np.asarray(sched(0)).item(), 0.0)
    self.assertAlmostEqual(np.asarray(sched(5000)).item(), 1.5e-4)
    self.assertAlmostEqual(np.asarray(sched(10000)).item(), 3e-4)
    self.assertAlmostEqual(np.asarray(sched(30000)).item(), 3e-4)
    self.assertAlmostEqual(np.asarray(sched(50000)).item(), 3e-4)
    self.assertAlmostEqual(np.asarray(sched(75000)).item(), 1.505e-4, places=6)
    self.assertAlmostEqual(np.asarray(sched(100000)).item(), 1e-6)

  def test_get_lr_schedule_validation(self):
    sched = schedules.get_lr_schedule(
        name="cosine_decay",
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=100,
        decay_steps=900,
        end_value=1e-6,
    )
    self.assertAlmostEqual(np.asarray(sched(0)).item(), 0.0)
    self.assertAlmostEqual(np.asarray(sched(100)).item(), 3e-4)

    with self.assertRaisesRegex(ValueError, "Unknown schedule name"):
      schedules.get_lr_schedule(
          name="invalid",
          init_value=0.0,
          peak_value=3e-4,
          warmup_steps=100,
          decay_steps=900,
          end_value=1e-6,
      )


if __name__ == "__main__":
  absltest.main()
