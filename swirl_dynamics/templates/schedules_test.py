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
    self.assertAlmostEqual(np.asarray(sched(125000)).item(), 1e-6)

  def test_warmup_hold_cosine_zero_warmup_and_zero_hold(self):
    sched = schedules.warmup_hold_cosine_schedule(
        init_value=0.0,
        peak_value=2e-4,
        warmup_steps=0,
        hold_steps=0,
        decay_steps=100,
        end_value=2e-5,
    )
    self.assertAlmostEqual(np.asarray(sched(0)).item(), 2e-4)
    self.assertAlmostEqual(np.asarray(sched(50)).item(), 1.1e-4, places=7)
    self.assertAlmostEqual(np.asarray(sched(100)).item(), 2e-5)
    self.assertAlmostEqual(np.asarray(sched(200)).item(), 2e-5)

  @parameterized.parameters(
      dict(peak_value=0.0, warmup_steps=10, hold_steps=10, decay_steps=10),
      dict(peak_value=-1e-4, warmup_steps=10, hold_steps=10, decay_steps=10),
      dict(peak_value=1e-4, warmup_steps=-1, hold_steps=10, decay_steps=10),
      dict(peak_value=1e-4, warmup_steps=10, hold_steps=-1, decay_steps=10),
      dict(peak_value=1e-4, warmup_steps=10, hold_steps=10, decay_steps=0),
      dict(peak_value=1e-4, warmup_steps=10, hold_steps=10, decay_steps=-5),
  )
  def test_warmup_hold_cosine_invalid_params(
      self, peak_value, warmup_steps, hold_steps, decay_steps
  ):
    with self.assertRaises(ValueError):
      schedules.warmup_hold_cosine_schedule(
          init_value=0.0,
          peak_value=peak_value,
          warmup_steps=warmup_steps,
          hold_steps=hold_steps,
          decay_steps=decay_steps,
          end_value=1e-6,
      )

  def test_get_lr_schedule_modes_and_validation(self):
    cosine = schedules.get_lr_schedule(
        name="cosine_decay",
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=100,
        decay_steps=1000,
        end_value=1e-6,
    )
    self.assertAlmostEqual(np.asarray(cosine(0)).item(), 0.0)
    self.assertAlmostEqual(np.asarray(cosine(100)).item(), 3e-4)
    self.assertAlmostEqual(np.asarray(cosine(1000)).item(), 1e-6)
    self.assertAlmostEqual(np.asarray(cosine(1500)).item(), 1e-6)

    whc = schedules.get_lr_schedule(
        name="warmup_hold_cosine",
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=100,
        hold_steps=400,
        decay_steps=1000,
        end_value=1e-6,
    )
    self.assertAlmostEqual(np.asarray(whc(0)).item(), 0.0)
    self.assertAlmostEqual(np.asarray(whc(100)).item(), 3e-4)
    self.assertAlmostEqual(np.asarray(whc(500)).item(), 3e-4)
    self.assertAlmostEqual(np.asarray(whc(1000)).item(), 1e-6)

    with self.assertRaisesRegex(ValueError, "Unknown schedule name"):
      schedules.get_lr_schedule(
          name="invalid",
          init_value=0.0,
          peak_value=3e-4,
          warmup_steps=100,
          decay_steps=900,
          end_value=1e-6,
      )

  @parameterized.parameters(
      dict(
          name="cosine_decay",
          peak_value=0.0,
          warmup_steps=10,
          decay_steps=100,
          hold_steps=0,
      ),
      dict(
          name="cosine_decay",
          peak_value=3e-4,
          warmup_steps=-1,
          decay_steps=100,
          hold_steps=0,
      ),
      dict(
          name="cosine_decay",
          peak_value=3e-4,
          warmup_steps=10,
          decay_steps=0,
          hold_steps=0,
      ),
      dict(
          name="cosine_decay",
          peak_value=3e-4,
          warmup_steps=10,
          decay_steps=100,
          hold_steps=10,
      ),
      dict(
          name="warmup_hold_cosine",
          peak_value=3e-4,
          warmup_steps=60,
          decay_steps=100,
          hold_steps=40,
      ),
      dict(
          name="warmup_hold_cosine",
          peak_value=3e-4,
          warmup_steps=10,
          decay_steps=100,
          hold_steps=-5,
      ),
  )
  def test_get_lr_schedule_invalid_params(
      self, name, peak_value, warmup_steps, decay_steps, hold_steps
  ):
    with self.assertRaises(ValueError):
      schedules.get_lr_schedule(
          name=name,
          init_value=0.0,
          peak_value=peak_value,
          warmup_steps=warmup_steps,
          decay_steps=decay_steps,
          end_value=1e-6,
          hold_steps=hold_steps,
      )


if __name__ == "__main__":
  absltest.main()
