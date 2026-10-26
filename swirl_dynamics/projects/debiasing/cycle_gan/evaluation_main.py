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

"""Evaluates a trained CycleGAN debiasing model on the held-out samples.

Restores the latest checkpoint written by `main_train` in `--workdir`,
translates the held-out low-fidelity samples and writes the distributional,
spectral and cycle-consistency metrics (see `experiment.evaluate`) as JSON to
`--output_path` (default: `<workdir>/metrics.json`).

Example:
  evaluation_main --config=configs/config_ks_1d.py --workdir=/tmp/cycle_gan_ks
"""

from absl import app
from absl import flags
from ml_collections import config_flags
from swirl_dynamics.projects.debiasing.cycle_gan import experiment

_WORKDIR = flags.DEFINE_string(
    "workdir",
    "",
    "Directory with the training checkpoints (required).",
    allow_override=True,
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "",
    "Where to write the metrics; defaults to <workdir>/metrics.json.",
)
if "config" not in flags.FLAGS:
  config_flags.DEFINE_config_file("config")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if not _WORKDIR.value:
    raise app.UsageError("--workdir is required.")
  experiment.evaluate_experiment(
      flags.FLAGS.config, _WORKDIR.value, _OUTPUT_PATH.value
  )


if __name__ == "__main__":
  app.run(main)
