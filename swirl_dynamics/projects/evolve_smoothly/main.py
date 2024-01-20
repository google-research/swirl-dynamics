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

r"""The main entry point for running training loops.

This file is intentionally kept short. The majority of the logic is in the
libraries, which can be easily tested and imported in Colab.

"""

import enum

from absl import app
from absl import flags
from absl import logging
from clu import platform
import gin
import jax
from swirl_dynamics.templates import train
import tensorflow as tf


@enum.unique
class RunMode(enum.Enum):
  TRAIN = "train"


FLAGS = flags.FLAGS

_GIN_FILE = flags.DEFINE_multi_string(
    "gin_file",
    default="third_party/py/swirl_dynamics/projects/evolve_smoothly/configs/batch_decode.gin",
    help=(
        "Path to gin configuration file. Multiple paths may be passed and "
        "will be imported in the given order, with later configurations  "
        "overriding earlier ones."
    ),
)

_GIN_BINDINGS = flags.DEFINE_multi_string(
    "gin_bindings", default=[], help="Individual gin bindings."
)

_RUN_MODE = flags.DEFINE_enum_class(
    "run_mode",
    default=RunMode.TRAIN,
    enum_class=RunMode,
    help="The mode to run - only `train` supported now.",
)

_MODULE_BY_RUN_MODE = {
    RunMode.TRAIN: train.run,
}


def main(argv):
  del argv

  # Hide any GPUs and TPUs from TensorFlow. Otherwise TF might reserve memory
  # and make it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")
  tf.config.experimental.set_visible_devices([], "TPU")

  # Flags --jax_backend_target and --jax_xla_backend are available through JAX.
  if FLAGS.jax_backend_target:
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
    jax_xla_backend = (
        "None" if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
    )
    logging.info("Using JAX XLA backend %s", jax_xla_backend)

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f"process_index: {jax.process_index()}, "
      f"process_count: {jax.process_count()}"
  )

  gin.parse_config_files_and_bindings(_GIN_FILE.value, _GIN_BINDINGS.value)
  logging.info("Gin configuration:")
  for line in gin.config_str().splitlines():
    logging.info(line)

  entry_func = _MODULE_BY_RUN_MODE[_RUN_MODE.value]
  configurable_run = gin.get_configurable(entry_func)
  configurable_run()


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
