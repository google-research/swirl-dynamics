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

"""Reusable learning-rate and EMA schedules for training."""

import optax


def warmup_hold_cosine_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    hold_steps: int,
    decay_steps: int,
    end_value: float,
) -> optax.Schedule:
  """Creates a schedule with linear warmup, hold, and cosine decay.

  Args:
    init_value: Initial learning rate.
    peak_value: Peak learning rate after warmup.
    warmup_steps: Number of steps to warmup.
    hold_steps: Number of steps to hold at peak learning rate.
    decay_steps: Number of steps to decay.
    end_value: Learning rate at the end of decay.

  Returns:
    An optax learning rate schedule.

  Raises:
    ValueError: If `peak_value <= 0`, `warmup_steps < 0`, `hold_steps < 0`, or
      `decay_steps <= 0`.
  """
  if peak_value <= 0:
    raise ValueError(f"`peak_value` must be positive, got {peak_value}.")
  if warmup_steps < 0:
    raise ValueError(
        f"`warmup_steps` must be non-negative, got {warmup_steps}."
    )
  if hold_steps < 0:
    raise ValueError(f"`hold_steps` must be non-negative, got {hold_steps}.")
  if decay_steps <= 0:
    raise ValueError(f"`decay_steps` must be positive, got {decay_steps}.")
  warmup_fn = optax.linear_schedule(
      init_value, peak_value, transition_steps=warmup_steps
  )
  hold_fn = optax.constant_schedule(peak_value)
  alpha = end_value / peak_value
  decay_fn = optax.cosine_decay_schedule(peak_value, decay_steps, alpha=alpha)
  return optax.join_schedules(
      schedules=[warmup_fn, hold_fn, decay_fn],
      boundaries=[warmup_steps, warmup_steps + hold_steps],
  )


def get_lr_schedule(
    name: str,
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float,
    hold_steps: int = 0,
) -> optax.Schedule:
  """Resolves a learning rate schedule by name and configures it.

  Args:
    name: Schedule name ('cosine_decay' or 'warmup_hold_cosine').
    init_value: Initial learning rate.
    peak_value: Peak learning rate.
    warmup_steps: Warmup steps.
    decay_steps: Total steps (including warmup and hold phases).
    end_value: End learning rate.
    hold_steps: Hold steps (only for 'warmup_hold_cosine').

  Returns:
    An optax learning rate schedule.

  Raises:
    ValueError: If `name` is unknown or schedule step/value parameters are out
      of range.
  """
  if peak_value <= 0:
    raise ValueError(f"`peak_value` must be positive, got {peak_value}.")
  if warmup_steps < 0:
    raise ValueError(
        f"`warmup_steps` must be non-negative, got {warmup_steps}."
    )
  if hold_steps < 0:
    raise ValueError(f"`hold_steps` must be non-negative, got {hold_steps}.")
  if decay_steps <= 0:
    raise ValueError(f"`decay_steps` must be positive, got {decay_steps}.")
  if name == "cosine_decay":
    if hold_steps != 0:
      raise ValueError(
          f"`hold_steps` is unsupported for 'cosine_decay', got {hold_steps};"
          " use 'warmup_hold_cosine' instead."
      )
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_value,
    )
  elif name == "warmup_hold_cosine":
    total_steps = decay_steps
    decay_phase_steps = total_steps - warmup_steps - hold_steps
    return warmup_hold_cosine_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        hold_steps=hold_steps,
        decay_steps=decay_phase_steps,
        end_value=end_value,
    )
  else:
    raise ValueError(f"Unknown schedule name: {name}")
