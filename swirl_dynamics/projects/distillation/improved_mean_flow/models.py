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

r"""Improved Mean Flow (iMF) models with v-loss and flexible CFG.

Implements Geng et al. (CVPR 2026, arXiv:2512.02012v1):
  * `v`-loss reformulation via compound velocity
    `V_theta(x_t, t, s) = u_theta(x_t, t, s) + (t - s) * sg(d/dt u_theta)`
    differentiated along `sg(v_theta(x_t, t))` (zero conditional-tangent
    variance in the JVP).
  * Boundary (`v_theta(x_t, t) = u_theta(x_t, t, t)`) and dedicated auxiliary
    `v`-head (`head="v"` trained jointly, omitted at inference) realizations.
  * Flexible classifier-free guidance over scale `omega` and interval
    `(t_min, t_max)` using the contraction-stable Appendix A Eq. 17 target.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
from typing import Any, Literal, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from swirl_dynamics.lib import flow_map
from swirl_dynamics.lib.flow_matching import interpolants
from swirl_dynamics.templates import models
from swirl_dynamics.templates import trainers

Array = jax.Array
ArrayShape = Sequence[int]
CfgMode = Literal["none", "fixed", "scale", "interval"]
CondDict = flow_map.CondDict
ShapeDict = flow_map.ShapeDict
VariableDict = trainers.VariableDict


def _resolve_teacher_vars(
    params_flow: VariableDict | models.PyTree,
) -> Mapping[str, Any]:
  """Normalizes either raw teacher params or a full Flax variable dict."""
  if isinstance(params_flow, Mapping) and "params" in params_flow:
    return params_flow
  return {"params": params_flow}


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImprovedMeanFlowModel(models.BaseModel):
  """Unconditional Improved Mean Flow model (Geng et al., CVPR 2026).

  Attributes:
    input_shape: Tensor shape of a single sample (excluding batch dimension).
    mean_flow_model: Flax module predicting `u(x_t, t, s)` (and optionally
      supporting `head="u"|"v"|"both"` when `use_aux_v_head=True`).
    flow_model: Optional frozen teacher velocity module `v_phi(x, sigma)`. When
      `None`, the base velocity is taken directly from `interpolant`.
    params_flow: Parameters or full Flax variable dictionary of `flow_model`.
    interpolant: Stochastic interpolant linking `x_0` (data) and `x_1` (prior).
    noising_process: Callable sampling latent noise matching `x_0.shape`.
    time_sampling: Callable sampling ordered `(t, s)` time pairs (`t >= s`).
    min_train_time: Lower bound on sampled training times.
    max_train_time: Upper bound on sampled training times.
    diagonal_sampling_rate: Probability of sampling `s = t` during training
      (Table 4 default is `0.5`).
    norm_power: Exponent `p` in adaptive per-sample loss weighting.
    norm_regularization: Positive stabilizer `c` in adaptive loss weighting.
    use_aux_v_head: Whether to use a dedicated auxiliary `v`-head (`head="v"` at
      diagonal `s = t`) instead of the boundary `u(x_t, t, t)`.
    aux_v_loss_weight: Non-negative weight `lambda_v` on the auxiliary `v`-loss.
    number_of_eval_steps: Step counts evaluated via backward sampling.
  """

  input_shape: tuple[int, ...]
  mean_flow_model: nn.Module
  flow_model: nn.Module | None = None
  params_flow: VariableDict | models.PyTree | None = None
  interpolant: interpolants.StochasticInterpolant = interpolants.RectifiedFlow()
  noising_process: Callable[[Array, ArrayShape], Array] = jax.random.normal
  time_sampling: Callable[..., tuple[Array, Array]] = (
      flow_map.time_sampler_mean_flow
  )
  min_train_time: float = 1e-4
  max_train_time: float = 1.0 - 1e-4
  diagonal_sampling_rate: float = 0.5
  norm_power: float = 1.0
  norm_regularization: float = 1e-2
  use_aux_v_head: bool = False
  aux_v_loss_weight: float = 1.0
  number_of_eval_steps: tuple[int, ...] = (1, 2, 4)

  def __post_init__(self) -> None:
    if self.aux_v_loss_weight < 0.0:
      raise ValueError(
          "`aux_v_loss_weight` must be non-negative, got "
          f"{self.aux_v_loss_weight}."
      )

  def _init_cond(self) -> CondDict | None:
    """Returns dummy conditioning for `initialize` (or `None` if uncond)."""
    return None

  def initialize(self, rng: Array) -> Mapping[str, Any]:
    """Initializes variables of `mean_flow_model` (both heads if enabled)."""
    x = jnp.ones((1,) + self.input_shape)
    t = jnp.ones((1,))
    s = jnp.ones((1,))
    kwargs: dict[str, Any] = {"is_training": False}
    cond = self._init_cond()
    if cond is not None:
      kwargs["cond"] = cond
    if self.use_aux_v_head:
      kwargs["head"] = "both"
    return self.mean_flow_model.init(rng, x_s=x, t=t, s=s, **kwargs)

  def _prepare_training_cond(
      self,
      batch: models.BatchType,
      time_t: Array,
      cfg_rng: Array,
  ) -> tuple[CondDict | None, Mapping[str, Array] | None]:
    """Prepares `(cond, cfg_aux)` for training (overridden by conditional)."""
    del batch, time_t, cfg_rng
    return None, None

  def _prepare_eval_cond(self, batch: models.BatchType) -> CondDict | None:
    """Prepares conditioning for evaluation (overridden by conditional)."""
    del batch
    return None

  def _prepare_eval_cfg_aux(
      self, batch: models.BatchType, cond: CondDict | None
  ) -> Mapping[str, Array] | None:
    """Prepares optional CFG metadata for guided `eval_fn` targets."""
    del batch, cond
    return None

  def _apply_teacher_flow(
      self,
      flow_model: nn.Module,
      params_flow: VariableDict | models.PyTree,
      x_t: Array,
      time: Array,
      cond: CondDict | None,
  ) -> Array:
    """Evaluates the frozen teacher `flow_model` at `(x_t, time)`."""
    del cond
    teacher_vars = _resolve_teacher_vars(params_flow)
    return cast(
        Array,
        flow_model.apply(teacher_vars, x=x_t, sigma=time, is_training=False),
    )

  def _compute_base_velocity(
      self,
      time: Array,
      x_t: Array,
      batch: models.BatchType,
      noise: Array,
      cond: CondDict | None,
  ) -> Array:
    """Computes base velocity from teacher `flow_model` or `interpolant`."""
    if self.flow_model is not None:
      if self.params_flow is None:
        raise ValueError(
            "`params_flow` must be provided when `flow_model` is set."
        )
      return self._apply_teacher_flow(
          self.flow_model, self.params_flow, x_t, time, cond
      )
    return self.interpolant.calculate_time_derivative_interpolant(
        time, batch["x_0"], batch["x_1"], noise
    )

  def _apply_network(
      self,
      variables: models.PyTree,
      x_s: Array,
      t: Array,
      s: Array,
      cond: CondDict | None,
      *,
      is_training: bool,
      head: Literal["u", "v"] | None = None,
      rngs: dict[str, Array] | None = None,
      mutable: Any = False,
  ) -> Any:
    """Applies `mean_flow_model` with optional conditioning and head."""
    kwargs: dict[str, Any] = {
        "is_training": is_training,
        "rngs": rngs,
        "mutable": mutable,
    }
    if cond is not None:
      kwargs["cond"] = cond
    if self.use_aux_v_head and head is not None:
      kwargs["head"] = head
    return self.mean_flow_model.apply(variables, x_s, t, s, **kwargs)

  def _compute_velocity_target(
      self,
      variables: models.PyTree,
      x_t: Array,
      time_t: Array,
      v_base: Array,
      v_cond: Array,
      cond: CondDict | None,
      cfg_aux: Mapping[str, Array] | None,
  ) -> Array:
    """Returns stop-gradient velocity target `v_tgt` (unguided in base)."""
    del variables, x_t, time_t, v_cond, cond, cfg_aux
    return jax.lax.stop_gradient(v_base)

  def loss_fn(
      self,
      params: models.PyTree,
      batch: models.BatchType,
      rng: Array,
      mutables: models.PyTree,
  ) -> models.LossAndAux:
    """Computes the iMF `v`-loss (and optional auxiliary `v`-head loss)."""
    batch_size = len(batch["x_0"])
    time_rng, noise_rng, cfg_rng, drop_u_rng, drop_v_rng = jax.random.split(
        rng, num=5
    )
    time_t, time_s = self.time_sampling(
        time_rng,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    cond, cfg_aux = self._prepare_training_cond(batch, time_t, cfg_rng)
    v_base = self._compute_base_velocity(time_t, x_t, batch, noise, cond)

    mutable_keys = list(mutables.keys())
    variables = {"params": params, **mutables}

    if self.use_aux_v_head:
      v_cond_raw, _ = self._apply_network(
          variables,
          x_t,
          time_t,
          time_t,
          cond,
          is_training=True,
          head="v",
          rngs={"dropout": drop_v_rng},
          mutable=mutable_keys,
      )
      v_cond = cast(Array, v_cond_raw)
    else:
      v_cond = jax.lax.stop_gradient(
          cast(
              Array,
              self._apply_network(
                  variables,
                  x_t,
                  time_t,
                  time_t,
                  cond,
                  is_training=False,
                  head=None,
                  mutable=False,
              ),
          )
      )

    v_target = self._compute_velocity_target(
        variables, x_t, time_t, v_base, v_cond, cond, cfg_aux
    )

    def _u_fn(x_in: Array, t_in: Array, s_in: Array) -> tuple[Array, Any]:
      u_out, updated_mut = self._apply_network(
          variables,
          x_in,
          t_in,
          s_in,
          cond,
          is_training=True,
          head="u" if self.use_aux_v_head else None,
          rngs={"dropout": drop_u_rng},
          mutable=mutable_keys,
      )
      return cast(Array, u_out), jax.lax.stop_gradient(updated_mut)

    compound_v, _, _, updated_mutables = flow_map.compute_compound_velocity(
        _u_fn, x_t, time_t, time_s, v_cond
    )
    loss_u, raw_mse_u = flow_map.adaptive_weighted_loss(
        compound_v - v_target,
        norm_power=self.norm_power,
        norm_regularization=self.norm_regularization,
    )

    if self.use_aux_v_head:
      loss_v, raw_mse_v = flow_map.adaptive_weighted_loss(
          v_cond - v_target,
          norm_power=self.norm_power,
          norm_regularization=self.norm_regularization,
      )
      loss = loss_u + self.aux_v_loss_weight * loss_v
    else:
      loss_v = jnp.zeros((), dtype=jnp.float32)
      raw_mse_v = jnp.zeros((), dtype=jnp.float32)
      loss = loss_u

    metric = {
        "loss": loss,
        "loss_u": loss_u,
        "loss_v": loss_v,
        "raw_mse": raw_mse_u,
        "raw_mse_v": raw_mse_v,
    }
    return loss, (metric, updated_mutables)

  def eval_fn(
      self,
      variables: models.PyTree,
      batch: models.BatchType,
      rng: Array,
  ) -> models.ArrayDict:
    """Computes `v`-loss eval MSE and multi-step backward samples."""
    batch_size = len(batch["x_0"])
    time_rng, noise_rng = jax.random.split(rng, num=2)
    time_t, time_s = self.time_sampling(
        time_rng,
        batch_size,
        self.min_train_time,
        self.max_train_time,
        self.diagonal_sampling_rate,
    )

    noise = self.noising_process(noise_rng, batch["x_0"].shape)
    x_t = self.interpolant(time_t, batch["x_0"], batch["x_1"], noise)
    cond = self._prepare_eval_cond(batch)
    eval_cfg_aux = self._prepare_eval_cfg_aux(batch, cond)
    v_base = self._compute_base_velocity(time_t, x_t, batch, noise, cond)

    v_cond = cast(
        Array,
        self._apply_network(
            variables,
            x_t,
            time_t,
            time_t,
            cond,
            is_training=False,
            head="v" if self.use_aux_v_head else None,
        ),
    )
    v_target = self._compute_velocity_target(
        variables, x_t, time_t, v_base, v_cond, cond, eval_cfg_aux
    )

    def _u_fn_aux(x_in: Array, t_in: Array, s_in: Array) -> tuple[Array, None]:
      out = cast(
          Array,
          self._apply_network(
              variables,
              x_in,
              t_in,
              s_in,
              cond,
              is_training=False,
              head="u" if self.use_aux_v_head else None,
          ),
      )
      return out, None

    compound_v, _, _, _ = flow_map.compute_compound_velocity(
        _u_fn_aux, x_t, time_t, time_s, v_cond
    )
    eval_loss = jnp.mean(
        jnp.square(jnp.asarray(compound_v - v_target, dtype=jnp.float32))
    )
    if self.use_aux_v_head:
      eval_loss_v = jnp.mean(
          jnp.square(jnp.asarray(v_cond - v_target, dtype=jnp.float32))
      )
    else:
      eval_loss_v = jnp.zeros((), dtype=jnp.float32)

    def _u_infer(x_in: Array, t_in: Array, s_in: Array) -> Array:
      return _u_fn_aux(x_in, t_in, s_in)[0]

    eval_out: dict[str, Array] = {
        "eval_loss": eval_loss,
        "eval_loss_v": eval_loss_v,
    }
    for num_steps in self.number_of_eval_steps:
      eval_out[f"samples_{num_steps}"] = flow_map.sample_backward_mean_flow(
          _u_infer, batch["x_1"], num_steps=num_steps
      )
    if "samples_1" in eval_out:
      eval_out["eval_samples"] = eval_out["samples_1"]
    else:
      eval_out["eval_samples"] = flow_map.sample_backward_mean_flow(
          _u_infer, batch["x_1"], num_steps=1
      )
    return eval_out

  @staticmethod
  def inference_fn(
      variables: models.PyTree,
      mean_flow_model: nn.Module,
      *,
      use_aux_v_head: bool = False,
  ) -> Callable[..., Array]:
    """Returns an inference callable evaluating average velocity `u_theta`."""

    def _flow(
        x: Array,
        t: float | Array,
        s: float | Array,
        cond: CondDict | None = None,
    ) -> Array:
      t_arr = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float32), (x.shape[0],))
      s_arr = jnp.broadcast_to(jnp.asarray(s, dtype=jnp.float32), (x.shape[0],))
      kwargs: dict[str, Any] = {"is_training": False}
      if cond is not None:
        kwargs["cond"] = cond
      if use_aux_v_head:
        kwargs["head"] = "u"
      return cast(
          Array,
          mean_flow_model.apply(variables, x_s=x, t=t_arr, s=s_arr, **kwargs),
      )

    return _flow


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalImprovedMeanFlowModel(ImprovedMeanFlowModel):
  """Class- and CFG-conditioned Improved Mean Flow model (Section 3.2, Alg. 2).

  Supports four CFG training modes:
    * `"none"`: Standard conditional iMF without classifier-free guidance.
    * `"fixed"`: Fixed guidance scale `cfg_scale` on `cfg_interval`.
    * `"scale"`: Power-law sampled `omega ~ w**(-cfg_scale_beta)` conditioned
      via `cond["cfg_scale"]` on fixed `cfg_interval`.
    * `"interval"`: Jointly sampled `omega` and interval `(t_min, t_max)`
      conditioned via `cond["cfg_scale"]`, `cond["cfg_t_min"]`, and
      `cond["cfg_t_max"]`.

  Attributes:
    cond_shape: Mapping from conditioning key names to per-sample shapes.
    num_classes: Number of data classes (index `num_classes` is the null class).
    label_key: Key in `batch`/`cond` storing integer class labels.
    cfg_mode: Guidance mode (`"none"`, `"fixed"`, `"scale"`, or `"interval"`).
    class_dropout_rate: Probability of replacing label `c` with null class `C`.
    cfg_scale: Fixed guidance scale used when `cfg_mode == "fixed"`.
    cfg_scale_min: Lower bound of sampled guidance scale.
    cfg_scale_max: Upper bound of sampled guidance scale.
    cfg_scale_beta: Power-law exponent `beta` in `p(w) propto w**(-beta)`.
    cfg_interval: Fixed `(t_min, t_max)` interval for `"fixed"` / `"scale"`.
    cfg_t_min_range: Sampling range for `t_min` when `cfg_mode == "interval"`.
    cfg_t_max_range: Sampling range for `t_max` when `cfg_mode == "interval"`.
    eval_cfg_scale: Guidance scale injected during `eval_fn`.
    eval_cfg_interval: Guidance interval injected during `eval_fn`.
  """

  cond_shape: ShapeDict | None = None
  num_classes: int | None = None
  label_key: str = "label"
  cfg_mode: CfgMode = "none"
  class_dropout_rate: float = 0.1
  cfg_scale: float = 1.0
  cfg_scale_min: float = 1.0
  cfg_scale_max: float = 8.0
  cfg_scale_beta: float = 1.0
  cfg_interval: tuple[float, float] = (0.0, 1.0)
  cfg_t_min_range: tuple[float, float] = (0.0, 0.5)
  cfg_t_max_range: tuple[float, float] = (0.5, 1.0)
  eval_cfg_scale: float = 2.0
  eval_cfg_interval: tuple[float, float] = (0.0, 1.0)

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.cfg_mode not in ("none", "fixed", "scale", "interval"):
      raise ValueError(
          f"Unsupported `cfg_mode`: {self.cfg_mode!r}; expected one of "
          "'none', 'fixed', 'scale', 'interval'."
      )
    if self.cfg_mode != "none" and (
        self.num_classes is None or self.num_classes < 1
    ):
      raise ValueError(
          f"`num_classes` must be >= 1 when cfg_mode={self.cfg_mode!r}, got "
          f"{self.num_classes}."
      )
    if not 0.0 <= self.class_dropout_rate <= 1.0:
      raise ValueError(
          "`class_dropout_rate` must be in [0, 1], got "
          f"{self.class_dropout_rate}."
      )
    if self.cfg_scale < 1.0:
      raise ValueError(f"`cfg_scale` must be >= 1.0, got {self.cfg_scale}.")
    if self.cfg_scale_min <= 0.0 or self.cfg_scale_max < self.cfg_scale_min:
      raise ValueError(
          "Expected `0 < cfg_scale_min <= cfg_scale_max`, got "
          f"{self.cfg_scale_min=}, {self.cfg_scale_max=}."
      )
    if self.cfg_interval[0] > self.cfg_interval[1]:
      raise ValueError(f"Invalid `cfg_interval`: {self.cfg_interval}.")
    if (
        self.cfg_t_min_range[0] > self.cfg_t_min_range[1]
        or self.cfg_t_max_range[0] > self.cfg_t_max_range[1]
    ):
      raise ValueError(
          f"Invalid CFG interval ranges: {self.cfg_t_min_range=}, "
          f"{self.cfg_t_max_range=}."
      )
    if self.eval_cfg_scale < 1.0:
      raise ValueError(
          f"`eval_cfg_scale` must be >= 1.0, got {self.eval_cfg_scale}."
      )
    if self.eval_cfg_interval[0] > self.eval_cfg_interval[1]:
      raise ValueError(
          f"Invalid `eval_cfg_interval`: {self.eval_cfg_interval}."
      )

  def _init_cond(self) -> CondDict | None:
    cond: dict[str, Array] = {}
    if self.cond_shape is not None:
      cond.update(flow_map.cond_sample_from_shape(self.cond_shape, (1,)))
    if self.cfg_mode != "none":
      if self.label_key not in cond:
        cond[self.label_key] = jnp.zeros((1,), dtype=jnp.int32)
      cond["cfg_scale"] = jnp.ones((1,), dtype=jnp.float32)
      if self.cfg_mode == "interval":
        cond["cfg_t_min"] = jnp.zeros((1,), dtype=jnp.float32)
        cond["cfg_t_max"] = jnp.ones((1,), dtype=jnp.float32)
    return cond if cond else None

  def _apply_teacher_flow(
      self,
      flow_model: nn.Module,
      params_flow: VariableDict | models.PyTree,
      x_t: Array,
      time: Array,
      cond: CondDict | None,
  ) -> Array:
    teacher_vars = _resolve_teacher_vars(params_flow)
    return cast(
        Array,
        flow_model.apply(
            teacher_vars, x=x_t, sigma=time, cond=cond, is_training=False
        ),
    )

  def _extract_base_cond(self, batch: models.BatchType) -> dict[str, Array]:
    if self.cond_shape is None:
      return {}
    return {key: jnp.asarray(batch[key]) for key in self.cond_shape.keys()}

  def _prepare_training_cond(
      self,
      batch: models.BatchType,
      time_t: Array,
      cfg_rng: Array,
  ) -> tuple[CondDict | None, Mapping[str, Array] | None]:
    del time_t
    cond = self._extract_base_cond(batch)
    if self.cfg_mode == "none":
      return (cond if self.cond_shape is not None else None), None

    assert self.num_classes is not None
    batch_size = len(batch["x_0"])
    drop_rng, scale_rng, interval_rng = jax.random.split(cfg_rng, num=3)
    labels = jnp.asarray(batch[self.label_key], dtype=jnp.int32).reshape((
        batch_size,
    ))
    is_dropped = jax.random.bernoulli(
        drop_rng, self.class_dropout_rate, (batch_size,)
    )
    dropped_labels = jnp.where(
        is_dropped,
        jnp.full((batch_size,), self.num_classes, dtype=jnp.int32),
        labels,
    )
    cond[self.label_key] = dropped_labels

    if self.cfg_mode == "fixed":
      omega = jnp.full((batch_size,), self.cfg_scale, dtype=jnp.float32)
      t_min = jnp.full((batch_size,), self.cfg_interval[0], dtype=jnp.float32)
      t_max = jnp.full((batch_size,), self.cfg_interval[1], dtype=jnp.float32)
    elif self.cfg_mode == "scale":
      omega = flow_map.sample_cfg_scale(
          scale_rng,
          (batch_size,),
          w_min=self.cfg_scale_min,
          w_max=self.cfg_scale_max,
          beta=self.cfg_scale_beta,
      )
      t_min = jnp.full((batch_size,), self.cfg_interval[0], dtype=jnp.float32)
      t_max = jnp.full((batch_size,), self.cfg_interval[1], dtype=jnp.float32)
    else:
      omega = flow_map.sample_cfg_scale(
          scale_rng,
          (batch_size,),
          w_min=self.cfg_scale_min,
          w_max=self.cfg_scale_max,
          beta=self.cfg_scale_beta,
      )
      t_min, t_max = flow_map.sample_cfg_interval(
          interval_rng,
          batch_size,
          t_min_range=self.cfg_t_min_range,
          t_max_range=self.cfg_t_max_range,
      )

    cond["cfg_scale"] = omega
    if self.cfg_mode == "interval":
      cond["cfg_t_min"] = t_min
      cond["cfg_t_max"] = t_max

    cfg_aux = {
        "omega": omega,
        "t_min": t_min,
        "t_max": t_max,
        "is_dropped": is_dropped,
    }
    return cond, cfg_aux

  def _prepare_eval_cond(self, batch: models.BatchType) -> CondDict | None:
    cond = self._extract_base_cond(batch)
    if self.cfg_mode == "none":
      return cond if self.cond_shape is not None else None
    batch_size = len(batch["x_0"])
    if self.label_key in batch:
      cond[self.label_key] = jnp.asarray(
          batch[self.label_key], dtype=jnp.int32
      ).reshape((batch_size,))
    cond["cfg_scale"] = jnp.full(
        (batch_size,), self.eval_cfg_scale, dtype=jnp.float32
    )
    if self.cfg_mode == "interval":
      cond["cfg_t_min"] = jnp.full(
          (batch_size,), self.eval_cfg_interval[0], dtype=jnp.float32
      )
      cond["cfg_t_max"] = jnp.full(
          (batch_size,), self.eval_cfg_interval[1], dtype=jnp.float32
      )
    return cond

  def _prepare_eval_cfg_aux(
      self, batch: models.BatchType, cond: CondDict | None
  ) -> Mapping[str, Array] | None:
    if self.cfg_mode == "none" or cond is None:
      return None
    batch_size = len(batch["x_0"])
    t_min, t_max = (
        self.eval_cfg_interval
        if self.cfg_mode == "interval"
        else self.cfg_interval
    )
    return {
        "omega": cond["cfg_scale"],
        "t_min": jnp.full((batch_size,), t_min, dtype=jnp.float32),
        "t_max": jnp.full((batch_size,), t_max, dtype=jnp.float32),
        "is_dropped": jnp.zeros((batch_size,), dtype=jnp.bool_),
    }

  def _compute_velocity_target(
      self,
      variables: models.PyTree,
      x_t: Array,
      time_t: Array,
      v_base: Array,
      v_cond: Array,
      cond: CondDict | None,
      cfg_aux: Mapping[str, Array] | None,
  ) -> Array:
    if cfg_aux is None or cond is None:
      return jax.lax.stop_gradient(v_base)
    assert self.num_classes is not None
    batch_size = x_t.shape[0]
    uncond = dict(cond)
    uncond[self.label_key] = jnp.full(
        (batch_size,), self.num_classes, dtype=jnp.int32
    )
    v_uncond = jax.lax.stop_gradient(
        cast(
            Array,
            self._apply_network(
                variables,
                x_t,
                time_t,
                time_t,
                uncond,
                is_training=False,
                head="v" if self.use_aux_v_head else None,
                mutable=False,
            ),
        )
    )
    v_tgt, _ = flow_map.compute_guided_velocity_target(
        v_base=v_base,
        v_cond=v_cond,
        v_uncond=v_uncond,
        cfg_scale=cfg_aux["omega"],
        time_t=time_t,
        t_min=cfg_aux["t_min"],
        t_max=cfg_aux["t_max"],
        is_dropped=cfg_aux["is_dropped"],
    )
    return v_tgt

  @staticmethod
  def inference_fn(
      variables: models.PyTree,
      mean_flow_model: nn.Module,
      *,
      cfg_mode: CfgMode = "none",
      use_aux_v_head: bool = False,
      default_cfg_scale: float = 1.0,
      default_cfg_interval: tuple[float, float] = (0.0, 1.0),
  ) -> Callable[..., Array]:
    """Returns an inference callable with optional runtime CFG overrides."""

    def _flow(
        x: Array,
        t: float | Array,
        s: float | Array,
        cond: CondDict | None = None,
        *,
        cfg_scale: float | Array | None = None,
        cfg_interval: tuple[float | Array, float | Array] | None = None,
    ) -> Array:
      batch_size = x.shape[0]
      t_arr = jnp.broadcast_to(jnp.asarray(t, dtype=jnp.float32), (batch_size,))
      s_arr = jnp.broadcast_to(jnp.asarray(s, dtype=jnp.float32), (batch_size,))
      cond_out = dict(cond) if cond is not None else {}
      if cfg_mode != "none":
        if cfg_scale is not None:
          cond_out["cfg_scale"] = jnp.broadcast_to(
              jnp.asarray(cfg_scale, dtype=jnp.float32), (batch_size,)
          )
        elif "cfg_scale" not in cond_out:
          cond_out["cfg_scale"] = jnp.full(
              (batch_size,), default_cfg_scale, dtype=jnp.float32
          )
        if cfg_mode == "interval":
          if cfg_interval is not None:
            lo, hi = cfg_interval
            cond_out["cfg_t_min"] = jnp.broadcast_to(
                jnp.asarray(lo, dtype=jnp.float32), (batch_size,)
            )
            cond_out["cfg_t_max"] = jnp.broadcast_to(
                jnp.asarray(hi, dtype=jnp.float32), (batch_size,)
            )
          else:
            if "cfg_t_min" not in cond_out:
              cond_out["cfg_t_min"] = jnp.full(
                  (batch_size,), default_cfg_interval[0], dtype=jnp.float32
              )
            if "cfg_t_max" not in cond_out:
              cond_out["cfg_t_max"] = jnp.full(
                  (batch_size,), default_cfg_interval[1], dtype=jnp.float32
              )
      kwargs: dict[str, Any] = {"is_training": False}
      if cond is not None or cfg_mode != "none":
        kwargs["cond"] = cond_out
      if use_aux_v_head:
        kwargs["head"] = "u"
      return cast(
          Array,
          mean_flow_model.apply(variables, x_s=x, t=t_arr, s=s_arr, **kwargs),
      )

    return _flow
