# Improved Mean Flows (iMF)

JAX/Flax implementation of **Improved Mean Flows (iMF)** [1] for from-scratch
one-step (1-NFE) and few-step generative modeling and distillation, built on top
of the shared two-time primitives in `swirl_dynamics.lib.flow_map`.

Within `swirl_dynamics.projects.distillation`, this package extends the
two-time average-velocity formulation of `../mean_flow` [2] alongside the
Lagrangian flow-map distillation baselines in `../flow_map_matching` [3, 4].
Reusable primitives (log-space power-law CFG scale and interval samplers,
adaptive loss weighting, compound-velocity JVP operator, contraction-stable
guided targets, and multi-step samplers) reside in
`swirl_dynamics.lib.flow_map`, while the iMF backbones, `v`-loss models,
trainers, and visualizations live here.

## Mathematical Formulation & Conventions

Let $x_0 \sim p_{\mathrm{data}}$ (`batch["x_0"]`, shape `(B, *spatial, C)`),
$x_1 \sim p_{\mathrm{prior}}$ (`batch["x_1"]`, shape `(B, *spatial, C)`), and
let ordered times $t \ge s \in [t_{\min}^{\mathrm{tr}}, t_{\max}^{\mathrm{tr}}]$
(`time_t`, `time_s`, where `s` corresponds to $r$ in [1, 2]) define the linear
interpolant $x_t = (1 - t) x_0 + t x_1$ with conditional velocity
$v_{\mathrm{base}} = x_1 - x_0$ (or a frozen teacher $v_\phi(x_t, t, c)$ when
`flow_model` is provided).

### 1. From $u$-Loss to $v$-Loss & Compound Velocity Variance Elimination

Defining the average velocity
$u(x_t, t, s) = \frac{1}{t - s} \int_s^t v(x_\tau, \tau)\thinspace d\tau$ along
the probability-flow ODE $\frac{dx_\tau}{d\tau} = v(x_\tau, \tau)$ and
differentiating
$(t - s) u(x_t, t, s) = \int_s^t v(x_\tau, \tau)\thinspace d\tau$
with respect to $t$ yields the MeanFlow identity [2]:

$$
v(x_t, t) = u(x_t, t, s) + (t - s)\frac{d}{dt} u(x_t, t, s)
= u(x_t, t, s) + (t - s)\big(\partial_x u(x_t, t, s)\thinspace v(x_t, t) +
\partial_t u(x_t, t, s)\big).
$$

Original MeanFlow [2] forms a regression target
$u_{\mathrm{tgt}} = v_c - (t - s)\operatorname{sg}(\frac{d}{dt} u_\theta)$
using the noisy conditional tangent $v_c = x_1 - x_0$ inside the JVP, causing
target variance that depends on $\theta$. Improved Mean Flows [1] reformulates
the objective as a **$v$-loss** by predicting the $x_t$-measurable **compound
velocity** (`flow_map.compute_compound_velocity`):

$$
V_\theta(x_t, t, s) = u_\theta(x_t, t, s) +
(t - s)\thinspace\operatorname{sg}\Big(
  \partial_x u_\theta(x_t, t, s)\thinspace\operatorname{sg}\big(v_\theta(x_t, t)\big) +
  \partial_t u_\theta(x_t, t, s)
\Big),
$$

evaluated with a single forward-mode `jax.jvp` along tangent
$(\operatorname{sg}(v_\theta(x_t, t)), 1, 0)$ and regressed against
$\operatorname{sg}(v_{\mathrm{tgt}})$ under the adaptive per-sample loss
(`flow_map.adaptive_weighted_loss`, default $p = 1.0$, $c = 10^{-2}$):

$$
\mathcal{L}_{\mathrm{adapt}}(e) = \mathbb{E}_b\left[
  \frac{\bar{e}_b^2}{\operatorname{sg}\big((\bar{e}_b^2 + c)^p\big)}
\right],
\qquad
\bar{e}_b^2 = \mathrm{mean}_{\mathrm{spatial,ch}}(e_b^2).
$$

Because $v_\theta(x_t, t)$ is a deterministic function of $(x_t, t)$, the JVP
tangent has zero conditional sampling variance given $(x_t, t)$.

### 2. Boundary vs. Auxiliary $v$-Head Realizations

`ImprovedMeanFlowModel` supports both realizations from [1]:

* **Boundary realization (`use_aux_v_head=False`)**: Sets
  $v_\theta(x_t, t) = u_\theta(x_t, t, t)$ evaluated under `stop_gradient`
  (supervised via diagonal samples $s = t$, default
  `diagonal_sampling_rate=0.5`, where
  $V_\theta(x_t, t, t) = u_\theta(x_t, t, t)$).
  Compatible with any standard two-time backbone including
  `flow_map.RescaledMeanFlowUNet`.
* **Auxiliary $v$-head (`use_aux_v_head=True`)**: Evaluates
  $v_\theta(x_t, t) = f_\theta^{(v)}(x_t, t, t)$ from a dedicated $L$-block
  auxiliary branch (`head="v"`, Table 4 uses $L = 8$ across `iMF-B/M/L/XL`)
  trained jointly with

$$
\mathcal{L} = \mathcal{L}_{\mathrm{adapt}}\big(V_\theta - \operatorname{sg}(v_{\mathrm{tgt}})\big) +
\lambda_v\thinspace\mathcal{L}_{\mathrm{adapt}}\big(v_\theta - \operatorname{sg}(v_{\mathrm{tgt}})\big).
$$

During initialization (`initialize`), both heads are initialized via
`head="both"`. During JVP and 1-NFE inference (`head="u"`), the auxiliary head
is skipped completely, adding zero inference parameters or FLOPs.

### 3. Flexible Classifier-Free Guidance (Section 3.2 & Appendix A Eq. 17)

`ConditionalImprovedMeanFlowModel` supports four guidance regimes
(`cfg_mode` in `{"none", "fixed", "scale", "interval"}`):

1. Sample guidance scale $\omega \sim p(\omega) \propto \omega^{-\beta}$ on
   $[w_{\min}, w_{\max}]$ (`flow_map.sample_cfg_scale`, exact cancellation-free
   log-space inverse CDF with smooth Taylor continuation at $\beta = 1$) and
   interval $(t_{\min}, t_{\max})$ (`flow_map.sample_cfg_interval`).
2. Drop class label $c \in \lbrace 0, \dots, C - 1 \rbrace$ to the null class
   $\varnothing = C$ with probability $p_{\mathrm{drop}}$
   (`class_dropout_rate`).
3. Condition the network on $(\tilde{c}, \Omega)$ where
   $\Omega = (\omega, t_{\min}, t_{\max})$, set effective scale
   $\omega_{\mathrm{eff}} = \omega$ when
   $(d = 0) \land (t_{\min} \le t \le t_{\max})$ else $1.0$, and form the
   contraction-stable ($\rho = 1 - 1/\omega_{\mathrm{eff}} \in [0, 1)$)
   Appendix A Eq. 17 target (`flow_map.compute_guided_velocity_target`):

$$
v_{\mathrm{tgt}} = \operatorname{sg}\left(
  v_{\mathrm{base}} +
  \left(1 - \frac{1}{\omega_{\mathrm{eff}}}\right)
    \big(v_\theta(x_t, t \mid \tilde{c}, \Omega) -
         v_\theta(x_t, t \mid \varnothing, \Omega)\big)
\right).
$$

At equilibrium,
$v_\theta(x_t, t \mid \varnothing, \Omega) = v^*(x_t, t \mid \varnothing)$ and

$$
v_\theta(x_t, t \mid c, \Omega) =
\omega_{\mathrm{eff}}\thinspace v^*(x_t, t \mid c) +
(1 - \omega_{\mathrm{eff}})\thinspace v^*(x_t, t \mid \varnothing),
$$

while setting $\omega = 1.0$ recovers exact unguided sampling from the same
checkpoint.

### 4. Architecture & Table 2/4 Presets

`InContextMeanFlowTransformer` removes adaLN-zero modulation in favor of
prepended multi-token in-context conditioning over
$(t, t - s, c, \omega, t_{\min}, t_{\max})$ (8 unconditional tokens; 28 full
CFG-interval tokens), pre-RMSNorm blocks with axial 1D/2D RoPE on spatial
tokens only, QK-RMSNorm, SwiGLU FFN, `float32` JVP-exact scaled dot-product
attention, and Appendix A zero-initialized per-channel residual scales
$\gamma \in \mathbb{R}^D$ (`residual_scale_init=0.0`).

| Preset (`imf_transformer_config`) | `hidden_size` | `depth` | `num_heads` | `aux_head_depth` | Table 4 `cfg_scale_beta` |
| :--- | :-: | :-: | :-: | :-: | :-: |
| `"B"` (`iMF-B`) | 768 | 12 | 12 | 8 | 1.0 |
| `"M"` (`iMF-M`) | 768 | 24 | 12 | 8 | 2.0 |
| `"L"` (`iMF-L`) | 1024 | 32 | 16 | 8 | 2.0 |
| `"XL"` (`iMF-XL`) | 1024 | 48 | 16 | 8 | 2.0 |

`DualHeadMeanFlowUNet` provides a 1D/2D convolutional counterpart compatible
with `flow_map.MergeGuidanceEmbCond`.

### 5. Sign & Time-Ordering Conventions

| Quantity | Convention in `improved_mean_flow` | Paper [1] Symbol |
| :--- | :--- | :--- |
| Data / Prior | $x_0 \sim p_{\mathrm{data}}$, $x_1 \sim p_{\mathrm{prior}}$ | $x$, $\epsilon$ |
| Interpolant | $x_t = (1 - t) x_0 + t x_1$ (`RectifiedFlow`) | $z_t = (1 - t) x + t \epsilon$ |
| Time arguments | `(x_s, t, s, cond)` with $t \ge s$ | $(z_t, t, r)$ with $t \ge r$ |
| 1-NFE generation | $x_0 = x_1 - u_\theta(x_1, 1, 0 \mid c, \Omega)$ | $x = e - u_\theta(e, 1, 0 \mid c, \Omega)$ |
| $K$-step generation | `flow_map.sample_backward_mean_flow` ($1 = \tau_0 > \dots > \tau_K = 0$) | Backward multi-step update |

## Code Map

| File | Role |
| :--- | :--- |
| `backbones.py` | `InContextMeanFlowTransformer`, `imf_transformer_config`, `DualHeadMeanFlowUNet`. |
| `models.py` | `ImprovedMeanFlowModel`, `ConditionalImprovedMeanFlowModel`. |
| `trainers.py` | `TrainState`, `ImprovedMeanFlowTrainer`, `DistributedImprovedMeanFlowTrainer`. |
| `plot_figures.py` | `PlotSamples` TensorBoard callback for 1D and 2D fields. |
| `configs/config_1d.py` | `ml_collections.ConfigDict` for the 1-D class-conditional iMF demo. |
| `colab/demo_improved_mean_flow.ipynb` | Self-contained CPU demo comparing MF vs. iMF JVP variance and 1-NFE CFG sweeps. |

## Quick Start

```python
import jax
import jax.numpy as jnp
import optax
from swirl_dynamics.lib import flow_map
from swirl_dynamics.projects.distillation.improved_mean_flow import backbones
from swirl_dynamics.projects.distillation.improved_mean_flow import models
from swirl_dynamics.projects.distillation.improved_mean_flow import trainers

net = backbones.InContextMeanFlowTransformer(
    out_channels=1,
    hidden_size=32,
    depth=3,
    num_heads=4,
    patch_size=4,
    aux_head_depth=1,
    num_classes=3,
    use_cfg_conditioning=True,
    use_interval_conditioning=True,
)
model = models.ConditionalImprovedMeanFlowModel(
    input_shape=(32, 1),
    mean_flow_model=net,
    cond_shape={"label": ()},
    num_classes=3,
    cfg_mode="interval",
    use_aux_v_head=True,
)
trainer = trainers.ImprovedMeanFlowTrainer(
    ema_decay=0.99,
    model=model,
    rng=jax.random.PRNGKey(0),
    optimizer=optax.adam(1e-3),
)
# After training, build a 1-NFE inference callable with runtime CFG controls:
infer_fn = trainers.ImprovedMeanFlowTrainer.inference_fn_from_state_dict(
    trainer.train_state,
    net,
    use_ema=True,
    cfg_mode="interval",
    use_aux_v_head=True,
)
x_1 = jax.random.normal(jax.random.PRNGKey(1), (8, 32, 1))
cond = {"label": jnp.zeros((8,), dtype=jnp.int32)}
x_0_hat = flow_map.sample_backward_mean_flow(
    lambda x, t, s, c: infer_fn(
        x, t, s, c, cfg_scale=2.0, cfg_interval=(0.2, 0.8)
    ),
    x_1,
    num_steps=1,
    cond=cond,
)
```

## Interactive Demo

Open
[`colab/demo_improved_mean_flow.ipynb`](colab/demo_improved_mean_flow.ipynb)
for a self-contained CPU notebook demonstrating MF vs. iMF compound-velocity
JVP variance elimination, auxiliary-head training with
`InContextMeanFlowTransformer`, and 1-NFE CFG scale and interval sweeps.

## References

1. Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, and
   Kaiming He. *Improved Mean Flows: On the Challenges of Fastforward
   Generative Models*. CVPR 2026 (`arXiv:2512.02012`).
2. Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He.
   *Mean Flows for One-step Generative Modeling*. NeurIPS 2025
   (`arXiv:2505.13447`).
3. Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. *Flow Map
   Matching with Stochastic Interpolants: A Mathematical Framework for
   Consistency Models*. 2024 (`arXiv:2406.07507`).
4. Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. *How to
   Build a Consistency Model: Learning Flow Maps via Self-Distillation*. 2025
   (`arXiv:2505.18825`).
