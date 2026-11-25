# Flow Map Matching & Lagrangian Self-Distillation

JAX/Flax implementation of **Flow Map Matching (FMM)** [1] and teacher-free
**Lagrangian Self-Distillation (LSD)** [2] for fast few-step and one-step
transport along stochastic interpolant probability-flow ODEs.

Within `swirl_dynamics.projects.distillation`, this package sits alongside
`../mean_flow` [3] and `../improved_mean_flow` [4]. All core two-time backbones,
flow-map models, samplers, and EMA trainers have graduated to
`swirl_dynamics.lib.flow_map`; the Python modules in this directory are
backwards-compatible re-export shims.

## Mathematical Formulation & Conventions

Let $x_0 \sim \mu_0$ and $x_1 \sim \mu_1$ be coupled by a stochastic
interpolant $x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z$
(`lib.flow_matching`) whose probability-flow ODE is
$\frac{dx_t}{dt} = v(x_t, t)$ for $t \in [0, 1]$. The **flow map** $X_{s, t}$
($0 \le s \le t \le 1$) is the two-time pushforward operator satisfying

$$
X_{s, t}(x_s) = x_t, \qquad X_{s, s}(x) = x, \qquad
\partial_t X_{s, t}(x_s) = v\big(X_{s, t}(x_s), t\big).
$$

### 1. Exact Boundary-Condition Parameterization (`RescaledFlowMapUNet`)

To enforce $X_{s, s}(x_s) = x_s$ identically for all parameters $\theta$,
`flow_map.RescaledFlowMapUNet` wraps `flow_map.FlowMapUNet` with an affine
boundary skip:

$$
X_{s, t}^\theta(x_s) = \big(1 - (t - s)\big)\thinspace x_s +
(t - s)\thinspace s_\theta(x_s, t, s).
$$

### 2. Teacher-Based Lagrangian Distillation (`LagrangianFlowMapModel`)

Given a pre-trained frozen teacher velocity field $v_\phi(x, t)$ (`flow_model`,
`params_flow`), `LagrangianFlowMapModel` and `ConditionalLagrangianFlowMapModel`
differentiate $X_{s, t}^\theta(x_s)$ with respect to $t$ via forward-mode
`jax.jvp` along tangent $(0, 1, 0)$ and minimize the Lagrangian residual [1]:

$$
\mathcal{L}_{\mathrm{LFM}}(\theta) = \mathbb{E}_{s \le t, x_s}\Big[
  \big\lvert \partial_t X_{s, t}^\theta(x_s) -
  \operatorname{sg}\big(v_\phi(X_{s, t}^\theta(x_s), t)\big) \big\rvert
\Big].
$$

### 3. Teacher-Free Lagrangian Self-Distillation (`LagrangianSelfDistilledFlowMapModel`)

`LagrangianSelfDistilledFlowMapModel` (and its conditional subclass) eliminates
the separate teacher by parameterizing the flow map through an average-velocity
backbone $u_\theta(x_s, t, s)$ (`RescaledMeanFlowUNet`) [2]:

$$
X_{s, t}^\theta(x_s) = x_s + (t - s)\thinspace u_\theta(x_s, t, s), \qquad
\partial_t X_{s, t}^\theta(x_s)\big|_{s = t} = u_\theta(x_t, t, t).
$$

It minimizes a convex combination (`weight_v` in $[0, 1]$) of instantaneous
velocity matching at the diagonal $s = t$ and Lagrangian self-consistency:

$$
\mathcal{L}_{\mathrm{LSD}}(\theta) =
w_v\thinspace \mathbb{E}\big[\lVert u_\theta(x_t, t, t) - \dot{x}_t \rVert_2^2\big] +
(1 - w_v)\thinspace \mathbb{E}\big[
  \lVert \partial_t X_{s, t}^\theta(x_s) - u_\theta(X_{s, t}^\theta(x_s), t, t) \rVert_2^2
\big],
$$

with optional `stop_gradient_target` and adaptive loss weighting
(`use_adaptive_norm`, `per_sample_adaptive_norm`).

### 4. Multi-Step Forward Composition

By the ODE semigroup property $X_{s, t} = X_{u, t} \circ X_{s, u}$, any
ascending partition $0 = \tau_0 < \tau_1 < \dots < \tau_K = 1$ yields a
$K$-NFE forward sampler (`flow_map.sample_forward_flow_map`):

$$
x^{(k + 1)} = X_{\tau_k, \tau_{k + 1}}^\theta\big(x^{(k)}\big), \qquad
k = 0, \dots, K - 1.
$$

## Code Map

| File | Role |
| :--- | :--- |
| `backbones.py` | Re-export shim for `FlowMapUNet`, `FourierEmbedding`, and time samplers (`lib.flow_map`). |
| `models.py` | Re-export shim for `[Conditional]Lagrangian[SelfDistilled]FlowMapModel` and rescaled UNets. |
| `trainers.py` | Re-export shim for `FlowMapTrainState` and single/distributed EMA trainers (`lib.flow_map`). |
| `plot_figures.py` | `PlotSamples` TensorBoard callback for multi-step generated fields. |
| `configs/config_1d.py` | `ml_collections.ConfigDict` for the 1-D teacher and self-distillation demo. |
| `colab/demo_flow_map_matching.ipynb` | Fast CPU demo comparing teacher distillation and teacher-free LSD at 1/2/4 NFE. |
| `colabs/flow_map_distillation_demo.ipynb` | Extended conditional MNIST flow-map distillation tutorial notebook. |

## Quick Start

```python
import jax
import optax
from swirl_dynamics.lib import flow_map
from swirl_dynamics.lib import flow_matching

net = flow_map.RescaledMeanFlowUNet(
    out_channels=1,
    num_channels=(16, 32),
    downsample_ratio=(2, 2),
    num_blocks=1,
    use_attention=False,
)
lsd_model = flow_map.LagrangianSelfDistilledFlowMapModel(
    input_shape=(32, 1),
    mean_flow_model=net,
    interpolant=flow_matching.RectifiedFlow(),
    weight_v=0.5,
)
trainer = flow_map.LagrangianSelfDistilledFlowMapTrainer(
    ema_decay=0.99,
    model=lsd_model,
    rng=jax.random.PRNGKey(0),
    optimizer=optax.adam(1e-3),
)
```

## Interactive Demos

* **Self-contained CPU demo**:
  [`colab/demo_flow_map_matching.ipynb`](colab/demo_flow_map_matching.ipynb)
  trains a lightweight `lib.flow_matching` teacher on a 1-D periodic field,
  distills it via `LagrangianFlowMapModel`, and compares against teacher-free
  `LagrangianSelfDistilledFlowMapModel` at 1, 2, and 4 NFE versus a 32-step
  Euler ODE baseline.
* **MNIST distillation tutorial**:
  [`colabs/flow_map_distillation_demo.ipynb`](colabs/flow_map_distillation_demo.ipynb).

## References

1. Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. *Flow Map
   Matching with Stochastic Interpolants: A Mathematical Framework for
   Consistency Models*. 2024 (`arXiv:2406.07507`).
2. Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. *How to
   Build a Consistency Model: Learning Flow Maps via Self-Distillation*. 2025
   (`arXiv:2505.18825`).
3. Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He.
   *Mean Flows for One-step Generative Modeling*. NeurIPS 2025
   (`arXiv:2505.13447`).
4. Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, and
   Kaiming He. *Improved Mean Flows: On the Challenges of Fastforward
   Generative Models*. CVPR 2026 (`arXiv:2512.02012`).
