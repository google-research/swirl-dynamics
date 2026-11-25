# Mean Flows for One-Step Generative Modeling

JAX/Flax implementation of **Mean Flows** [1] for from-scratch one-step (1-NFE)
and few-step generative modeling and teacher distillation.

Within `swirl_dynamics.projects.distillation`, this package sits alongside
`../flow_map_matching` [2, 3] and `../improved_mean_flow` [4] (which extends
MeanFlow with a variance-free `v`-loss, auxiliary `v`-head, and flexible CFG).
All core mean-flow models, time samplers, rescaled UNet backbones, and EMA
trainers have graduated to `swirl_dynamics.lib.flow_map`; the modules in this
folder are backwards-compatible re-export shims.

## Mathematical Formulation & Conventions

Let $x_0 \sim p_{\mathrm{data}}$ (`batch["x_0"]`) and
$x_1 \sim p_{\mathrm{prior}}$ (`batch["x_1"]`) be connected by the linear
interpolant $x_t = (1 - t) x_0 + t x_1$ (`flow_matching.RectifiedFlow`) with
instantaneous velocity $v(x_t, t) = \mathbb{E}[x_1 - x_0 \mid x_t]$ (or a
frozen teacher $v_\phi(x_t, t)$ when `flow_model` is provided).

### 1. Average Velocity & the MeanFlow Identity

For ordered times $t \ge s \in [0, 1]$ (`s` denotes $r$ in [1]), the average
velocity along the probability-flow ODE
$\frac{dx_\tau}{d\tau} = v(x_\tau, \tau)$ is defined as

$$
u(x_t, t, s) = \frac{1}{t - s} \int_s^t v(x_\tau, \tau)\thinspace d\tau.
$$

Differentiating
$(t - s) u(x_t, t, s) = \int_s^t v(x_\tau, \tau)\thinspace d\tau$ with respect
to $t$ yields the **MeanFlow identity** [1]:

$$
u(x_t, t, s) = v(x_t, t) - (t - s)\frac{d}{dt} u(x_t, t, s), \qquad
\frac{d}{dt} u(x_t, t, s) = \partial_x u(x_t, t, s)\thinspace v(x_t, t) +
\partial_t u(x_t, t, s).
$$

`MeanFlowModel` and `ConditionalMeanFlowModel` compute $\frac{d}{dt} u_\theta$
via a single forward-mode `jax.jvp` along tangent $(v_t, 1, 0)$, form the
stop-gradient target
$u_{\mathrm{tgt}} = \operatorname{sg}(v_t - (t - s)\frac{d}{dt} u_\theta)$,
and minimize the adaptive per-sample weighted MSE
(`flow_map.adaptive_weighted_loss`):

$$
\mathcal{L}_{\mathrm{MF}}(\theta) = \mathbb{E}_b\left[
  \frac{\bar{e}_b^2}{\operatorname{sg}\big((\bar{e}_b^2 + c)^p\big)}
\right], \qquad
\bar{e}_b^2 = \mathrm{mean}_{\mathrm{spatial,ch}}\big((u_\theta - u_{\mathrm{tgt}})^2\big).
$$

### 2. Symmetric $(t, s)$ MeanFlow (`ConditionalSymmetricMeanFlowModel`)

Differentiating
$(t - s) u(x_t, t, s) = \int_s^t v(x_\tau, \tau)\thinspace d\tau$ with respect
to the lower endpoint $s$ gives the complementary identity
$u(x_t, t, s) = v(x_s, s) + (t - s)\thinspace \partial_s u(x_t, t, s)$.
`ConditionalSymmetricMeanFlowModel` evaluates both directional derivatives in a
single primal pass using `jax.linearize` and averages the per-sample squared
residuals against $u_{\mathrm{tgt}, t}$ and $u_{\mathrm{tgt}, s}$.

### 3. Time Sampling & Backward Multi-Step Generation

`flow_map.time_sampler_mean_flow` draws $(t, s)$ with $t \ge s$ (optionally
from `flow_map.logit_normal_dist`), setting $s = t$ with probability
`diagonal_sampling_rate` (default `0.25`) to supervise the boundary condition
$u_\theta(x_t, t, t) = v(x_t, t)$.

At inference, samples flow **backward** from $t = 1$
($x_1 \sim p_{\mathrm{prior}}$) to $t = 0$ ($x_0 \sim p_{\mathrm{data}}$) using
`flow_map.sample_backward_mean_flow` over a descending partition
$1 = \tau_0 > \tau_1 > \dots > \tau_K = 0$:

$$
x^{(k + 1)} = x^{(k)} - (\tau_k - \tau_{k + 1})\thinspace
u_\theta\big(x^{(k)}, \tau_k, \tau_{k + 1} \mid c\big), \qquad
x_0 = x_1 - u_\theta(x_1, 1, 0 \mid c) \text{ for } K = 1.
$$

### 4. Sign & Direction Convention vs. `flow_map_matching`

| Package | Transport Direction | 1-NFE Formula | Multi-Step Helper |
| :--- | :--- | :--- | :--- |
| `flow_map_matching` | Forward $x_0 \to x_1$ ($0 \to 1$) | $x_1 = X_{0, 1}(x_0)$ or $x_0 + u_\theta(x_0, 1, 0)$ | `flow_map.sample_forward_flow_map` |
| `mean_flow` | Backward $x_1 \to x_0$ ($1 \to 0$) | $x_0 = x_1 - u_\theta(x_1, 1, 0 \mid c)$ | `flow_map.sample_backward_mean_flow` |

## Code Map

| File | Role |
| :--- | :--- |
| `models.py` | Re-export shim for `[Conditional][Symmetric]MeanFlowModel`, `RescaledMeanFlowUNet`, and samplers (`lib.flow_map`). |
| `trainers.py` | Re-export shim for `MeanFlowTrainState`, `MeanFlowTrainer`, and `DistributedMeanFlowTrainer` (`lib.flow_map`). |
| `plot_figures.py` | `PlotSamples` TensorBoard callback for generated 2-D samples. |
| `configs/config_1d.py` | `ml_collections.ConfigDict` for the 1-D class-conditional MeanFlow demo. |
| `colab/demo_mean_flow.ipynb` | Fast CPU demo training class-conditional MeanFlow from scratch (1/2/4 NFE). |
| `colabs/mean_flow_demo_mnist.ipynb` | Extended conditional MNIST MeanFlow tutorial notebook. |

## Quick Start

```python
import jax
import jax.numpy as jnp
import optax
from swirl_dynamics.lib import flow_map
from swirl_dynamics.lib import flow_matching

net = flow_map.RescaledMeanFlowUNet(
    out_channels=1,
    num_channels=(16, 32),
    downsample_ratio=(2, 2),
    num_blocks=1,
    use_attention=False,
    cond_embed_fn=flow_map.MergeCategoricalEmbCond,
    cond_embed_kwargs={
        "cond_key": "emb:label",
        "num_classes": 3,
        "features_embedding": 16,
    },
)
model = flow_map.ConditionalMeanFlowModel(
    input_shape=(32, 1),
    cond_shape={"emb:label": ()},
    mean_flow_model=net,
    interpolant=flow_matching.RectifiedFlow(),
)
trainer = flow_map.MeanFlowTrainer(
    ema_decay=0.99,
    model=model,
    rng=jax.random.PRNGKey(0),
    optimizer=optax.adam(1e-3),
)
infer_fn = flow_map.MeanFlowTrainer.inference_fn_from_state_dict(
    trainer.train_state, net, use_ema=True
)
x_1 = jax.random.normal(jax.random.PRNGKey(1), (8, 32, 1))
cond = {"emb:label": jnp.zeros((8,), dtype=jnp.int32)}
x_0_hat = flow_map.sample_backward_mean_flow(
    infer_fn, x_1, num_steps=1, cond=cond
)
```

## Interactive Demos

* **Self-contained CPU demo**:
  [`colab/demo_mean_flow.ipynb`](colab/demo_mean_flow.ipynb) trains a
  class-conditional 1-D bimodal waveform generator from scratch and compares
  1-, 2-, and 4-NFE backward sampling.
* **MNIST tutorial**:
  [`colabs/mean_flow_demo_mnist.ipynb`](colabs/mean_flow_demo_mnist.ipynb).

## References

1. Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He.
   *Mean Flows for One-step Generative Modeling*. NeurIPS 2025
   (`arXiv:2505.13447`).
2. Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. *Flow Map
   Matching with Stochastic Interpolants: A Mathematical Framework for
   Consistency Models*. 2024 (`arXiv:2406.07507`).
3. Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. *How to
   Build a Consistency Model: Learning Flow Maps via Self-Distillation*. 2025
   (`arXiv:2505.18825`).
4. Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, and
   Kaiming He. *Improved Mean Flows: On the Challenges of Fastforward
   Generative Models*. CVPR 2026 (`arXiv:2512.02012`).
