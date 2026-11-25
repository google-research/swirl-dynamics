# Stochastic Interpolants Debiasing

Unpaired distribution translation and generative modeling via **Stochastic
Interpolants** [1], implemented in JAX/Flax. Stochastic interpolants unify
deterministic probability-flow ODEs (flow matching / rectified flow) and
stochastic diffusion bridges under a single time-indexed bridge between an
arbitrary source distribution $\mu_0$ and target distribution $\mu_1$.

Within `swirl_dynamics.projects.debiasing`, this package sits alongside
`../rectified_flow`, `../optimal_transport`, and `../cycle_gan`. Its core
interpolant schedules, loss functions, and single-time models have graduated to
`swirl_dynamics.lib.flow_matching`, and `MergeCategoricalEmbCond` has graduated
to `swirl_dynamics.lib.flow_map`. All legacy module paths in this folder remain
available as backwards-compatible re-export shims.

## Mathematical Formulation & Conventions

For independent draws $x_0 \sim \mu_0$, $x_1 \sim \mu_1$, and latent Gaussian
noise $z \sim \mathcal{N}(0, I)$, the stochastic interpolant at time
$t \in [0, 1]$ is

$$
x_t = \alpha(t)\thinspace x_0 + \beta(t)\thinspace x_1 + \gamma(t)\thinspace z,
$$

with boundary conditions $\alpha(0) = \beta(1) = 1$,
$\alpha(1) = \beta(0) = \gamma(0) = \gamma(1) = 0$.

### Interpolant Schedules (`swirl_dynamics.lib.flow_matching.interpolants`)

| Class | $\alpha(t)$ | $\beta(t)$ | $\gamma(t)$ | Mode |
| :--- | :--- | :--- | :--- | :--- |
| `RectifiedFlow` | $1 - t$ | $t$ | $0$ | Deterministic |
| `TrigonometricInterpolant` | $\cos(\pi t / 2)$ | $\sin(\pi t / 2)$ | $0$ | Deterministic |
| `LinearInterpolant(sigma)` | $1 - t$ | $t$ | $\sigma\sqrt{2t(1 - t)}$ | Stochastic |
| `LinearInterpolantSinusoidalNoise(sigma)` | $1 - t$ | $t$ | $\sigma\sin^2(\pi t)$ | Stochastic (smooth $\dot{\gamma}$) |

### Training Objectives (`swirl_dynamics.lib.flow_matching.losses`)

* **Velocity matching (`velocity_loss`)**: Regresses $v_\theta(x_t, t)$ onto
  $\dot{x}_t = \dot{\alpha}(t) x_0 + \dot{\beta}(t) x_1 + \dot{\gamma}(t) z$:
  $$
  \mathcal{L}_v(\theta) = \mathbb{E}_{t, x_0, x_1, z}\Big[
    \lVert v_\theta(x_t, t) - \dot{x}_t \rVert_2^2
  \Big].
  $$
* **Score matching (`score_loss`)**: Regresses $s_\theta(x_t, t)$ onto
  $-\gamma(t)^{-1} z$ when $\gamma(t) > 0$.
* **Denoising (`denoising_loss`)**: Regresses $\eta_\theta(x_t, t)$ directly
  onto $z$, recovering the score via
  $s_\theta(x_t, t) = -\eta_\theta(x_t, t) / \gamma(t)$ without a
  $1 / \gamma(t)$ singularity in the training target.
* **Fair CRPS (`StochasticInterpolantCRPSModel`)**: Trains a stochastic
  velocity network with a two-member fair Continuous Ranked Probability Score
  loss.

### Probability-Flow ODE vs. Forward SDE Sampling

The marginal density $p_t(x)$ of $x_t$ solves both the deterministic
probability-flow ODE and a family of forward SDEs parameterized by any
non-negative diffusion schedule $\epsilon(t) \ge 0$ [1]:

$$
\frac{d x_t}{dt} = v_\theta(x_t, t), \qquad
d x_t = \Big(v_\theta(x_t, t) + \epsilon(t)\thinspace s_\theta(x_t, t)\Big)\thinspace dt +
\sqrt{2\epsilon(t)}\thinspace dW_t.
$$

Setting $\epsilon(t) = c\thinspace \gamma(t)$ with a denoiser head
$\eta_\theta \approx \mathbb{E}[z \mid x_t]$ cancels the $1 / \gamma(t)$ factor
cleanly, yielding drift $v_\theta(x_t, t) - c\thinspace \eta_\theta(x_t, t)$ and
diffusion $\sqrt{2 c\thinspace \gamma(t)}\thinspace dW_t$.

## Code Map

| File | Role |
| :--- | :--- |
| `interpolants.py` | Re-export shim to `swirl_dynamics.lib.flow_matching.interpolants`. |
| `losses.py` | Re-export shim to `swirl_dynamics.lib.flow_matching.losses`. |
| `models.py` | Re-export shim to `swirl_dynamics.lib.flow_matching.models`. |
| `backbones.py` | Single-time categorical `UNet` and `MergeCategoricalEmbCond` shim (`lib.flow_map`). |
| `trainers.py` | `StochasticInterpolantTrainer`, `StochasticInterpolantFlowScoreTrainer`, and distributed variants. |
| `colab/demo_stochastic_interpolants_simple.ipynb` | Fast CPU demo contrasting ODE vs. flow+score SDE debiasing on a 1-D bimodal field. |
| `colabs/demo_stochastic_interpolants*.ipynb` | Extended tutorial notebooks (rectified flow, denoising, and two-sided schedules). |

## Quick Start

```python
import jax
import optax
from swirl_dynamics.lib import flow_matching
from swirl_dynamics.projects.debiasing.stochastic_interpolants import trainers

net = flow_matching.RescaledUnet(
    out_channels=1,
    num_channels=(16, 32),
    downsample_ratio=(2, 2),
    num_blocks=1,
    use_attention=False,
)
model = flow_matching.StochasticInterpolantModel(
    input_shape=(32, 1),
    flow_model=net,
    interpolant=flow_matching.LinearInterpolantSinusoidalNoise(sigma=0.25),
    loss_stochastic_interpolant=flow_matching.velocity_loss,
)
trainer = trainers.StochasticInterpolantTrainer(
    ema_decay=0.99,
    model=model,
    rng=jax.random.PRNGKey(0),
    optimizer=optax.adam(1e-3),
)
```

## Interactive Demos

* **Self-contained CPU demo**:
  [`colab/demo_stochastic_interpolants_simple.ipynb`](colab/demo_stochastic_interpolants_simple.ipynb)
  compares deterministic probability-flow ODE debiasing against joint
  velocity+denoiser SDE sampling on a 1-D unpaired bimodal periodic field.
* **Extended notebooks**:
  [`colabs/demo_stochastic_interpolants.ipynb`](colabs/demo_stochastic_interpolants.ipynb),
  [`colabs/demo_stochastic_interpolants_denoising.ipynb`](colabs/demo_stochastic_interpolants_denoising.ipynb),
  and
  [`colabs/demo_stochastic_interpolants_two_sided.ipynb`](colabs/demo_stochastic_interpolants_two_sided.ipynb).

## References

1. Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. *Stochastic
   Interpolants: A Unifying Framework for Flows and Diffusions*. 2023
   (`arXiv:2303.08797`).
2. Xingchao Liu, Chengyue Gong, and Qiang Liu. *Flow Straight and Fast:
   Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023
   (`arXiv:2209.03003`).
3. Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt
   Le. *Flow Matching for Generative Modeling*. ICLR 2023 (`arXiv:2210.02747`).
