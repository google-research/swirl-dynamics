# Entropic Optimal Transport Debiasing

Unpaired statistical debiasing of physical fields via entropic optimal transport
(OT), implemented in JAX. Given empirical snapshots from a biased or coarse
numerical solver (domain A) and an unpaired reference distribution (domain B),
this package computes the entropic optimal coupling with a stabilized log-domain
Sinkhorn algorithm (`SinkhornSolver`) [1, 7] and maps out-of-sample snapshots to
the reference distribution via the entropic barycentric projection
(`transport_fn`) of Pooladian and Niles-Weed (2021) [2]. An optional companion
module (`divergence.py`) provides the symmetric Sinkhorn divergence [4], the
centered (debiased) entropic transport map [5], and a Jacobian eigenvalue
diagnostic for auditing local monotonicity [5].

Within `swirl_dynamics.projects.debiasing`, this package sits alongside the
neural unpaired translation baselines in `../rectified_flow` and `../cycle_gan`.
It implements both a standalone non-parametric distribution-matching baseline
and the coarse-scale statistical debiasing stage of the two-step downscaling
framework of Wan et al. (2023) [8], evaluated on the shared 1-D
Kuramoto-Sivashinsky (KS) benchmark comparing a low-order finite-volume solver
against a high-resolution pseudo-spectral solver.

## Mathematical Formulation & Conventions

Let $X = (x_1, \dots, x_n)^\top \in \mathbb{R}^{n \times d}$ and $Y = (y_1,
\dots, y_m)^\top \in \mathbb{R}^{m \times d}$ denote flattened source and target
point clouds with probability weights $a \in \Delta_n$ ($\sum_i a_i = 1$,
defaulting to $a_i = 1/n$) and $b \in \Delta_m$ ($\sum_j b_j = 1$, defaulting to
$b_j = 1/m$). `SinkhornSolver` constructs the pairwise cost matrix $C \in
\mathbb{R}^{n \times m}$ by summing a user-supplied elementwise `metric` across
the feature dimension:

$$
C_{ij} = \sum_{k=1}^d \texttt{metric}(x_{ik} - y_{jk}).
$$

For the default `metric(t) = jnp.pow(t, 2)`, $C_{ij} = \lVert x_i - y_j
\rVert_2^2$ is the **unscaled** squared Euclidean distance (with **no**
$\tfrac{1}{2}$ prefactor).

### Primal and Dual Entropic OT

Given an entropic regularization parameter $\varepsilon > 0$ and the transport
polytope $U(a, b) = \lbrace P \in \mathbb{R}_+^{n \times m} : P\mathbf{1}_m = a,
P^\top\mathbf{1}_n = b \rbrace$, the primal entropic OT problem [1, 7] is

$$
\mathrm{OT}_\varepsilon(a, b) = \min_{P \in U(a, b)} \langle P, C \rangle + \varepsilon\thinspace\mathrm{KL}(P \Vert a \otimes b),
$$

where $\langle P, C \rangle = \sum_{i, j} P_{ij} C_{ij}$ is the bare transport
cost and $\mathrm{KL}(P \Vert a \otimes b)$ is the generalized Kullback-Leibler
divergence with respect to the product measure $(a \otimes b)_{ij} = a_i b_j$:

$$
\mathrm{KL}(P \Vert a \otimes b) = \sum_{i=1}^n \sum_{j=1}^m P_{ij}\left(\log\frac{P_{ij}}{a_i b_j} - 1\right) + \left(\sum_{i=1}^n a_i\right)\left(\sum_{j=1}^m b_j\right).
$$

The Fenchel-Rockafellar dual maximizes over Kantorovich potentials $(f, g) \in
\mathbb{R}^n \times \mathbb{R}^m$:

$$
\mathrm{OT}_\varepsilon(a, b) = \max_{f \in \mathbb{R}^n, g \in \mathbb{R}^m} \langle f, a \rangle + \langle g, b \rangle + \varepsilon\left(\left(\sum_{i=1}^n a_i\right)\left(\sum_{j=1}^m b_j\right) - \sum_{i=1}^n \sum_{j=1}^m a_i b_j \exp\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right)\right).
$$

### Scaling Gauge vs. Kantorovich Gauge

Two dual potential conventions appear in the literature and in software.

**Scaling-gauge potentials $(u, v)$** (`SinkhornOutput.potentials = (fu, gv)`):
Absorb the log-marginals $\varepsilon \log a_i$ and $\varepsilon \log b_j$
directly into the potentials so that the Gibbs coupling is parameterized without
an explicit $a_i b_j$ prefactor:

$$
P_{ij} = \exp\left(\frac{u_i + v_j - C_{ij}}{\varepsilon}\right).
$$

On zero-weight support entries ($a_i = 0$ or $b_j = 0$ with
`eps_marginal = 0.0`), $u_i = -\infty$ or $v_j = -\infty$, which enforces
$P_{ij} = 0$ identically.

**Kantorovich-gauge potentials $(f, g)$**
(`SinkhornOutput.kantorovich_potentials`): Factor out the reference measure $a
\otimes b$ so that

$$
P_{ij} = a_i b_j \exp\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right).
$$

On positive-support entries ($a_i > 0$, $b_j > 0$), the two gauges are related
by $f_i = u_i - \varepsilon \log a_i$ and $g_j = v_j - \varepsilon \log b_j$. On
zero-weight entries ($a_i = 0$ or $b_j = 0$), where $u_i = -\infty$ or $v_j =
-\infty$, `SinkhornOutput.kantorovich_potentials` evaluates the finite entropic
$c$-transform extension

$$
f_i = -\varepsilon \log \sum_{j : b_j > 0} \exp\left(\frac{v_j - C_{ij}}{\varepsilon}\right), \qquad g_j = -\varepsilon \log \sum_{i : a_i > 0} \exp\left(\frac{u_i - C_{ij}}{\varepsilon}\right)
$$

using double-`jnp.where` masking so that $f$ and $g$ are finite at every index.

### Log-Domain Sinkhorn Updates

`SinkhornSolver` performs alternating dual ascent directly on the scaling-gauge
potentials $(u, v)$ in the log domain using `jax.scipy.special.logsumexp`,
avoiding matrix-vector products with an exponentiated Gibbs kernel $K_{ij} =
\exp(-C_{ij}/\varepsilon)$ that would underflow or overflow at small
$\varepsilon$:

$$
u_i^{(t+1)} = \varepsilon \log a_i - \varepsilon \log \sum_{j=1}^m \exp\left(\frac{v_j^{(t)} - C_{ij}}{\varepsilon}\right), \qquad v_j^{(t+1)} = \varepsilon \log b_j - \varepsilon \log \sum_{i=1}^n \exp\left(\frac{u_i^{(t+1)} - C_{ij}}{\varepsilon}\right).
$$

When `eps_marginal > 0`, the solver replaces $a$ and $b$ with `a + eps_marginal`
and `b + eps_marginal` inside the log-domain updates.

### Conventions Comparison Table

The table below summarizes how the conventions in this package map to Pooladian
and Niles-Weed [2] and to `ott-jax` (`ott`, [9]):

<!-- mdformat off(prevent table wrapping) -->

| Aspect | This package (`swirl_dynamics`) | Pooladian & Niles-Weed [2] | OTT-JAX (`ott`, [9]) |
| :--- | :--- | :--- | :--- |
| **Quadratic cost $C(x, y)$** | $C(x, y) = \lVert x - y \rVert_2^2$ (default `metric(t) = t**2`, **no** $\tfrac{1}{2}$ factor) | $c(x, y) = \tfrac{1}{2}\lVert x - y \rVert_2^2$ | `costs.SqEuclidean()` defines $C(x, y) = \lVert x - y \rVert_2^2$ with `scale_cost=1.0` |
| **Regularization $\varepsilon$** | Absolute $\varepsilon > 0$ in `SinkhornSolver`; `experiment.resolve_epsilon` supports `epsilon_scale * median(C)` | Absolute $\varepsilon > 0$ ($\varepsilon_{\text{here}} = 2\thinspace\varepsilon_{\text{[2]}}$ yields the identical coupling $P$) | `PointCloud(x, y, epsilon=...)` defaults to relative (`0.05 * mean_cost`); set `relative_epsilon=False` for absolute $\varepsilon$ |
| **Stored dual potentials** | `potentials = (fu, gv)` in scaling gauge ($u_i = f_i + \varepsilon \log a_i$, $v_j = g_j + \varepsilon \log b_j$); `kantorovich_potentials` returns $(f, g)$ | Kantorovich potentials $(f, g)$ relative to $\mu \otimes \nu$ | `out.f, out.g` stored in scaling gauge (matching `fu, gv`) |
| **Transport cost properties** | `reg_ot_cost` $= \langle P_\varepsilon, C \rangle$; `entropic_ot_cost` = Fenchel-Rockafellar dual value $\langle f, a \rangle + \langle g, b \rangle + \varepsilon(\sum a \sum b - \sum P_\varepsilon)$, equal to $\langle P_\varepsilon, C \rangle + \varepsilon\thinspace\mathrm{KL}(P_\varepsilon \Vert a \otimes b)$ at marginal convergence | $S_\varepsilon(\mu, \nu) = \min_{P \in U(\mu, \nu)} \langle P, c \rangle + \varepsilon\thinspace\mathrm{KL}(P \Vert \mu \otimes \nu)$ | `out.primal_cost` $= \langle P_\varepsilon, C \rangle$; `out.reg_ot_cost` (`kl_reg_cost`) $= \langle P_\varepsilon, C \rangle + \varepsilon\thinspace\mathrm{KL}(P_\varepsilon \Vert a \otimes b)$; `out.ent_reg_cost` $= \langle P_\varepsilon, C \rangle - \varepsilon H(P_\varepsilon)$, equal to `out.kl_reg_cost` $- \varepsilon(H(a) + H(b))$ with $H(p) = -\sum_i p_i \log p_i$ |
| **Out-of-sample potential & map** | `_potential_fn` and `transport_fn` consume scaling-gauge `gv` directly and are exact for arbitrary non-uniform weights $(a, b)$; $T_\varepsilon(x) = x - \tfrac{1}{2}\nabla f_\varepsilon(x)$ for $C = \lVert x - y \rVert_2^2$ | $f_\varepsilon(x) = -\varepsilon \log \sum_j b_j \exp((g_j - \tfrac{1}{2}\lVert x - y_j \rVert_2^2)/\varepsilon)$ using Kantorovich $g_j$; $T_\varepsilon(x) = x - \nabla f_\varepsilon(x)$ | `out.to_dual_potentials()` (`EntropicPotentials`) passes scaling-gauge `out.g` into a $c$-transform that adds $\varepsilon \log b_j$ again (`lse_step`), double-counting $\varepsilon \log b_j$ when target weights `b` are non-uniform |

<!-- mdformat on -->

## Stopping Criteria & Convergence Semantics

`SinkhornSolver` supports two stopping criteria via the `criterion` argument,
evaluated every `inner_iterations` Sinkhorn sweeps (`inner_iterations >= 1`).

**L1 marginal violation (`criterion="marginal"`, default):**

$$
E_{\text{marg}}(u, v) = \lVert P\mathbf{1}_m - a \rVert_1 + \lVert P^\top\mathbf{1}_n - b \rVert_1 \le \tau,
$$

where the row and column marginals $(P\mathbf{1}_m)_i =
\exp(\mathrm{logsumexp}_j((u_i + v_j - C_{ij})/\varepsilon))$ and
$(P^\top\mathbf{1}_n)_j = \exp(\mathrm{logsumexp}_i((u_i + v_j -
C_{ij})/\varepsilon))$ are computed in log space. Because $v^{(t+1)}$ is updated
last in each sweep, the column marginal $P^\top\mathbf{1}_n = b$ is satisfied to
machine precision when `eps_marginal = 0.0`, and $E_{\text{marg}}$ directly
bounds the L1 distance between the coupling's row marginal and $a$ [6, 7].

**Potential increment (`criterion="potentials"`, legacy):**

$$
E_{\text{pot}}(u, v) = \lVert u^{(t)} - u^{(t-1)} \rVert_2 + \lVert v^{(t)} - v^{(t-1)} \rVert_2 \le \tau.
$$

In a Sinkhorn update, the change in $u_i$ across one sweep is

$$
u_i^{(t)} - u_i^{(t-1)} = \varepsilon\left(\log a_i - \log(P^{(t-1)}\mathbf{1}_m)_i\right) \approx -\varepsilon\thinspace\frac{(P^{(t-1)}\mathbf{1}_m)_i - a_i}{a_i}.
$$

Because the potential increment is scaled by $\varepsilon$, stopping when
$\lVert \Delta u \rVert_2 + \lVert \Delta v \rVert_2 \le \tau$ only bounds the
marginal error at the scale $O(\tau / \varepsilon)$ [6, 7]. When $\varepsilon
\ll \mathrm{median}(C)$, `criterion="potentials"` terminates prematurely while
the coupling $P$ still violates the marginal constraints by a large fraction.
`criterion="marginal"` is invariant to this $\varepsilon$ scaling and is
therefore the default.

Setting `inner_iterations > 1` (e.g., `inner_iterations = 10` in
`configs/config_ks_1d.py`) runs multiple Sinkhorn sweeps inside a
`jax.lax.fori_loop` per `jax.lax.while_loop` check, amortizing the cost of the
marginal reduction and device synchronization. Upon termination,
`SinkhornOutput.marginal_error` always reports $E_{\text{marg}}(u, v)$
regardless of which `criterion` was used to stop the loop, and
`SinkhornOutput.converged` is `True` if and only if the active error is finite
and $\le \tau$.

## Out-of-Sample Entropic Map, Sinkhorn Divergence & Debiased Map

### Entropic Barycentric Projection (`SinkhornSolver.transport_fn`)

Once `out = solver(x, y, a, b)` has converged, `solver.transport_fn(out.gv, y)`
returns a callable evaluating the out-of-sample entropic barycentric projection
[2] for any query point $x \in \mathbb{R}^d$ (or batch $X_q \in \mathbb{R}^{B
\times d}$):

$$
T_\varepsilon(x) = \sum_{j=1}^m w_j(x)\thinspace y_j, \qquad w_j(x) = \frac{\exp\big((v_j - c(x, y_j))/\varepsilon\big)}{\sum_{k=1}^m \exp\big((v_k - c(x, y_k))/\varepsilon\big)},
$$

where $c(x, y_j) = \sum_k \texttt{metric}(x_k - y_{jk})$ and $v =
\texttt{out.gv}$ is the target scaling-gauge potential:

*   **General `metric` validity**: Because `transport_fn` directly evaluates the
    conditional expectation $\mathbb{E}_{P_\varepsilon}[Y \mid X = x]$ via
    `jax.nn.softmax` in log domain, it is valid for any user-supplied `metric`.
*   **Why `weights` must not be passed again**: Because $v_j = g_j + \varepsilon
    \log b_j$ is in the scaling gauge, the target marginal weight $b_j$ is
    already absorbed inside $\exp(v_j / \varepsilon) = b_j \exp(g_j /
    \varepsilon)$ (and masked to $-\infty$ when $b_j = 0$). Passing
    `weights is not None` to `transport_fn` or `_potential_fn` would
    double-weight by $b_j^2$ and raises a `ValueError`.

**Exact $\tfrac{1}{2}$ gradient identity (`_gradient_transport_fn`):** The
out-of-sample entropic dual potential (`_potential_fn`) is

$$
f_\varepsilon(x) = -\varepsilon \log \sum_{j=1}^m \exp\left(\frac{v_j - c(x, y_j)}{\varepsilon}\right).
$$

When the cost is the unscaled squared Euclidean distance $c(x, y) = \lVert x - y
\rVert_2^2$ (equal to $2 \cdot \tfrac{1}{2}\lVert x - y \rVert_2^2$),
differentiating $f_\varepsilon(x)$ with respect to $x$ gives

$$
\nabla f_\varepsilon(x) = \sum_{j=1}^m w_j(x)\thinspace\nabla_x \lVert x - y_j \rVert_2^2 = 2 \sum_{j=1}^m w_j(x)(x - y_j) = 2\big(x - T_\varepsilon(x)\big),
$$

which rearranges to the identity implemented in `_gradient_transport_fn`:

$$
T_\varepsilon(x) = x - \frac{1}{2}\nabla f_\varepsilon(x).
$$

The factor of $\tfrac{1}{2}$ appears because this package uses $C(x, y) = \lVert
x - y \rVert_2^2$ rather than $\tfrac{1}{2}\lVert x - y \rVert_2^2$ as in
Pooladian and Niles-Weed [2].

### Sinkhorn Divergence & Centered Map (`divergence.py`)

Entropic regularization introduces a systematic shrinkage bias: even when two
distributions are identical ($(X, a) = (Y, b)$), the self-transport cost is
positive ($\mathrm{OT}_\varepsilon(a, a) > 0$) and $T_{\varepsilon, x \to x}(q)$
contracts query points toward the center of mass as $\varepsilon$ increases [3,
4, 5]. The opt-in module `divergence.py` provides both the debiased divergence
and the centered transport map.

**Sinkhorn divergence (`sinkhorn_divergence`,
`sinkhorn_divergence_with_potentials`) [4]:**

$$
S_\varepsilon(a, b) = \mathrm{OT}_\varepsilon(a, b) - \frac{1}{2}\mathrm{OT}_\varepsilon(a, a) - \frac{1}{2}\mathrm{OT}_\varepsilon(b, b),
$$

where $\mathrm{OT}_\varepsilon(a, b)$ is `ot_xy.entropic_ot_cost` and the
symmetric self-potentials $u_{xx}$ on $(X, a)$ and $v_{yy}$ on $(Y, b)$ are
computed via the damped symmetric fixed-point iteration [4, Eq. 25]:

$$
u_{xx}^{(t+1)} = \frac{1}{2}\left(u_{xx}^{(t)} + \varepsilon \log a - \varepsilon \log \sum_{j=1}^n \exp\left(\frac{u_{xx,j}^{(t)} - C_{ij}^{XX}}{\varepsilon}\right)\right).
$$

By construction, $S_\varepsilon(a, a) = 0$ identically and $S_\varepsilon(a, b)
\ge 0$ for positive-definite universal kernels [4].

**Centered (debiased) entropic map (`debiased_transport_fn`) [5]:** Formed from
the first variation of $S_\varepsilon(a, b)$, the centered map corrects the
cross-transport map by subtracting the self-transport residual:

$$
T^{\text{deb}}_\varepsilon(q) = q + T_{\varepsilon, x \to y}(q) - T_{\varepsilon, x \to x}(q),
$$

where $T_{\varepsilon, x \to y}$ is `solver.transport_fn(ot_xy.gv, y)` and
$T_{\varepsilon, x \to x}$ is `solver.transport_fn(self_potential_x, x)`. When
$(X, a) = (Y, b)$, $T^{\text{deb}}_\varepsilon(q) = q$ identically for all
$\varepsilon > 0$.

**Non-monotonicity caveat & `jacobian_min_eigenvalue` diagnostic [5]:** As
analyzed by Pooladian, Cuturi, and Niles-Weed (2022) [5], $T_{\varepsilon, x \to
y}$ is the gradient of a convex potential ($\tfrac{1}{2}\lVert q \rVert_2^2 -
\tfrac{1}{2}f_{\varepsilon, x \to y}(q)$ has positive-semidefinite Hessian),
whereas $T^{\text{deb}}_\varepsilon$ is a **difference** of convex gradients:

$$
T^{\text{deb}}_\varepsilon(q) = q - \frac{1}{2}\nabla\big(f_{\varepsilon, x \to y}(q) - f_{\varepsilon, x \to x}(q)\big).
$$

At small $\varepsilon$ or when source and target supports are misaligned, the
symmetrized Jacobian

$$
J_{\text{sym}}(q) = \frac{1}{2}\left(DT^{\text{deb}}_\varepsilon(q) + DT^{\text{deb}}_\varepsilon(q)^\top\right)
$$

can develop negative eigenvalues (loss of cyclical monotonicity / local
folding). Use `divergence.jacobian_min_eigenvalue(map_fn, points)` to compute
$\lambda_{\min}(J_{\text{sym}}(q))$ on evaluation points; values $< 0$ indicate
local non-monotonicity.

## Practical Guidance: Choosing $\varepsilon$, Precision & Memory

### Choosing $\varepsilon$ Relative to $\mathrm{median}(C)$

Because the Gibbs kernel depends on the dimensionless ratio
$C_{ij}/\varepsilon$, $\varepsilon$ should always be scaled to the typical
pairwise cost of the dataset (for example, `epsilon_scale`
$\cdot\thinspace\mathrm{median}(C)$, with `epsilon_scale = 0.01` in
`configs/config_ks_1d.py` via `experiment.resolve_epsilon`).

**Large-$\varepsilon$ regime (entropic shrinkage) [3, 5]:** As $\varepsilon /
\mathrm{median}(C) \gg 1$, the softmax weights flatten toward $w_j(x) \to b_j$
and $T_\varepsilon(x)$ collapses toward the target mean $\sum_j b_j y_j$. Even
for Gaussian measures $\mathcal{N}(0, \sigma_A^2)$ and $\mathcal{N}(0,
\sigma_B^2)$ in 1-D with $C(x, y) = (x - y)^2$, the closed-form entropic map [3]
contracts the positive linear gain $\alpha_\varepsilon$ below the exact Monge
slope $\sigma_B / \sigma_A$:

$$
T_\varepsilon^{\text{Gauss}}(x) = \alpha_\varepsilon x, \qquad \alpha_\varepsilon = \frac{\sqrt{\sigma_A^2\sigma_B^2 + \varepsilon^2/16} - \varepsilon/4}{\sigma_A^2} = \frac{2\sigma_B^2}{\sqrt{4\sigma_A^2\sigma_B^2 + \varepsilon^2/4} + \varepsilon/2} < \frac{\sigma_B}{\sigma_A}.
$$

**Small-$\varepsilon$ regime (approximation bound vs. conditioning) [1, 6]:** On
discrete uniform clouds of size $n$, the primal suboptimality satisfies the
sandwich bound [1, 6]

$$
0 \le \langle P_\varepsilon, C \rangle - W_2^2(a, b) \le \varepsilon \log n,
$$

so bias decreases linearly with $\varepsilon$. However, Sinkhorn iteration
counts grow as $O(\varepsilon^{-1})$ to $O(\varepsilon^{-2})$ [6], and for
$\varepsilon$ much smaller than nearest-neighbor squared distances the
barycentric projection reduces to a piecewise-constant 1-nearest-neighbor
lookup. Values in the range $\varepsilon \in [0.01, 0.1] \times
\mathrm{median}(C)$ typically balance tail fidelity against smooth
interpolation.

### Floating-Point Precision (`jax_enable_x64`)

Importing `sinkhorn` or `divergence` never mutates global JAX configuration at
import time. When `jax.config.jax_enable_x64` is `False`,
`SinkhornSolver.__init__` logs a warning because log-sum-exp reductions and
marginal errors in `float32` lose accuracy when $\varepsilon /
\mathrm{median}(C) \lesssim 10^{-2}$. Enable 64-bit precision explicitly at
program startup:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

In `experiment.fit_ot_map`, setting `config.enable_x64 = True` (the default in
`configs/config_ks_1d.py`) enables `jax_enable_x64` and casts the training and
evaluation snapshots to `jnp.float64`.

### Memory Scaling, Sharding & When to Use `ott-jax`

*   **$O(nm)$ cost matrix footprint**: `SinkhornSolver` materializes the dense
    cost matrix $C \in \mathbb{R}^{n \times m}$ and coupling $P \in
    \mathbb{R}^{n \times m}$. At $n = m = 20{,}000$ in `float64`, a single
    $20{,}000 \times 20{,}000$ array requires $4 \times 10^8 \times 8\text{
    bytes} = 3.2\text{ GB}$ (and multiple such arrays are allocated during solve
    and transport plan evaluation). For the KS benchmark,
    `config.max_train_samples = 4096` subsamples the training set so each
    $4{,}096 \times 4{,}096$ `float64` matrix uses $\approx 134\text{ MB}$.
*   **Multi-device sharding (`sharding`)**: Pass a
    `jax.sharding.NamedSharding(mesh, PartitionSpec("x", None))` to
    `SinkhornSolver(..., sharding=...)` to partition $C$ across devices via
    `jax.lax.with_sharding_constraint`. Note that XLA GSPMD requires the sharded
    matrix dimensions $(n, m)$ to be evenly divisible by the corresponding mesh
    axes; if not, XLA falls back to replication and `SinkhornSolver.__call__`
    logs a warning.
*   **Chunked out-of-sample evaluation (`batch_size`)**: Both
    `SinkhornSolver.transport_fn(..., batch_size=...)` and
    `divergence.debiased_transport_fn(..., batch_size=...)` accept an integer
    `batch_size` (e.g., `eval_batch_size = 256` in `configs/config_ks_1d.py`),
    evaluating out-of-sample queries in chunks via `jax.lax.map` and bounding
    peak intermediate memory to `O(batch_size * m)` instead of $O(B \cdot m)$.
*   **When to use `ott-jax` [9]**: This package provides a transparent,
    self-contained implementation tailored for moderate sample sizes ($n, m
    \lesssim 2 \times 10^4$). For large-scale problems requiring online/lazy
    `PointCloud` cost evaluation without materializing $O(nm)$ matrices,
    low-rank Sinkhorn solvers, or Anderson/momentum acceleration, use `ott-jax`
    (`from ott.geometry import pointcloud`,
    `from ott.solvers.linear import sinkhorn`) [9].

## Code Map, KS Data, Quick Start & Evaluation Metrics

### Code Map

*   `sinkhorn.py`: `SinkhornOutput` (`potentials`, `fu`, `gv`,
    `kantorovich_potentials`, `transport_plan`, `reg_ot_cost`,
    `entropic_ot_cost`, `marginal_error`, `converged`, `num_iterations`,
    `weights`) and `SinkhornSolver` (log-domain Sinkhorn solver, `transport_fn`,
    `_potential_fn`, `_gradient_transport_fn`).
*   `divergence.py`: `SinkhornDivergenceOutput`,
    `sinkhorn_divergence_with_potentials`, `sinkhorn_divergence`,
    `debiased_transport_fn`, and `jacobian_min_eigenvalue`.
*   `experiment.py`: 1-D debiasing pipeline (`Datasets`, `FitResult`,
    `load_datasets`, `synthetic_datasets`, `resolve_epsilon`, `fit_ot_map`,
    `apply_ot_map`, `evaluate`, `run_experiment`).
*   `configs/config_ks_1d.py`: `ml_collections.ConfigDict` for the 1-D
    Kuramoto-Sivashinsky experiment.
*   `main_ot_debiasing.py`: command-line runner (`--config`, `--workdir`,
    `--output_path`).
*   `colab/optimal_transport_sinkhorn.ipynb`: end-to-end interactive notebook
    covering 1-D and 2-D pedagogical examples, Gaussian shrinkage vs. centered
    debiasing, and the 1-D KS benchmark.

### Kuramoto-Sivashinsky (KS) Data

The KS benchmark shares the public dataset
`gs://gresearch/swirl_dynamics/downscaling/KS_finite_volumes_vs_pseudo_spectral.hdf5`
with `../cycle_gan` and `../rectified_flow`. It contains trajectories of a
low-order finite-volume solver (`u_fd`, domain A) and a $4\times$
higher-resolution pseudo-spectral solver (`u_sp`, domain B).
`experiment.load_datasets` calls `cycle_gan.dataloaders.load_ks_snapshots` to
subsample `u_sp` in space by `downsample_factor = 4` (yielding snapshots of
shape `(48, 1)`), holds out the last `num_test_trajectories = 16` trajectories
for evaluation, and flattens the training snapshots to shape `(n, 48)`. Leaving
`config.dataset_path = ""` generates a deterministic synthetic dataset
(`experiment.synthetic_datasets`) for fast smoke tests.

### Quick Start

Download the KS dataset with `gcloud storage cp` and run the experiment from the
command line:

```bash
gcloud storage cp gs://gresearch/swirl_dynamics/downscaling/KS_finite_volumes_vs_pseudo_spectral.hdf5 /tmp/ks.hdf5
python main_ot_debiasing.py --config=configs/config_ks_1d.py --workdir=/tmp/ot_ks \
  --config.dataset_path=/tmp/ks.hdf5
```

Standalone Python usage with `SinkhornSolver` and `divergence`:

```python
import jax
import jax.numpy as jnp
from swirl_dynamics.projects.debiasing.optimal_transport import divergence
from swirl_dynamics.projects.debiasing.optimal_transport import sinkhorn

jax.config.update("jax_enable_x64", True)

key_x, key_y, key_q = jax.random.split(jax.random.PRNGKey(0), 3)
x = jax.random.normal(key_x, (256, 8), dtype=jnp.float64) + 1.0
y = 1.5 * jax.random.normal(key_y, (256, 8), dtype=jnp.float64)
queries = jax.random.normal(key_q, (64, 8), dtype=jnp.float64) + 1.0

solver = sinkhorn.SinkhornSolver(
    epsilon=1.0,
    num_iterations=1000,
    threshold=1e-4,
    criterion="marginal",
    inner_iterations=10,
)
out = solver(x, y)

# Out-of-sample entropic barycentric projection:
t_map = solver.transport_fn(out.gv, y, batch_size=32)
mapped = t_map(queries)

# Optional centered (debiased) map and Sinkhorn divergence:
deb_map = divergence.debiased_transport_fn(solver, x, y, batch_size=32)
mapped_debiased = deb_map(queries)
sink_div = divergence.sinkhorn_divergence(mapped_debiased, y, solver=solver)
```

Running the full KS experiment pipeline in Python:

```python
from swirl_dynamics.projects.debiasing.optimal_transport import experiment
from swirl_dynamics.projects.debiasing.optimal_transport.configs import config_ks_1d

config = config_ks_1d.get_config()
config.dataset_path = "/tmp/ks.hdf5"
metrics = experiment.run_experiment(config, workdir="/tmp/ot_ks")
```

### Evaluation Metrics

`experiment.evaluate` computes metrics in physical units on the held-out test
split for both the transported snapshots (`debiased`) and the untranslated
low-fidelity baseline (`*_reference`), writing them to `<workdir>/metrics.json`:

*   `pdf_l1_error` / `pdf_l1_error_reference`: L1 distance between the
    normalized empirical histograms of the evaluated fields and the held-out
    high-fidelity fields (`test_b`).
*   `wasserstein1` / `wasserstein1_reference`: 1-D Wasserstein-1 distance
    (`evaluation_metrics.wasserstein1_1d`) pooled across spatial grid points.
*   `spectral_log_ratio` / `spectral_log_ratio_reference`: energy-weighted mean
    absolute $\log_{10}$ ratio of the 1-D spatial energy spectra.
*   `sinkhorn_divergence` / `sinkhorn_divergence_reference`: multi-dimensional
    Sinkhorn divergence $S_\varepsilon$ between the flattened snapshots and
    `test_b` (evaluated on up to `sinkhorn_divergence_max_samples = 512` samples
    using the fitted `SinkhornSolver`).
*   Solver diagnostics: resolved `epsilon`, `num_iterations`, `converged`,
    `marginal_error`, `reg_ot_cost` ($\langle P_\varepsilon, C \rangle$), and
    `entropic_ot_cost` (Fenchel-Rockafellar dual value).

## References

1.  Marco Cuturi, *"Sinkhorn Distances: Lightspeed Computation of Optimal
    Transport,"* Advances in Neural Information Processing Systems (NeurIPS) 26,
    pp. 2292–2300, 2013 ([arXiv:1306.0895](https://arxiv.org/abs/1306.0895)).
2.  Aram-Alexandre Pooladian and Jonathan Niles-Weed, *"Entropic estimation of
    optimal transport maps,"* arXiv preprint
    [arXiv:2109.12004](https://arxiv.org/abs/2109.12004), 2021.
3.  Hicham Janati, Boris Muzellec, Gabriel Peyré, and Marco Cuturi, *"Entropic
    Optimal Transport between Unbalanced Gaussian Measures has a Closed Form,"*
    Advances in Neural Information Processing Systems (NeurIPS) 33, pp.
    10468–10479, 2020 ([arXiv:2006.02572](https://arxiv.org/abs/2006.02572)).
4.  Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari,
    Alain Trouvé, and Gabriel Peyré, *"Interpolating between Optimal Transport
    and MMD using Sinkhorn Divergences,"* International Conference on Artificial
    Intelligence and Statistics (AISTATS), PMLR 89, pp. 2681–2690, 2019
    ([arXiv:1810.08278](https://arxiv.org/abs/1810.08278)).
5.  Aram-Alexandre Pooladian, Marco Cuturi, and Jonathan Niles-Weed, *"Debiaser
    Beware: Pitfalls of Centering Regularized Transport Maps,"* International
    Conference on Machine Learning (ICML), PMLR 162, pp. 17830–17847, 2022
    ([arXiv:2202.08919](https://arxiv.org/abs/2202.08919)).
6.  Jason Altschuler, Jonathan Weed, and Philippe Rigollet, *"Near-linear time
    approximation algorithms for optimal transport via Sinkhorn iteration,"*
    Advances in Neural Information Processing Systems (NeurIPS) 30, pp.
    1964–1974, 2017 ([arXiv:1705.09634](https://arxiv.org/abs/1705.09634)).
7.  Gabriel Peyré and Marco Cuturi, *"Computational Optimal Transport,"*
    Foundations and Trends in Machine Learning 11(5–6):355–607, 2019
    ([arXiv:1803.00567](https://arxiv.org/abs/1803.00567)).
8.  Zhong Yi Wan, Ricardo Baptista, Anudhyan Boral, Yi-Fan Chen, John Anderson,
    Fei Sha, and Leonardo Zepeda-Núñez, *"Debias Coarsely, Sample Conditionally:
    Statistical Downscaling through Optimal Transport and Probabilistic
    Diffusion Models,"* Advances in Neural Information Processing Systems
    (NeurIPS) 36, pp. 47749–47763, 2023
    ([arXiv:2305.15618](https://arxiv.org/abs/2305.15618),
    [OpenReview:5NxJuc0T1P](https://openreview.net/forum?id=5NxJuc0T1P)).
9.  Marco Cuturi, Laetitia Meng-Papaxanthos, Yingtao Tian, Charlotte Bunne,
    Geoff Davis, and Olivier Teboul, *"Optimal Transport Tools (OTT): A JAX
    Toolbox for all things Wasserstein,"* arXiv preprint
    [arXiv:2201.12324](https://arxiv.org/abs/2201.12324), 2022.
