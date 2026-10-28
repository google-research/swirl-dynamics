<!-- disableFinding(SNIPPET_INVALID_LANGUAGE) -->

# CycleGAN debiasing

Unpaired translation between two distributions of physical fields with a
CycleGAN [1], written in JAX/Flax on top of the `swirl_dynamics` templates. The
package is one of three debiasing baselines of `swirl_dynamics`, next to the
rectified-flow (`../rectified_flow`) and optimal-transport
(`../optimal_transport`) approaches, and ships the same public
Kuramoto-Sivashinsky (KS) example: snapshots of a low-order finite-volume solver
(domain A) are mapped onto the distribution of a pseudo-spectral solver (domain
B) without paired samples.

## Mathematical formulation

Notation: $a \sim \mu_A$, $b \sim \mu_B$; $G$ = `generator_a2b` ($A \to B$),
$F$ = `generator_b2a` ($B \to A$); $D_A$ and $D_B$ discriminate samples *of* $A$
and *of* $B$ (domain-based naming; the PyTorch reference implementation names
its discriminators after the generators instead); $\lVert\cdot\rVert_1$ is the
**mean** absolute error over all entries (`l1_loss`); $\ell(p, t)$ is the
adversarial criterion applied to a PatchGAN prediction map $p$ with the scalar
target $t$ broadcast; $\operatorname{sg}$ is `stop_gradient`.

**Generator step** (`CycleGANModel.generator_loss_and_samples`):

```math
\begin{aligned}
\mathcal{L}_{G,F} &= \ell\big(D_B(G(a)), 1\big) + \ell\big(D_A(F(b)), 1\big) \\
&\quad + \lambda_A \lVert F(G(a)) - a\rVert_1 + \lambda_B \lVert G(F(b)) - b\rVert_1 \\
&\quad + \lambda_{id}\big(\lambda_B \lVert G(b) - b\rVert_1 + \lambda_A \lVert F(a) - a\rVert_1\big).
\end{aligned}
```

The cycle terms are used if `use_cycle_loss`, the identity terms if
`same_dimension` **and** $\lambda_{id} > 0$ (otherwise the two extra generator
passes are skipped). The discriminator parameters are stop-gradiented in this
step.

**Discriminator step** (`CycleGANModel.discriminator_loss_and_metrics`), on the
real batch and the samples generated in the generator step:

```math
\begin{aligned}
\mathcal{L}_D &= \tfrac12\big[\ell(D_A(a), 1) + \ell(D_A(\operatorname{sg} F(b)), 0)\big] \\
&\quad + \tfrac12\big[\ell(D_B(b), 1) + \ell(D_B(\operatorname{sg} G(a)), 0)\big] \\
&\quad + \underbrace{\tfrac12\big[\ell(D_A(b), 0) + \ell(D_B(a), 0)\big]}_{\texttt{use\_crossed\_dis\_loss}}.
\end{aligned}
```

The factor $\tfrac12$ on the discriminator objective follows the reference
implementation ("we divide the objective by 2 while optimizing $D$", [1]); the
crossed term, which also trains each discriminator to reject real samples of the
other domain, is an extension of this package and is off by default.

**Criteria** (`loss_functions.GanLoss`), with labels $1$ (real) and $0$ (fake),
optionally smoothed through `target_real_label`:

*   LSGAN [3]: $\ell(p, t) = \overline{(p - t)^2}$;
*   BCE with logits: $\ell(p, t) = \overline{(1 - t)p + \mathrm{softplus}(-p)}$,
    the numerically stable form of $-\overline{t \log \sigma(p) + (1 -
    t)\log(1-\sigma(p))}$;
*   Wasserstein critic [4]: $\ell(p, 1) = -\bar p$, $\ell(p, 0) = +\bar p$.

**WGAN-GP** (`loss_functions.gradient_penalty`). With per-sample interpolates
$\hat u = \alpha u + (1 - \alpha)\operatorname{sg}(\tilde u)$, $\alpha \sim
\mathcal{U}(0, 1)$, and the per-sample critic score $s(\hat u) =
\mathrm{mean}_{\text{patch}} D(\hat u)$,

```math
\mathrm{GP} = \mathbb{E}_{\hat u}\Big[\big(\sqrt{\lVert\nabla_{\hat u} s(\hat u)\rVert_2^2 + \varepsilon_{\text{stab}}} - 1\big)^2\Big],
\qquad
\mathcal{L}_D^{\text{WGAN-GP}} = \tfrac12\big[\bar D(\tilde u) - \bar D(u) + \lambda_{gp}\thinspace\mathrm{GP}\big]
```

per discriminator. The common $\tfrac12$ keeps the ratio between the critic term
and the penalty of [4], so `lambda_gp = 10` means the paper's 10. The penalty is
computed from a single backward pass of $\sum_b s(\hat u_b)$, which equals the
per-sample gradients if and only if the critic has no cross-sample coupling
(true for instance, group or no normalisation; false for batch normalisation).
The stabiliser $\varepsilon_{\text{stab}}$ keeps the gradient of the penalty
finite where the critic gradient vanishes.

**Training loop** (`trainers.CycleGANTrainer`). Each step performs one generator
update followed by `n_critic` discriminator updates on the same real batch and
generated samples; the step counter advances once. Two optimizers are used (one
over both generators, one over both discriminators), which is identical to four
per-network optimizers for element-wise transforms such as Adam but not for
cross-parameter transforms such as global-norm clipping. Optional and off by
default: a replay pool [5] of generated samples for the discriminators
(`pool_size`; bypassed for the Wasserstein critic, whose penalty needs on-policy
samples) and an exponential moving average of the generator parameters for
inference (`ema_decay`).

**Agreement with the references.** Identity weighting $\lambda_B\lambda_{id}$ /
$\lambda_A\lambda_{id}$, the $\tfrac12$ on $\mathcal L_D$, LSGAN without
$\tfrac12$ on the generator side and pre-update fakes for the discriminator step
all match the reference implementation [7]. Code defaults are $\lambda_A =
\lambda_B = \lambda_{id} = 1$; the paper and the KS config use $\lambda_A =
\lambda_B = 10$, $\lambda_{id} = 0.5$, Adam($2\cdot10^{-4}$, $\beta_1 = 0.5$)
with a linear learning-rate decay and a 50-sample history buffer.

## Networks and scale sensitivity

`swirl_dynamics.lib.networks.cycle_gan` provides a ResNet/U-Net `Generator`, a
PatchGAN `Discriminator` and a `FilteredInterpolation` layer for inputs of shape
`(batch, *spatial, channels)` with one or two spatial dimensions. Their defaults
(instance normalisation, `tanh` output) are the image-translation defaults. For
physical fields the KS config changes three of them, for the following reasons.

*   **Instance normalisation makes the discriminator amplitude-blind.** The
    default discriminator is `Conv(bias) → PReLU` followed by
    `Conv(no bias) → GroupNorm(group_size=1) → PReLU` blocks. At initialisation
    the first layer is positively homogeneous (zero bias) and every later layer
    is normalised per sample and channel, so $D(cu) = D(u)$ for $c > 0$ up to
    $O(\varepsilon_{GN} / \sigma^2)$, where $\sigma^2$ is the per-channel
    activation variance (with the shipped `normal(0.02)` kernels $\sigma^2 \sim
    10^{-4}$, so the residual is small but not negligible: for a small
    discriminator at initialisation the measured $\max|D(3u) - D(u)|$ is about
    0.5% of the response to an independent input). After training, amplitude
    information can enter only through the bias and PReLU kink of the first
    layer. Per-channel biases of the normalised convolutions (`use_bias=True`)
    are cancelled algebraically by the mean subtraction, and in float32 with the
    default fast variance estimate $E[x^2] - E[x]^2$ biases much larger than the
    activations degrade the normalisation numerically, so `use_bias=True` is not
    a fix. Amplitude-aware debiasing therefore uses `norm_type="none"` (added in
    this package) with biases on.
*   **Degenerate instance norm.** If a normalised feature map has a single
    spatial element its normalised value is exactly $0$ and the discriminator is
    identically $0$ for every input (e.g. the default `n_layers=3`, i.e. four
    stride-2 convolutions, on a $16\times16$ input). The networks now raise a
    `ValueError` in this case; choose fewer layers or longer signals.
*   **Output head.** The default `final_act_fun=tanh` clips fields to $(-1, 1)$;
    the KS config uses the identity.
*   **Global skip.** `use_global_skip` requires as many input channels as
    `output_nc` (a mismatch now raises instead of silently broadcasting).

## Code map

*   `loss_functions.py`: `GanMode`, `GanLoss` (LSGAN / BCE / Wasserstein),
    `l1_loss`, `gradient_penalty`.
*   `models.py`: `CycleGANModelConfig`, `CycleGAN` (four networks, loss terms),
    `CycleGANModel` (`BaseModel` interface: `loss_fn`,
    `generator_loss_and_samples`, `discriminator_loss_and_metrics`, `eval_fn`,
    `inference_fn`).
*   `trainers.py`: `CycleGANTrainState`, `CycleGANTrainer`,
    `DistributedCycleGANTrainer`, replay pool utilities.
*   `dataloaders.py`: in-memory pygrain source, `UnpairedDataLoader`, KS loading
    helpers.
*   `evaluation_metrics.py`: `pdf_l1_error`, `wasserstein1_1d`,
    `energy_spectrum_1d`, `spectral_log_ratio`, `cycle_consistency_error`.
*   `experiment.py`: builders from a config, `train_experiment`,
    `evaluate_experiment`.
*   `configs/config_ks_1d.py`: the KS experiment.
*   `main_train.py`, `evaluation_main.py`: command-line entry points.
*   `colab/demo_cycle_gan.ipynb`: end-to-end demo on the KS data.
*   `../../../lib/networks/cycle_gan.py`: the N-D generator, discriminator and
    filtered interpolation.

## Data

The KS example uses the public file
`gs://gresearch/swirl_dynamics/downscaling/KS_finite_volumes_vs_pseudo_spectral.hdf5`
with keys `u_fd` (finite-volume trajectories, domain A), `u_sp` (pseudo-spectral
trajectories at four times the spatial resolution, domain B), `x` and `t`.
`dataloaders.load_ks_snapshots` sub-samples `u_sp` in space by
`downsample_factor=4`, holds out the last 16 trajectories and flattens the
trajectories into snapshots of shape `(48, 1)`. Batches are dictionaries with
keys `real_data_a` and `real_data_b`, each of shape `(batch, length, 1)`,
sampled independently from the two domains; the evaluation loader reuses the
normalisation statistics of the training split.

## Quick start

```bash
gsutil cp gs://gresearch/swirl_dynamics/downscaling/KS_finite_volumes_vs_pseudo_spectral.hdf5 /tmp/ks.hdf5
python main_train.py --config=configs/config_ks_1d.py --workdir=/tmp/cycle_gan_ks \
  --config.dataset_path=/tmp/ks.hdf5
python evaluation_main.py --config=configs/config_ks_1d.py --workdir=/tmp/cycle_gan_ks
```

The same experiment in Python:

```python
import jax
from swirl_dynamics.projects.debiasing.cycle_gan import experiment
from swirl_dynamics.projects.debiasing.cycle_gan import trainers
from swirl_dynamics.projects.debiasing.cycle_gan.configs import config_ks_1d

config = config_ks_1d.get_config()
config.dataset_path = "/tmp/ks.hdf5"
experiment.train_experiment(config, "/tmp/cycle_gan_ks")
metrics = experiment.evaluate_experiment(config, "/tmp/cycle_gan_ks")

# Inference from the checkpoint.
model = experiment.build_model(config)
state = trainers.CycleGANTrainState.restore_from_orbax_ckpt(
    "/tmp/cycle_gan_ks/checkpoints"
)
to_b = jax.jit(
    trainers.CycleGANTrainer.inference_fn_from_state_dict(
        state,
        model,
        direction="a2b",
        use_ema=config.ema_decay is not None,
    )
)
```

Leaving `config.dataset_path` empty trains on a small deterministic synthetic
dataset, which is meant for smoke tests only.

## Evaluation metrics

`experiment.evaluate` reports, in physical units, the L1 distance between the
normalised histograms (`pdf_l1_error`), the one-dimensional Wasserstein-1
distance (`wasserstein1`, exact when the smaller sample count divides the larger
one—in particular for equal sample counts—and a midpoint-rule approximation
otherwise) and the energy-weighted absolute $\log_{10}$ ratio of the energy
spectra (`spectral_log_ratio`) of the translated low-fidelity samples against
the held-out high-fidelity samples, the same quantities for the untranslated
samples (`*_reference`), and the relative $L_2$ cycle-consistency error of
$F(G(a))$.

## Known limitations

*   The networks support one or two spatial dimensions; the `LONLAT` padding of
    the convolution layers is two-dimensional only.
*   Networks with mutable Flax collections (e.g. batch statistics) are not
    supported by the trainer, which also rules out batch normalisation in the
    critic of WGAN-GP (the single-pass penalty assumes no cross-sample
    coupling).
*   The replay pool stores a fixed number of samples of a fixed shape and is
    disabled for the Wasserstein critic.
*   Normalisation statistics are per location and computed on the training
    split; `load_ks_snapshots` reads a local copy of the HDF5 file.

## References

1.  J.-Y. Zhu, T. Park, P. Isola, A. A. Efros. *Unpaired Image-to-Image
    Translation using Cycle-Consistent Adversarial Networks.* ICCV 2017
    (arXiv:1703.10593).
2.  P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros. *Image-to-Image Translation with
    Conditional Adversarial Networks* (PatchGAN). CVPR 2017.
3.  X. Mao et al. *Least Squares Generative Adversarial Networks.* ICCV 2017
    (arXiv:1611.04076).
4.  M. Arjovsky, S. Chintala, L. Bottou. *Wasserstein GAN.* ICML 2017; I.
    Gulrajani et al. *Improved Training of Wasserstein GANs.* NeurIPS 2017
    (arXiv:1704.00028).
5.  A. Shrivastava et al. *Learning from Simulated and Unsupervised Images
    through Adversarial Training.* CVPR 2017.
6.  Z. Y. Wan et al. *Debias Coarsely, Sample Conditionally.* NeurIPS 2023
    (arXiv:2305.15618).
7.  `junyanz/pytorch-CycleGAN-and-pix2pix` (loss conventions).
