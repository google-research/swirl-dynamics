# Swirl-Dynamics

*This is not an official Google product*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Unittests & Auto-publish](https://github.com/google-research/swirl-dynamics/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/swirl-dynamics/actions/workflows/pytest_and_autopublish.yml)

**Swirl-Dynamics** is a JAX/Flax library of models, numerical solvers,
training templates, and research benchmarks for scientific machine learning of
dynamical systems, generative modeling, and climate downscaling.

## Installation

Swirl-Dynamics requires Python 3.11 or newer and can be installed directly from
GitHub:

```bash
pip install git+https://github.com/google-research/swirl-dynamics.git@main
```

By default, this installs CPU-only JAX; for GPU or TPU support, see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

For local development and running the test suite:

```bash
git clone https://github.com/google-research/swirl-dynamics.git
cd swirl-dynamics
pip install -e ".[dev]"
pytest -vv -n auto
```

## Repository Structure

```text
swirl_dynamics/
├── lib/           # Reusable JAX/Flax models, layers, solvers, and metrics
├── templates/     # Standardized training, evaluation, and checkpointing loops
├── data/          # Zarr, HDF5, and Grain data pipeline utilities
└── projects/      # Self-contained research codebases, configs, and Colabs
```

### Core Library (`lib/`, `templates/`, `data/`)

| Package | Description | Key Components |
| :--- | :--- | :--- |
| `lib/diffusion` | Score-based & denoising diffusion frameworks, schedules, samplers, and guidance | `Diffusion`, `OdeSampler`, `SdeSampler`, `TStudentOdeSampler`, `ClassifierFreeHybrid`, `InfillFromSlices`, `PreconditionedDenoiserUNet`, `UNet3d`, `ViViTDiffusion` |
| `lib/networks` | Neural operators and backbone architectures for physical fields | `FNO` (1D/2D/3D), `UNet`, `HyperUNet`, `ConvNet`, `CycleGAN`, `RationalNetwork`, encoders |
| `lib/layers` | Domain-specific neural network layers for spatiotemporal and spherical grids | `AxialSelfAttention`, `AddAxialPositionEmbedding`, `LatLonConv`, `FilteredResize`, `CombineResidualWithSkip`, `channel_to_space` |
| `lib/solvers` | Differentiable JAX ODE and SDE time integrators | `ExplicitEuler`, `HeunsMethod`, `RungeKutta4`, `DoPri45`, `ScanOdeSolver`, `MultiStepDirect`, `EulerMaruyama` |
| `lib/metrics` | Probabilistic ensemble and deterministic regression metrics | `crps`, `threshold_brier_score`, `unreliability_score`, `mean_squared_error`, `mean_absolute_error` |
| `templates` | CLU, Orbax, and Gin-configurable training and evaluation scaffolding | `BaseModel`, `BasicTrainer`, `BasicDistributedTrainer`, `TrainState`, `Evaluator`, `Callback`, `run_train`, `run_eval` |
| `data` | Scientific array storage and streaming input pipeline utilities | Zarr/HDF5 readers and Grain transforms |

## Highlighted Research Projects

Each project directory under `swirl_dynamics/projects/` is self-contained with
its own configuration files, entry-point scripts, and interactive Colab
tutorials:

| Project | Domain & Method | Paper & Resources |
| :--- | :--- | :--- |
| [**GenFocal**](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/genfocal) | Two-stage regional climate downscaling combining Rectified Flow debiasing and Diffusion super-resolution | *Nature Machine Intelligence* 2026 ([article](https://www.nature.com/articles/s42256-026-01308-7), [arXiv:2412.08079](https://arxiv.org/abs/2412.08079)) · [Debiasing Colab](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/genfocal/debiasing/colabs/genfocal_debiasing_demo.ipynb) · [Super-Resolution Colab](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/genfocal/super_resolution/colabs/era5_demo.ipynb) |
| [**R2-D2**](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf) | Regional Residual Diffusion-based Downscaling of CMIP6 climate model ensembles over the Western US | *PNAS* 2025 ([doi:10.1073/pnas.2420288122](https://doi.org/10.1073/pnas.2420288122)) · [Inference Colab](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/OS_downscaling_inference.ipynb) |
| [**DySLIM (`ergodic`)**](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/ergodic) | Ergodicity-inducing regularization for long-term stability of autoregressive models on chaotic systems (Lorenz-63, KS, 2D Navier-Stokes) | *ICML* 2024 ([arXiv:2402.04467](https://arxiv.org/abs/2402.04467)) · [Training Colab](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/ergodic/colabs/demo_training.ipynb) |
| [**Evolve Smoothly**](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/evolve_smoothly) | Hypernetwork-based continuous-time latent dynamics surrogates for advection-dominated PDEs | *ICLR* 2023 ([arXiv:2301.10391](https://arxiv.org/abs/2301.10391)) · [Demo Colab](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/evolve_smoothly/colabs/demo.ipynb) |

## Contributing

Contributions are welcome! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for
details on our Contributor License Agreement (CLA), code review workflow, and
community guidelines.

## License

Swirl-Dynamics is licensed under the [Apache License, Version 2.0](LICENSE).
