# Regional climate risk assessment from climate models using probabilistic machine learning

![Diagram of GenFocal](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/genfocal/figures/genfocal_diagram.png)

This package contains example code to run downscaling models described in
research paper [GenFocal](https://arxiv.org/abs/2412.08079).

Code for the debiasing stage can be found in [projects/debiasing/rectified_flow](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/debiasing/rectified_flow)
and code for the super-resolution stage can be found in [projects/probabilistic_diffusion/downscaling/era5](https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion/downscaling/era5).

## Installation

To make the installation seamlessly we use pip. Although the installation should
work in any Linux-based system, we strongly recommend using a high-memory
GPU, such as a `A100` (80GB), `H100`, `B100`, or a `TPU v5p`.

1. Create a virtual environment.
```bash
python3 -m venv genfocal
```

2. Activate the virtual environment
```bash
source genfocal/bin/activate
```

3. Install jax with the accelerator support that matches your system. If you
have a GPU then, following jax installation [instructions](https://docs.jax.dev/en/latest/installation.html#)
you can type

```bash
pip install -U "jax[cuda12]"
```

4. Install `swirl-dynamics`. This steps will install all the required
dependencies and it will install genfocal. Here we assume that you have `git`
already installed in your system.

```bash
pip install git+https://github.com/google-research/swirl-dynamics.git@main --quiet
```

5. Clone the repository to your machine.

```bash
git clone https://github.com/google-research/swirl-dynamics.git
```

The training and inference routines are written to work in a single accelerator,
multiple accelerator, or multiple replica (several nodes) regimes.
This capabilities are access via a flag in the configuration files. We recommend
to use multiple accelerators for training and inference.

## Repository Structure

The repository is organized as follows:
```
genfocal/
├── analysis/
├── data/
├── debiasing/        # contains the code for th bias-correction step.
│   ├── colabs/       # demo for debiasing.
│   ├── configs/      # examples of configuration files.
│   └── figures/      # figures for the readmes and notebooks.
├── figures/
└── super-resolution/ # contains the code for the super-resolution step.
    └── configs/
```

As such, the repository is organized such that the two steps, debiasing and
super-resolution have their own folders. Demos for each step can be found in
a respective `colabs` subfolder.

We provide a data folder which contains notebooks for loading and
visualizing data.

The `analysis` folder contains code to showcase the analysis performed in the
paper. As some of the analysis requires `TempestExtremes` and the size of the
data is several Terabytes, thus running the full evaluation pipeline takes
considerable time, we only provide with notebooks to reproduce the
figure, while providing the code for the analysis which requires more
computational resources.

## Workflow

As GenFocal has two distinct steps the workshop is also split in two.
Each step is specified in the README files inside the corresponding folders.

## License

Licensed under the Apache License, Version 2.0.
