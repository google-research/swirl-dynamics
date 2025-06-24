# Regional climate risk assessment from climate models using probabilistic machine learning

![Diagram of GenFocal](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/genfocal/figures/genfocal_diagram.png)

This package contains example code to run downscaling models described in
research paper [GenFocal](https://arxiv.org/abs/2412.08079).

Code for the debiasing stage can be found in the `debiasing/` sub-folder
and code for the super-resolution stage can be found in the `super_resolution`
sub-folder.

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

4. Install `swirl-dynamics`. This steps will install all the
[basic required dependencies](https://github.com/google-research/swirl-dynamics/blob/main/pyproject.toml)
and it will install genfocal. Here we assume that you have `git`
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
│   ├── configs/      # examples configuration files.
│   └── figures/      # figures for the readmes and notebooks.
├── figures/
└── super-resolution/ # contains the code for the super-resolution step.
    └── colabs/       # demo for super-resolution.
    └── configs/      # example configuration files.
```

The repository is organized into two primary folders for the debiasing and
super-resolution steps. Each folder contains a `colabs` subfolder with
corresponding demos.

The `data` folder includes notebooks for loading and visualizing the data.

The `analysis` folder contains the code to reproduce the analysis in our paper.
Note that running the full evaluation pipeline is computationally expensive due
to its reliance on `TempestExtremes` and terabytes of data. Therefore, we
provide notebooks to reproduce the figures and the source code for the more
intensive analyses.

## Workflow

As GenFocal has two distinct steps the workshop is also split in two.
Each step is specified in the README files inside the corresponding folders.

## Tropical Cyclones

Below we show one of the main results in the paper, which shows the tracks and
their corresponding ensemble density, for the biased climate dataset (LENS2),
their downscaled version given by GenFocal, and the reference given by ERA5.
![GenFocal TC results](https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/genfocal/figures/tc_densities.png)

## License

Licensed under the Apache License, Version 2.0.
