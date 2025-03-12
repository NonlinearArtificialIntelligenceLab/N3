# N3

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)

### Code for evolving large neural networks from small ones paper.
This code base is uv compatible and pip installable.

### Authors
Anil Radhakrishnan, John F. Lindner, Scott T. Miller, Sudeshna Sinha, William L. Ditto

### Link to paper

### Key Results
- Growing networks can dynamically evolve their size during gradient descent to help solve problems involving nonlinear regression and classification
- We present two different algorithms:
    - an auxiliary weight algorithm implemented in Mathematica serving as a conceptual proof of concept where an additional weight is added to a standard multi-layer perceptron (MLP) to whose value dictates the size of the network
    - a more generalized scheme implemented in Python using JAX separating the auxiliary weight from the network and allowing for more flexibility in the network architecture in a controller-mlp paradigm that can be efficiently vectorized and parallelized in standard deep learning frameworks
    - We show improved results for growing networks over static(grown) networks for in both pipelines for regression and classification tasks

### Installation
We recommend using [uv](https://docs.astral.sh/uv/) to manage python and install the package.

Then, you can simply git clone the repository and run,

```bash
uv pip install .
```
to install the package with all dependencies.

### Usage

The notebooks in the `nbs` illustrate the pipeline for training and testing the growing and grown networks for regression and classification tasks.
To run the notebooks you can install the additional dependencies as noted in the `pyproject.toml` file.

The scripts in the `scripts` directory are the same as the notebooks but with argparsing for easy command line usage for use in batch processing.
To run the scripts, you can use the `uv run` command to run the scripts in the `scripts` directory.

The analysis of the statistical run results can be done using the `analysis` notebook in the `nbs` directory.

The mathematica notebooks used for this work can be accessed in the `mathematica` directory.

For running grid searches and sweeps, the `config/sweeps.yaml` config file can be edited accordingly and run with `scripts/run_sweep.py`

### Code References
- [Equinox](https://docs.kidger.site/equinox/) Pytorch like module for JAX
- [JAX](https://github.com/jax-ml/jax) Accelerator-oriented array computation and program transformation
- [Optax](https://github.com/google-deepmind/optax) Gradient processing and optimization library for JAX
<!-- ### dev notes

the package can be installed editably via pip:

```bash
pip install -e . # install in editable mode
```

if the cuda version of jax is causing issues, you can instead install the cpu version by editing out the `[cuda12]` from jax in the pyproject.toml file.

Basic scripts with argparsing are in the `scripts` directory.
The notebooks in the `nbs` directory illustrate what the scripts do. -->
