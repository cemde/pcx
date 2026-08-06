# PCX -- Predictive Coding Networks Made Simple

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/pcx.svg)](https://badge.fury.io/py/pcx)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://pcx.readthedocs.io/en/stable/)
[![CI](https://github.com/liukidar/pcx/actions/workflows/ci.yml/badge.svg)](https://github.com/liukidar/pcx/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/liukidar/pcx/graph/badge.svg)](https://codecov.io/gh/liukidar/pcx)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2407.01163-b31b1b.svg)](https://arxiv.org/abs/2407.01163)

PCX is a Python JAX-based library designed to develop highly configurable predictive coding networks. Refer to the tutorial notebooks in the [examples](examples/) folder to get started.

## Installation

PCX needs Python 3.11 or newer. Install JAX first, in the build that matches your accelerator — see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) — and then install PCX.

```shell
pip install -U "jax[cuda12]"   # NVIDIA GPU, CUDA 12 (Linux only)
pip install -U jax             # CPU, all platforms
pip install pcx
```

### Platform support

| Platform    | CPU | NVIDIA GPU                                                                                             |
| ----------- | --- | ------------------------------------------------------------------------------------------------------ |
| Linux       | ✅  | ✅ via `jax[cuda12]`                                                                                    |
| macOS       | ✅  | n/a — Apple Silicon acceleration is available through the experimental [`jax-metal`](https://developer.apple.com/metal/jax/) plugin |
| Windows     | ✅  | ❌ — JAX ships no CUDA wheels for native Windows; use [WSL2](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) for GPU work |

PCX itself is pure Python and ships a single universal wheel, so the package installs everywhere. The table above describes what JAX can do underneath it; CI runs the test suite on Linux, macOS and Windows against the CPU backend.

### Installing from source

To work on PCX itself, or to track `main`:

```shell
git clone https://github.com/liukidar/pcx.git
cd pcx
uv sync --group dev
```

That creates a `.venv` from the locked dependency set in `uv.lock`, so the environment is reproducible across machines. If you do not have [uv](https://docs.astral.sh/uv/) yet, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Prefer a plain editable install? `pip install -e .` also works.

## Development

Common tasks run through [just](https://github.com/casey/just):

```shell
just install    # create the dev environment
just fix        # format and auto-fix lint findings
just check      # format check, lint, type check
just test       # run the test suite
just all        # fix, check and test — run this before opening a PR
just            # list every recipe
```

The toolchain is [uv](https://docs.astral.sh/uv/) for packaging, [ruff](https://docs.astral.sh/ruff/) for formatting and linting, [ty](https://github.com/astral-sh/ty) for type checking and [pytest](https://docs.pytest.org/) for tests. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and the release process.

### Docker and Dev Containers

Run your development environment in a container — the most straightforward option, as everything is pre-configured. The `Dockerfile` lives in [docker/](docker/), with a `run.sh` that builds and runs it.

**Warning**: the image needs CUDA 12.2 or later. Check that `nvidia-smi` reports CUDA >= 12.2; if not, update the base `nvidia/cuda` image in [docker/Dockerfile](docker/Dockerfile) to match your host.

Requirements:

1. A CUDA >= 12.2 machine with an NVIDIA GPU. Without a GPU, omit the GPU passthrough steps.
2. [Docker](https://docs.docker.com/engine/install/), version > 20.10.9.
3. [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit), so Docker can reach the GPU. **Restart the Docker daemon afterwards** (`sudo systemctl restart docker` on Ubuntu).
4. [VS Code](https://code.visualstudio.com/download) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Open the project in VS Code and run `Dev Containers: Reopen in Container` (Ctrl/Cmd+Shift+P). The first build takes 15–30 minutes. `Dev Containers: Reopen folder locally` exits, and `Dev Containers: Rebuild Container` rebuilds. Running `hostname` tells you where you are: 12 meaningless characters means you are inside the container.

Inside the container, add packages with `uv add <package>` (or `uv add --group dev <package>` for tooling), which updates `pyproject.toml` and `uv.lock` together. Do not edit `uv.lock` by hand.

## Documentation

The documentation is available at [pcx.readthedocs.io](https://pcx.readthedocs.io/en/stable/). To build it yourself, see [docs/README.md](docs/README.md) or run `just docs`.

## Citation

If this library was useful in your work, please cite [our paper](https://arxiv.org/abs/2407.01163):

```bibtex
@article{pinchetti2024benchmarkingpredictivecodingnetworks,
      title={Benchmarking Predictive Coding Networks -- Made Simple},
      author={Luca Pinchetti and Chang Qi and Oleh Lokshyn and Gaspard Olivers and Cornelius Emde and Mufeng Tang and Amine M'Charrak and Simon Frieder and Bayar Menzat and Rafal Bogacz and Thomas Lukasiewicz and Tommaso Salvatori},
      year={2024},
      eprint={2407.01163},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01163},
}
```

For the code behind the experiments in that paper, see the [benchmark paper release](https://github.com/liukidar/pcax/releases/tag/v0.6.1).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md), and record user-visible changes in [CHANGELOG.md](CHANGELOG.md).
