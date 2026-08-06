# https://hub.docker.com/r/nvidia/cuda/tags
# Select the tag that matches the CUDA version on the host.
# Make sure to select a CuDNN image!
# See the compatibility matrix here: https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Python and the handful of CLI tools the dev container expects.
RUN apt update \
    && apt install --fix-missing -y python3 python3-pip curl vim less openssh-client git \
    && ln -s $(which python3) /usr/bin/python

# Dev containers should substitute the current user on startup, but that doesn't work:
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd -g 1000 pcx \
    && useradd -m -u 1000 -g pcx pcx \
    && chmod -R a+rwx /usr/local /usr/include/python3.10
USER pcx:pcx
WORKDIR /home/pcx/workspace

ENV PATH="/home/pcx/.local/bin:$PATH" \
    # Prevent JAX from allocating 90% of GPU memory.
    XLA_PYTHON_CLIENT_PREALLOCATE="false" \
    # Build the venv inside the image at a fixed path rather than in the bind-mounted
    # workspace, so a rebuild doesn't fight with the host's own .venv.
    UV_PROJECT_ENVIRONMENT="/home/pcx/.venv" \
    UV_LINK_MODE="copy" \
    UV_COMPILE_BYTECODE="1"

# Install uv and just.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
       | bash -s -- --to /home/pcx/.local/bin

# Resolve dependencies in their own layer so edits to the source don't invalidate it.
# The bracket in 'uv.loc[k]' makes the lockfile optional at build time.
COPY --chown=pcx:pcx pyproject.toml uv.loc[k] README.md ./
RUN uv sync --group dev --no-install-project

# Put the environment first on PATH so `python` is the project interpreter.
ENV PATH="/home/pcx/.venv/bin:$PATH"
