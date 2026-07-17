# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.11.11

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=hardlink \
    UV_CONCURRENT_DOWNLOADS=1 \
    UV_CONCURRENT_INSTALLS=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_CACHE_DIR=/workspace/wav2vec-mos/.uv-cache \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    WANDB_DIR=/workspace/wandb \
    PATH="/workspace/wav2vec-mos/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

# gsutil for syncing outputs/ to GCS during training (see worker-pool-spec.yaml);
# picks up credentials from the VM's metadata server automatically.
RUN uv pip install --system --break-system-packages --no-cache gsutil

WORKDIR /workspace/wav2vec-mos

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts

# The base image already ships a matching torch/torchvision/torchaudio +
# CUDA runtime build; --system-site-packages + skipping these packages
# avoids downloading a second copy during `uv sync`.
RUN uv venv --system-site-packages .venv \
    && uv sync --frozen --no-dev \
        --no-install-package torch \
        --no-install-package torchvision \
        --no-install-package torchaudio \
        --no-install-package triton \
        --no-install-package nvidia-cublas-cu12 \
        --no-install-package nvidia-cuda-cupti-cu12 \
        --no-install-package nvidia-cuda-nvrtc-cu12 \
        --no-install-package nvidia-cuda-runtime-cu12 \
        --no-install-package nvidia-cudnn-cu12 \
        --no-install-package nvidia-cufft-cu12 \
        --no-install-package nvidia-cufile-cu12 \
        --no-install-package nvidia-curand-cu12 \
        --no-install-package nvidia-cusolver-cu12 \
        --no-install-package nvidia-cusparse-cu12 \
        --no-install-package nvidia-cusparselt-cu12 \
        --no-install-package nvidia-nccl-cu12 \
        --no-install-package nvidia-nvjitlink-cu12 \
        --no-install-package nvidia-nvshmem-cu12 \
        --no-install-package nvidia-nvtx-cu12 \
    && rm -rf "${UV_CACHE_DIR}"

RUN python -c "import torch; assert torch.__version__.startswith('2.11.0'), torch.__version__; print('Training image ready:', torch.__version__, 'CUDA', torch.version.cuda)"

RUN mkdir -p "${HF_HOME}" "${WANDB_DIR}"

# Vertex AI Custom Jobs run one command to completion and are non-interactive:
# override `command`/`args` in the worker pool spec, e.g.
#   command: ["bash"]
#   args: ["scripts/train.sh"]
# See README.md for a full worker-pool-spec.yaml example.
CMD ["bash"]
