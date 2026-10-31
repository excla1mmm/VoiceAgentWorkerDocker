# syntax=docker/dockerfile:1.7
#
# Voice agent worker: GigaAM and Piper on the CPU, llama.cpp on the GPU, in one container.
#   docker compose up -d --build
#
# LLAMA_IMAGE: server-cuda-* for an NVIDIA GPU (x86_64), server-* for the CPU (x86_64 and arm64).
# The models are downloaded into the /models volume on the first start.

ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-b11669
FROM ${LLAMA_IMAGE}

ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_NO_CACHE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
ADD --chmod=755 https://astral.sh/uv/0.11.28/install.sh /tmp/uv-install.sh
RUN UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/uv-install.sh && rm /tmp/uv-install.sh

WORKDIR /worker
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 && uv sync --frozen --group dev --no-install-project
COPY proto ./proto
COPY scripts ./scripts
COPY src ./src
ENV PATH=/opt/venv/bin:/app:$PATH \
    PYTHONPATH=/worker/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/app \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    OMP_NUM_THREADS=1
RUN bash scripts/gen-proto.sh && python -c "import voice_agent_worker.server"

VOLUME ["/models"]
EXPOSE 50051 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60m --retries=3 CMD ["python", "-m", "voice_agent_worker", "healthcheck"]
ENTRYPOINT ["python", "-m", "voice_agent_worker"]
CMD ["serve"]
