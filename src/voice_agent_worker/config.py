"""Settings of the worker, all from the environment (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


@dataclass(frozen=True)
class Settings:
    api_key: str
    models_dir: Path
    grpc_port: int
    asr_model: str
    asr_threads: int
    asr_workers: int
    tts_workers: int
    llm_enabled: bool
    llm_port: int
    llm_repo: str
    llm_file: str
    llm_revision: str
    llm_gpu_layers: int
    llm_slots: int
    llm_ctx_per_slot: int
    llm_reasoning_effort: str

    @property
    def llm_path(self) -> Path:
        return self.models_dir / self.llm_file


def load() -> Settings:
    api_key = os.environ.get("WORKER_API_KEY", "")
    if not api_key:
        raise SystemExit("WORKER_API_KEY is not set")
    return Settings(
        api_key=api_key,
        models_dir=Path(os.environ.get("MODELS_DIR") or "/models"),
        grpc_port=_int("GRPC_PORT", 50051),
        asr_model=os.environ.get("ASR_MODEL") or "gigaam-v3-e2e-rnnt",
        asr_threads=_int("ASR_THREADS", 4),
        asr_workers=_int("ASR_WORKERS", 2),
        tts_workers=_int("TTS_WORKERS", 2),
        llm_enabled=os.environ.get("LLM_ENABLED", "1") != "0",
        llm_port=_int("LLM_PORT", 8080),
        llm_repo=os.environ.get("LLM_REPO") or "ggml-org/gpt-oss-20b-GGUF",
        llm_file=os.environ.get("LLM_FILE") or "gpt-oss-20b-MXFP4.gguf",
        llm_revision=os.environ.get("LLM_REVISION") or "15dcfc7e7c608c6161507211ed43c5b590801b4b",
        llm_gpu_layers=_int("LLM_GPU_LAYERS", 999),
        llm_slots=_int("LLM_SLOTS", 4),
        llm_ctx_per_slot=_int("LLM_CTX_PER_SLOT", 8000),
        llm_reasoning_effort=os.environ.get("LLM_REASONING_EFFORT") or "low",
    )
