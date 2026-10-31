"""Downloads the models into the /models volume on the first start."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import Settings
from .tts import REPO, VOICES, voice_files

log = logging.getLogger(__name__)


def use_models_dir(models_dir: Path) -> None:
    os.environ["HF_HOME"] = str(models_dir / "huggingface")


def fetch_voices() -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    paths = {}
    for voice in sorted(set(VOICES.values())):
        model, config = voice_files(voice)
        paths[voice] = hf_hub_download(REPO, model)
        hf_hub_download(REPO, config)
    return paths


def fetch_llm(settings: Settings) -> Path:
    from huggingface_hub import hf_hub_download

    if not settings.llm_path.exists():
        log.info("downloading %s from %s", settings.llm_file, settings.llm_repo)
        hf_hub_download(settings.llm_repo, settings.llm_file, revision=settings.llm_revision,
                        local_dir=settings.models_dir)
    return settings.llm_path
