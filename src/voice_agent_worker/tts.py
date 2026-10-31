"""Synthesis: Piper voices on the CPU behind the voice names of the module (it knows only the LiveTTS ones)."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .normalize import normalize, unstressed

log = logging.getLogger(__name__)

REPO = "rhasspy/piper-voices"
FEMALE = "ru_RU-irina-medium"
MALE = "ru_RU-dmitri-medium"
VOICES = {"sofia": FEMALE, "vera": FEMALE, "irina": FEMALE, "vasily": MALE, "sergey": MALE,
          "denis": "ru_RU-denis-medium"}
CHUNK_S = 0.1

PAUSE = re.compile(r"\s*sil<\[\d+\]>\s*")  # a SpeechKit pause
STRESSED_WORD = re.compile(r"[А-ЯЁ]*\+[А-ЯЁ]+")  # the core writes words with a stress mark in capitals for LiveTTS
SPEAKABLE = re.compile(r"[А-Яа-яЁё0-9]")


def voice_files(voice: str) -> tuple[str, str]:
    _, name_quality = voice.split("_", 1)  # "ru_RU-dmitri-medium" -> "RU-dmitri-medium"
    _, speaker, quality = name_quality.split("-")
    base = f"ru/ru_RU/{speaker}/{quality}/{voice}.onnx"
    return base, base + ".json"


def readable(text: str) -> str:
    """Text the core prepared for LiveTTS as Piper should read it."""
    text = PAUSE.sub(", ", text)
    text = STRESSED_WORD.sub(lambda m: m.group().lower(), text)
    return normalize(unstressed(text))


def to_pcm(audio: np.ndarray, rate: int, out_rate: int) -> bytes:
    audio = np.asarray(audio)
    if audio.dtype == np.int16 or np.abs(audio).max(initial=0) > 1.5:
        audio = audio.astype(np.float32) / 32768
    if rate != out_rate:
        g = gcd(out_rate, rate)
        audio = resample_poly(audio.astype(np.float32), out_rate // g, rate // g)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


class Piper:
    def __init__(self, paths: dict[str, str]):
        from piper import PiperVoice

        self.voices = {name: PiperVoice.load(path) for name, path in paths.items()}
        self._espeak = threading.Lock()  # espeak-ng keeps global state
        self.synthesize("sofia", "Здравствуйте.", 8000).__next__()  # warm-up

    def synthesize(self, voice_id: str, text: str, rate: int) -> Iterator[bytes]:
        """PCM of ``text`` at ``rate``, sentence by sentence as each is ready."""
        voice = self.voices[VOICES.get(voice_id, FEMALE)]
        text = readable(text)
        if not SPEAKABLE.search(text):
            return
        with self._espeak:
            sentences = voice.phonemize(text)
        for phonemes in sentences:
            audio = voice.phoneme_ids_to_audio(voice.phonemes_to_ids(phonemes))
            yield to_pcm(audio[0] if isinstance(audio, tuple) else audio, voice.config.sample_rate, rate)
