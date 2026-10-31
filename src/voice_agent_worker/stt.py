"""Recognition: GigaAM over ONNX Runtime on the CPU, one utterance at a time.

The core streams the whole call, silence included, and announces the end of each utterance itself (external EOU).
GigaAM is not a streaming model, so the audio since the last EOU is cut down to the speech that ended there.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

FRAME_S = 0.02
LOUD_RMS = 150  # of 32768, about -47 dBFS: quiet phone speech is above it, line noise below
MIN_LOUD_S = 0.1  # a shorter burst is a click, not a word
MAX_GAP_S = 1.2  # an utterance has no longer pause inside: what is before it is echo or noise
PAD_S = 0.2
MAX_UTTERANCE_S = 30.0


def speech_region(pcm: np.ndarray, rate: int) -> np.ndarray | None:
    """The speech at the end of ``pcm`` (int16), or None when nothing was said."""
    frame = int(rate * FRAME_S)
    count = len(pcm) // frame
    if count == 0:
        return None
    frames = pcm[: count * frame].astype(np.float32).reshape(count, frame)
    loud = np.sqrt((frames**2).mean(axis=1)) > LOUD_RMS
    if loud.sum() * FRAME_S < MIN_LOUD_S:
        return None
    indices = np.flatnonzero(loud)
    end = int(indices[-1])
    start = end
    max_gap = int(MAX_GAP_S / FRAME_S)
    for index in indices[::-1]:
        if start - int(index) > max_gap:
            break
        start = int(index)
    pad = int(PAD_S / FRAME_S)
    first = max(max(start - pad, 0), end + pad - int(MAX_UTTERANCE_S / FRAME_S))
    return pcm[first * frame : min(end + pad + 1, count) * frame]


class GigaAm:
    def __init__(self, model: str, threads: int):
        import onnx_asr
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.asr = onnx_asr.load_model(model, sess_options=options, providers=["CPUExecutionProvider"])
        self.recognize(np.zeros(8000, dtype=np.int16), 8000)  # warm-up

    def recognize(self, pcm: np.ndarray, rate: int) -> str:
        return str(self.asr.recognize(pcm.astype(np.float32) / 32768, sample_rate=rate)).strip()
