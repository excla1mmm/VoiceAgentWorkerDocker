"""The gRPC API the voice core already speaks: three SpeechKit methods on one port.

The core calls Recognizer/RecognizeStreaming, Synthesizer/StreamSynthesis and Synthesizer/UtteranceSynthesis by their
method paths with the "Api-Key <key>" metadata, so it talks to this worker as to Yandex.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import grpc
import numpy as np

from yandex.cloud.ai.stt.v3 import stt_pb2
from yandex.cloud.ai.tts.v3 import tts_pb2

from .stt import MAX_UTTERANCE_S, speech_region
from .tts import CHUNK_S

log = logging.getLogger(__name__)

SENTENCE = re.compile(r".+?[.!?…]+(?:\s+|$)", re.S)
BUFFER_S = MAX_UTTERANCE_S * 2  # the audio kept since the last EOU


def take_sentences(text: str) -> tuple[list[str], str]:
    """The complete sentences of ``text`` and what is left of it."""
    sentences = [m.group() for m in SENTENCE.finditer(text)]
    return sentences, text[sum(map(len, sentences)):]


class Engines:
    """The models and the threads they run on: a call only waits for its own turn."""

    def __init__(self, asr: Any, tts: Any, asr_workers: int, tts_workers: int):
        self.asr, self.tts = asr, tts
        self._asr_pool = ThreadPoolExecutor(asr_workers, thread_name_prefix="asr")
        self._tts_pool = ThreadPoolExecutor(tts_workers, thread_name_prefix="tts")

    async def recognize(self, pcm: np.ndarray, rate: int) -> str:
        region = speech_region(pcm, rate)
        if region is None:
            return ""
        started = time.monotonic()
        text = await asyncio.get_running_loop().run_in_executor(self._asr_pool, self.asr.recognize, region, rate)
        log.info("asr %.1f s of speech in %d ms", len(region) / rate, (time.monotonic() - started) * 1000)
        return text

    async def synthesize(self, voice: str, text: str, rate: int) -> AsyncIterator[bytes]:
        """Audio in chunks; each sentence is made on a TTS thread."""
        loop = asyncio.get_running_loop()
        sentences: Iterator[bytes] = self.tts.synthesize(voice, text, rate)
        chunk = int(rate * CHUNK_S) * 2
        while (pcm := await loop.run_in_executor(self._tts_pool, next, sentences, None)) is not None:
            for start in range(0, len(pcm), chunk):
                yield pcm[start : start + chunk]


def utterance_voice(request: tts_pb2.UtteranceSynthesisRequest) -> str:
    return next((hint.voice for hint in request.hints if hint.WhichOneof("Hint") == "voice"), "")


class SpeechKit:
    def __init__(self, engines: Engines, api_key: str):
        self.engines = engines
        self.authorization = f"Api-Key {api_key}"

    async def _authorize(self, context: grpc.aio.ServicerContext) -> None:
        if dict(context.invocation_metadata() or ()).get("authorization") != self.authorization:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "wrong api key")

    async def recognize_streaming(self, requests: AsyncIterator[stt_pb2.StreamingRequest],
                                  context: grpc.aio.ServicerContext) -> AsyncIterator[stt_pb2.StreamingResponse]:
        await self._authorize(context)
        rate, held, size = 8000, [], 0
        async for request in requests:
            event = request.WhichOneof("Event")
            if event == "session_options":
                raw = request.session_options.recognition_model.audio_format.raw_audio
                rate = raw.sample_rate_hertz or rate
            elif event == "chunk":
                held.append(request.chunk.data)
                size += len(request.chunk.data)
                while size - len(held[0]) > BUFFER_S * rate * 2:
                    size -= len(held.pop(0))
            elif event == "eou":
                pcm = np.frombuffer(b"".join(held), dtype=np.int16)
                held, size = [], 0
                if text := await self.engines.recognize(pcm, rate):
                    yield stt_pb2.StreamingResponse(final=stt_pb2.AlternativeUpdate(
                        alternatives=[stt_pb2.Alternative(text=text)]))
                yield stt_pb2.StreamingResponse(eou_update=stt_pb2.EouUpdate())

    async def utterance_synthesis(self, request: tts_pb2.UtteranceSynthesisRequest,
                                  context: grpc.aio.ServicerContext) -> AsyncIterator[tts_pb2.UtteranceSynthesisResponse]:
        await self._authorize(context)
        rate = request.output_audio_spec.raw_audio.sample_rate_hertz or 8000
        async for chunk in self.engines.synthesize(utterance_voice(request), request.text, rate):
            yield tts_pb2.UtteranceSynthesisResponse(audio_chunk=tts_pb2.AudioChunk(data=chunk))

    async def stream_synthesis(self, requests: AsyncIterator[tts_pb2.StreamSynthesisRequest],
                               context: grpc.aio.ServicerContext) -> AsyncIterator[tts_pb2.StreamSynthesisResponse]:
        """The text comes in pieces: a sentence is spoken once complete, the rest when forced or at the end."""
        await self._authorize(context)
        pieces: asyncio.Queue[str | None] = asyncio.Queue()
        options = tts_pb2.SynthesisOptions()

        async def read() -> None:
            nonlocal options
            text = ""
            try:
                async for request in requests:
                    event = request.WhichOneof("Event")
                    if event == "options":
                        options = request.options
                    elif event == "synthesis_input":
                        sentences, text = take_sentences(text + request.synthesis_input.text)
                        for sentence in sentences:
                            pieces.put_nowait(sentence)
                    elif event == "force_synthesis" and text.strip():
                        pieces.put_nowait(text)
                        text = ""
                if text.strip():
                    pieces.put_nowait(text)
            finally:
                pieces.put_nowait(None)

        reader = asyncio.create_task(read())
        try:
            while (piece := await pieces.get()) is not None:
                rate = options.output_audio_spec.raw_audio.sample_rate_hertz or 8000
                async for chunk in self.engines.synthesize(options.voice, piece, rate):
                    yield tts_pb2.StreamSynthesisResponse(audio_chunk=tts_pb2.AudioChunk(data=chunk))
        finally:
            reader.cancel()


def handlers(service: SpeechKit) -> list[grpc.GenericRpcHandler]:
    return [
        grpc.method_handlers_generic_handler("speechkit.stt.v3.Recognizer", {
            "RecognizeStreaming": grpc.stream_stream_rpc_method_handler(
                service.recognize_streaming, request_deserializer=stt_pb2.StreamingRequest.FromString,
                response_serializer=stt_pb2.StreamingResponse.SerializeToString),
        }),
        grpc.method_handlers_generic_handler("speechkit.tts.v3.Synthesizer", {
            "UtteranceSynthesis": grpc.unary_stream_rpc_method_handler(
                service.utterance_synthesis, request_deserializer=tts_pb2.UtteranceSynthesisRequest.FromString,
                response_serializer=tts_pb2.UtteranceSynthesisResponse.SerializeToString),
            "StreamSynthesis": grpc.stream_stream_rpc_method_handler(
                service.stream_synthesis, request_deserializer=tts_pb2.StreamSynthesisRequest.FromString,
                response_serializer=tts_pb2.StreamSynthesisResponse.SerializeToString),
        }),
    ]


def create_server(service: SpeechKit, port: int) -> grpc.aio.Server:
    # the core pings an idle channel every 5.5 min
    server = grpc.aio.server(options=[("grpc.keepalive_permit_without_calls", 1)])
    server.add_generic_rpc_handlers(handlers(service))
    server.add_insecure_port(f"[::]:{port}")
    return server
