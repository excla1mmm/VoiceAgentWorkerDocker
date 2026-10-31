"""An end-to-end check through the ports the module uses: synthesis, recognition of that speech, an LLM tool call."""

from __future__ import annotations

import json
import time
import urllib.request

import grpc

from yandex.cloud.ai.stt.v3 import stt_pb2
from yandex.cloud.ai.tts.v3 import tts_pb2

from .config import Settings

PHRASE = "Здравствуйте, соедините меня, пожалуйста, с отделом продаж."
RATE = 8000
LLM_TIMEOUT_S = 180


def tool_request(settings: Settings) -> dict:
    return {
        "model": settings.llm_file, "stream": False, "max_tokens": 300, "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "Ты секретарь компании. Переводи звонки инструментом transfer."},
            {"role": "user", "content": PHRASE},
        ],
        "tools": [{"type": "function", "function": {
            "name": "transfer", "description": "Перевести звонок на отдел",
            "parameters": {"type": "object", "properties": {"department": {"type": "string"}},
                           "required": ["department"]},
        }}],
        "tool_choice": "auto",
    }


async def run(settings: Settings, host: str) -> int:
    metadata = (("authorization", f"Api-Key {settings.api_key}"),)
    async with grpc.aio.insecure_channel(f"{host}:{settings.grpc_port}") as channel:
        started = time.monotonic()
        first_ms = None
        pcm = bytearray()
        synthesize = channel.unary_stream("/speechkit.tts.v3.Synthesizer/UtteranceSynthesis",
                                          request_serializer=tts_pb2.UtteranceSynthesisRequest.SerializeToString,
                                          response_deserializer=tts_pb2.UtteranceSynthesisResponse.FromString)
        request = tts_pb2.UtteranceSynthesisRequest(
            text=PHRASE, hints=[tts_pb2.Hints(voice="sofia")],
            output_audio_spec=tts_pb2.AudioFormatOptions(raw_audio=tts_pb2.RawAudio(
                audio_encoding=tts_pb2.RawAudio.LINEAR16_PCM, sample_rate_hertz=RATE)))
        async for response in synthesize(request, metadata=metadata):
            first_ms = first_ms or (time.monotonic() - started) * 1000
            pcm += response.audio_chunk.data
        print(f"tts: {len(pcm) / 2 / RATE:.1f} s of speech, first audio in {first_ms:.0f} ms")

        recognize = channel.stream_stream("/speechkit.stt.v3.Recognizer/RecognizeStreaming",
                                          request_serializer=stt_pb2.StreamingRequest.SerializeToString,
                                          response_deserializer=stt_pb2.StreamingResponse.FromString)
        call = recognize(metadata=metadata)
        await call.write(stt_pb2.StreamingRequest(session_options=stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(audio_format=stt_pb2.AudioFormatOptions(
                raw_audio=stt_pb2.RawAudio(sample_rate_hertz=RATE))))))
        await call.write(stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=bytes(pcm))))
        started = time.monotonic()
        await call.write(stt_pb2.StreamingRequest(eou=stt_pb2.Eou()))
        text = ""
        while True:
            response = await call.read()
            if response.WhichOneof("Event") == "final":
                text = response.final.alternatives[0].text
            elif response.WhichOneof("Event") == "eou_update":
                break
        call.cancel()
        print(f"stt: {text!r} in {(time.monotonic() - started) * 1000:.0f} ms")
        if not text:
            return 1

    if not settings.llm_enabled:
        print("llm: disabled")
        return 0
    started = time.monotonic()
    llm = urllib.request.Request(f"http://{host}:{settings.llm_port}/v1/chat/completions",
                                 data=json.dumps(tool_request(settings)).encode(),
                                 headers={"Authorization": f"Bearer {settings.api_key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(llm, timeout=LLM_TIMEOUT_S) as response:
        message = json.load(response)["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    print(f"llm: {(time.monotonic() - started) * 1000:.0f} ms, tool calls: {json.dumps(calls, ensure_ascii=False)}")
    return 0 if calls else 1
