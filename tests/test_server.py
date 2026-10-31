"""The service over a real gRPC channel, the way the core calls it, with fake models."""

import grpc
import numpy as np
import pytest
import pytest_asyncio

from voice_agent_worker.server import Engines, SpeechKit, create_server
from yandex.cloud.ai.stt.v3 import stt_pb2
from yandex.cloud.ai.tts.v3 import tts_pb2

KEY = "secret"
METADATA = (("authorization", f"Api-Key {KEY}"),)
RATE = 8000


class FakeAsr:
    def recognize(self, pcm, rate):
        return f"{len(pcm) / rate:.1f} секунд"


class FakeTts:
    def __init__(self):
        self.said = []

    def synthesize(self, voice, text, rate):
        self.said.append((voice, text.strip(), rate))
        yield b"\x01\x00" * 1000


@pytest_asyncio.fixture
async def worker():
    tts = FakeTts()
    server = create_server(SpeechKit(Engines(FakeAsr(), tts, 1, 1), KEY), 50599)
    await server.start()
    async with grpc.aio.insecure_channel("127.0.0.1:50599") as channel:
        yield channel, tts
    await server.stop(0)


def recognizer(channel):
    return channel.stream_stream("/speechkit.stt.v3.Recognizer/RecognizeStreaming",
                                 request_serializer=stt_pb2.StreamingRequest.SerializeToString,
                                 response_deserializer=stt_pb2.StreamingResponse.FromString)


def voice(seconds):
    t = np.arange(int(RATE * seconds)) / RATE
    return (np.sin(2 * np.pi * 300 * t) * 3000).astype(np.int16).tobytes()


def options():
    return stt_pb2.StreamingRequest(session_options=stt_pb2.StreamingOptions(
        recognition_model=stt_pb2.RecognitionModelOptions(audio_format=stt_pb2.AudioFormatOptions(
            raw_audio=stt_pb2.RawAudio(sample_rate_hertz=RATE)))))


async def test_a_final_per_eou_then_the_next_utterance(worker):
    channel, _ = worker
    call = recognizer(channel)(metadata=METADATA)
    await call.write(options())
    await call.write(stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=voice(1))))
    await call.write(stt_pb2.StreamingRequest(eou=stt_pb2.Eou()))
    assert (await call.read()).final.alternatives[0].text.endswith("секунд")
    assert (await call.read()).WhichOneof("Event") == "eou_update"
    await call.write(stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=bytes(8000))))
    await call.write(stt_pb2.StreamingRequest(eou=stt_pb2.Eou()))
    assert (await call.read()).WhichOneof("Event") == "eou_update"  # silence: no final
    await call.done_writing()


async def test_a_wrong_key_is_refused(worker):
    channel, _ = worker
    call = recognizer(channel)(metadata=(("authorization", "Api-Key wrong"),))
    await call.write(options())
    with pytest.raises(grpc.aio.AioRpcError) as refused:
        await call.read()
    assert refused.value.code() == grpc.StatusCode.UNAUTHENTICATED


async def test_utterance_synthesis(worker):
    channel, tts = worker
    method = channel.unary_stream("/speechkit.tts.v3.Synthesizer/UtteranceSynthesis",
                                  request_serializer=tts_pb2.UtteranceSynthesisRequest.SerializeToString,
                                  response_deserializer=tts_pb2.UtteranceSynthesisResponse.FromString)
    request = tts_pb2.UtteranceSynthesisRequest(
        text="Здравствуйте", hints=[tts_pb2.Hints(voice="denis"), tts_pb2.Hints(role="neutral")],
        output_audio_spec=tts_pb2.AudioFormatOptions(raw_audio=tts_pb2.RawAudio(sample_rate_hertz=16000)))
    chunks = [r.audio_chunk.data async for r in method(request, metadata=METADATA)]
    assert sum(map(len, chunks)) == 2000
    assert tts.said == [("denis", "Здравствуйте", 16000)]


async def test_stream_synthesis_speaks_sentences_forced_text_and_the_rest(worker):
    channel, tts = worker
    method = channel.stream_stream("/speechkit.tts.v3.Synthesizer/StreamSynthesis",
                                   request_serializer=tts_pb2.StreamSynthesisRequest.SerializeToString,
                                   response_deserializer=tts_pb2.StreamSynthesisResponse.FromString)
    call = method(metadata=METADATA)
    await call.write(tts_pb2.StreamSynthesisRequest(options=tts_pb2.SynthesisOptions(
        voice="sofia", output_audio_spec=tts_pb2.AudioFormatOptions(raw_audio=tts_pb2.RawAudio(sample_rate_hertz=RATE)))))
    await call.write(tts_pb2.StreamSynthesisRequest(synthesis_input=tts_pb2.SynthesisInput(text="Да, ")))
    await call.write(tts_pb2.StreamSynthesisRequest(force_synthesis=tts_pb2.ForceSynthesisEvent()))
    assert (await call.read()).audio_chunk.data
    await call.write(tts_pb2.StreamSynthesisRequest(synthesis_input=tts_pb2.SynthesisInput(text="конечно. Соединяю ")))
    await call.done_writing()
    audio = 1
    while (response := await call.read()) is not grpc.aio.EOF:
        audio += bool(response.audio_chunk.data)
    assert [text for _, text, _ in tts.said] == ["Да,", "конечно.", "Соединяю"]
    assert audio >= 3
