"""One process tree in one container: the gRPC speech service and llama-server as its child."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import signal
import socket
import sys
import urllib.request

from . import config, models
from .config import Settings

log = logging.getLogger("worker")


def llm_command(settings: Settings) -> list[str]:
    return [
        shutil.which("llama-server") or "/app/llama-server",
        "--model", str(settings.llm_path), "--host", "0.0.0.0", "--port", str(settings.llm_port),
        "--api-key", settings.api_key,
        "--jinja",  # the model's own template: tools
        "--chat-template-kwargs", json.dumps({"reasoning_effort": settings.llm_reasoning_effort}),
        "--n-gpu-layers", str(settings.llm_gpu_layers),
        "--parallel", str(settings.llm_slots),
        "--ctx-size", str(settings.llm_slots * settings.llm_ctx_per_slot),
        "--no-webui",
    ]


async def serve(settings: Settings) -> int:
    from .server import Engines, SpeechKit, create_server
    from .stt import GigaAm
    from .tts import Piper

    models.use_models_dir(settings.models_dir)
    llm = None
    if settings.llm_enabled:
        await asyncio.to_thread(models.fetch_llm, settings)
        llm = await asyncio.create_subprocess_exec(*llm_command(settings))
    log.info("loading the speech models")
    voices = await asyncio.to_thread(models.fetch_voices)
    tts = await asyncio.to_thread(Piper, voices)
    asr = await asyncio.to_thread(GigaAm, settings.asr_model, settings.asr_threads)
    engines = Engines(asr, tts, settings.asr_workers, settings.tts_workers)
    server = create_server(SpeechKit(engines, settings.api_key), settings.grpc_port)
    await server.start()
    log.info("speech service on port %d", settings.grpc_port)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(name, stop.set)
    waits = [asyncio.create_task(stop.wait())]
    if llm is not None:
        waits.append(asyncio.create_task(llm.wait()))
    done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
    failed = waits[0] not in done
    if failed:
        log.error("llama-server exited with %s", llm.returncode)
    await server.stop(5)
    if llm is not None and llm.returncode is None:
        llm.terminate()
        await llm.wait()
    return 1 if failed else 0


def healthcheck(settings: Settings) -> int:
    with socket.create_connection(("127.0.0.1", settings.grpc_port), timeout=3):
        pass
    if settings.llm_enabled:
        with urllib.request.urlopen(f"http://127.0.0.1:{settings.llm_port}/health", timeout=5) as response:
            return 0 if response.status == 200 else 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="voice-agent-worker")
    parser.add_argument("command", choices=["serve", "healthcheck", "check"])
    parser.add_argument("--host", default="127.0.0.1", help="check: where the worker is")
    command = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # a line per file of the model downloads
    settings = config.load()
    if command.command == "healthcheck":
        return healthcheck(settings)
    if command.command == "check":
        from .check import run

        return asyncio.run(run(settings, command.host))
    return asyncio.run(serve(settings))


