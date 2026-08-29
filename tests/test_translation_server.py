import json
import queue
import threading
import urllib.parse
import urllib.request
from unittest.mock import MagicMock

import pytest

from meeting_translator.local_service import LocalProcessor
from meeting_translator.translation_client import check_translation_server
from meeting_translator.translation_protocol import translation_server_base_url
from meeting_translator.translation_server import (
    OllamaTranslationBackend,
    RocmDiagnostics,
    TranslationHTTPServer,
    build_parser,
    parse_rocminfo,
)


class FakeTranslationBackend:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def check_model(self, model: str) -> None:
        if model != "test-model":
            raise RuntimeError("model not found")

    def translate(
        self,
        *,
        model: str,
        text: str,
        language: str | None,
        history: list[tuple[str, str]],
    ) -> str:
        self.requests.append(
            {
                "model": model,
                "text": text,
                "language": language,
                "history": history,
            }
        )
        return "こんにちは"


def test_r9700_rocm_diagnostics() -> None:
    names, targets = parse_rocminfo(
        """
          Name:                    gfx1201
          Marketing Name:          AMD Radeon AI PRO R9700
          Name:                    amdgcn-amd-amdhsa--gfx1201
        """
    )
    diagnostics = RocmDiagnostics(
        rocminfo_available=True,
        device_names=names,
        targets=targets,
        kfd_exists=True,
        kfd_accessible=True,
    )
    assert names == ("AMD Radeon AI PRO R9700",)
    assert targets == ("gfx1201",)
    assert diagnostics.ready is True
    assert diagnostics.is_r9700 is True


def test_translation_server_gpu_flags() -> None:
    args = build_parser().parse_args(["--check-gpu", "--allow-cpu"])
    assert args.check_gpu is True
    assert args.allow_cpu is True


def _urlopen_context(payload: dict[str, object]) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    context = MagicMock()
    context.__enter__.return_value = response
    return context


def test_ollama_backend_requires_model_vram(monkeypatch) -> None:
    responses = iter(
        [
            _urlopen_context({"models": [{"name": "translategemma:12b"}]}),
            _urlopen_context(
                {
                    "models": [
                        {
                            "name": "translategemma:12b",
                            "size_vram": 12 * 1024**3,
                        }
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "meeting_translator.translation_server.urllib.request.urlopen",
        lambda *_args, **_kwargs: next(responses),
    )
    backend = OllamaTranslationBackend("127.0.0.1", 11434, require_gpu=True)
    assert backend.check_model("translategemma:12b") == 12 * 1024**3


def test_ollama_backend_rejects_cpu_only_model(monkeypatch) -> None:
    responses = iter(
        [
            _urlopen_context({"models": [{"name": "translategemma:12b"}]}),
            _urlopen_context({"models": []}),
            _urlopen_context({"done": True}),
            _urlopen_context(
                {
                    "models": [
                        {"name": "translategemma:12b", "size_vram": 0}
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr(
        "meeting_translator.translation_server.urllib.request.urlopen",
        lambda *_args, **_kwargs: next(responses),
    )
    backend = OllamaTranslationBackend("127.0.0.1", 11434, require_gpu=True)
    with pytest.raises(RuntimeError, match="GPUへロードされていません"):
        backend.check_model("translategemma:12b")


def test_translation_server_health_and_remote_translation() -> None:
    backend = FakeTranslationBackend()
    server = TranslationHTTPServer(("127.0.0.1", 0), backend)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        assert check_translation_server(host, port, "test-model") == (True, "")

        processor = LocalProcessor(
            language="en",
            translation_enabled=True,
            whisper_model="small",
            ollama_model="test-model",
            translation_target="server",
            translation_server_host=host,
            translation_server_port=port,
            audio_queue=queue.Queue(),
            on_transcript=lambda _segment: None,
            on_translation=lambda _sequence, _text: None,
            on_status=lambda _text: None,
            on_error=lambda _text: None,
        )
        assert processor._translate_with_server(
            "Hello", history=[("Good morning", "おはようございます")]
        ) == "こんにちは"
        assert backend.requests == [
            {
                "model": "test-model",
                "text": "Hello",
                "language": "en",
                "history": [("Good morning", "おはようございます")],
            }
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_translation_server_reports_missing_model() -> None:
    backend = FakeTranslationBackend()
    server = TranslationHTTPServer(("127.0.0.1", 0), backend)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        ready, error = check_translation_server(host, port, "missing-model")
        assert ready is False
        assert "model not found" in error
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_translation_server_rejects_invalid_request() -> None:
    backend = FakeTranslationBackend()
    server = TranslationHTTPServer(("127.0.0.1", 0), backend)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    request = urllib.request.Request(
        f"http://{host}:{port}/translate",
        data=json.dumps({"model": "test-model"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=2)
        assert caught.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_translation_server_url_supports_hostnames_and_ipv6() -> None:
    assert translation_server_base_url("translator.local", 8765) == (
        "http://translator.local:8765"
    )
    assert translation_server_base_url("2001:db8::1", 8765) == "http://[2001:db8::1]:8765"
    with pytest.raises(ValueError):
        translation_server_base_url("http://example.com", 8765)
