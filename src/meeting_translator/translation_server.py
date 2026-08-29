from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .translation_protocol import (
    DEFAULT_TRANSLATION_SERVER_PORT,
    translation_messages,
    translation_server_base_url,
)

DEFAULT_OLLAMA_PORT = 11434
MAX_REQUEST_BYTES = 1_000_000


class OllamaTranslationBackend:
    def __init__(self, host: str, port: int, timeout: float = 60.0) -> None:
        self.base_url = translation_server_base_url(host, port)
        self.timeout = timeout

    def check_model(self, model: str) -> None:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Ollamaに接続できません: {reason}") from exc
        available = {str(item.get("name", "")) for item in result.get("models", [])}
        if model not in available and f"{model}:latest" not in available:
            raise RuntimeError(
                f"翻訳モデル `{model}` がありません。`ollama pull {model}` を実行してください。"
            )

    def translate(
        self,
        *,
        model: str,
        text: str,
        language: str | None,
        history: list[tuple[str, str]],
    ) -> str:
        body = json.dumps(
            {
                "model": model,
                "stream": False,
                "think": False,
                "messages": translation_messages(model, text, language, history),
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 4096,
                    "num_predict": 384,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            return str(result["message"]["content"]).strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Ollama翻訳に失敗しました: {reason}") from exc


class TranslationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        backend: OllamaTranslationBackend,
    ) -> None:
        super().__init__(server_address, TranslationRequestHandler)
        self.backend = backend


class TranslationRequestHandler(BaseHTTPRequestHandler):
    server: TranslationHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - HTTP handler API
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        model = urllib.parse.parse_qs(parsed.query).get("model", [""])[0].strip()
        if not model:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "model is required"})
            return
        try:
            self.server.backend.check_model(model)
        except RuntimeError as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, {"ok": True, "model": model})

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
        if urllib.parse.urlsplit(self.path).path != "/translate":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            payload = self._read_payload()
            model = _required_string(payload, "model")
            text = _required_string(payload, "text")
            language_value = payload.get("language")
            language = str(language_value) if language_value else None
            history = _parse_history(payload.get("history", []))
            translated = self.server.backend.translate(
                model=model,
                text=text,
                language=language,
                history=history,
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, {"translation": translated})

    def _read_payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("invalid request size")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        super().log_message(format, *args)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _parse_history(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise ValueError("history must be a list")
    history: list[tuple[str, str]] = []
    for item in value[-3:]:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
        ):
            raise ValueError("each history item must contain source and Japanese strings")
        history.append((item[0], item[1]))
    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Meeting Live Translator用のOllama翻訳サーバー"
    )
    parser.add_argument("--host", default="0.0.0.0", help="待受IP（既定: 0.0.0.0）")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_TRANSLATION_SERVER_PORT, help="待受ポート"
    )
    parser.add_argument("--ollama-host", default="127.0.0.1", help="OllamaのIP/ホスト")
    parser.add_argument("--ollama-port", type=int, default=DEFAULT_OLLAMA_PORT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        backend = OllamaTranslationBackend(args.ollama_host, args.ollama_port)
        server = TranslationHTTPServer((args.host, args.port), backend)
    except (OSError, ValueError) as exc:
        print(f"翻訳サーバーを起動できません: {exc}")
        return 1
    print(f"翻訳サーバーを起動しました: http://{args.host}:{args.port}", flush=True)
    print("GUIで「別PC（翻訳サーバー）」を選び、このマシンのIPとポートを指定してください。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n翻訳サーバーを終了します。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
