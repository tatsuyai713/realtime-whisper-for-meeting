from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from .translation_protocol import translation_server_base_url


def check_translation_server(
    host: str,
    port: int,
    model: str,
) -> tuple[bool, str]:
    try:
        base_url = translation_server_base_url(host, port)
    except ValueError as exc:
        return False, f"翻訳サーバーの指定が正しくありません: {exc}"
    query = urllib.parse.urlencode({"model": model})
    request = urllib.request.Request(f"{base_url}/health?{query}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            result = json.loads(exc.read().decode("utf-8"))
            reason = result.get("error", exc.reason)
        except (json.JSONDecodeError, UnicodeDecodeError):
            reason = exc.reason
        return False, f"翻訳サーバーを利用できません: {reason}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"翻訳サーバーに接続できません: {reason}"
    if not result.get("ok"):
        return False, f"翻訳サーバーを利用できません: {result.get('error', 'unknown error')}"
    return True, ""


def translate_with_server(
    *,
    host: str,
    port: int,
    model: str,
    text: str,
    language: str | None,
    history: list[tuple[str, str]],
) -> str:
    base_url = translation_server_base_url(host, port)
    body = json.dumps(
        {
            "model": model,
            "text": text,
            "language": language,
            "history": history,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/translate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read().decode("utf-8")).get("error", exc.reason)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error = exc.reason
        raise RuntimeError(str(error)) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(str(reason)) from exc
    try:
        return str(result["translation"]).strip()
    except KeyError as exc:
        raise RuntimeError("翻訳サーバーから不正な応答を受信しました") from exc
