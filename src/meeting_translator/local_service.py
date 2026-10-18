from __future__ import annotations

import io
import json
import platform
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
from openai import OpenAI

from .models import AudioChunk, TranscriptSegment, remove_repeated_transcript_prefix
from .openai_service import pcm_to_wav
from .performance import format_realtime_speed
from .translation_client import translate_with_server
from .translation_protocol import (
    DEFAULT_TRANSLATION_SERVER_HOST,
    DEFAULT_TRANSLATION_SERVER_PORT,
)
from .translation_protocol import (
    translation_messages as _translation_messages,
)

DEFAULT_WHISPER_MODEL = "small"
DEFAULT_OLLAMA_MODEL = "translategemma:4b"
LOCAL_TRANSLATION_MODELS = (
    ("高速・軽量 (TranslateGemma 4B)", "translategemma:4b"),
    ("高精度 (TranslateGemma 12B)", "translategemma:12b"),
    ("高速・汎用 (Qwen 2.5 3B)", "qwen2.5:3b"),
    ("高精度・汎用 (Qwen 2.5 7B)", "qwen2.5:7b"),
)
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
WHISPERKIT_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "small-fast": "small_216MB",
    "medium": "medium",
    "turbo": "large-v3-v20240930_turbo_632MB",
    "large-v3-compressed": "large-v3-v20240930_626MB",
    "large-v3": "large-v3-v20240930",
}
MLX_MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-compressed": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small-fast": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}
FASTER_WHISPER_MODELS = {
    "small-fast": "small",
    "large-v3-compressed": "large-v3",
}
WHISPERKIT_COMPUTE_UNITS = {
    "ane": ("cpuAndNeuralEngine", "cpuAndNeuralEngine"),
    "hybrid": ("cpuAndNeuralEngine", "cpuAndGPU"),
    "gpu": ("cpuAndGPU", "cpuAndGPU"),
}


def uses_mlx() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def whisperkit_available() -> bool:
    return uses_mlx() and shutil.which("whisperkit-cli") is not None


def cuda_available() -> bool:
    """Return whether CTranslate2 can run faster-whisper on an NVIDIA CUDA GPU."""
    try:
        import ctranslate2

        return "float16" in ctranslate2.get_supported_compute_types("cuda")
    except (ImportError, RuntimeError):
        return False


def local_engine_label(engine: str = "auto") -> str:
    if engine == "whisperkit" or (engine == "auto" and whisperkit_available()):
        return "WhisperKit / Neural Engine"
    if engine == "mlx" or (engine == "auto" and uses_mlx()):
        return "MLX / Metal GPU"
    if engine == "cuda":
        return "faster-whisper / NVIDIA CUDA"
    if engine == "cpu":
        return "faster-whisper / CPU"
    return "faster-whisper / CUDA" if cuda_available() else "faster-whisper / CPU"


def check_ollama(model: str) -> tuple[bool, str]:
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/tags",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"Ollamaに接続できません: {reason}"

    available = {str(item.get("name", "")) for item in result.get("models", [])}
    if model not in available and f"{model}:latest" not in available:
        return (
            False,
            f"翻訳モデル `{model}` がありません。`ollama pull {model}` を実行してください。",
        )
    return True, ""


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    source = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or source.size == 0:
        return source
    target_size = max(1, round(source.size * target_rate / source_rate))
    source_positions = np.arange(source.size, dtype=np.float64)
    target_positions = np.linspace(0, source.size - 1, target_size, dtype=np.float64)
    return np.interp(target_positions, source_positions, source).astype(np.float32)


class LocalProcessor:
    """Runs speech recognition locally and optionally translates through Ollama."""

    def __init__(
        self,
        language: str | None,
        translation_enabled: bool,
        whisper_model: str,
        ollama_model: str,
        audio_queue: queue.Queue[AudioChunk | None],
        on_transcript: Callable[[TranscriptSegment], None],
        on_translation: Callable[[int, str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        local_engine: str = "auto",
        translation_target: str = "local",
        translation_server_host: str = DEFAULT_TRANSLATION_SERVER_HOST,
        translation_server_port: int = DEFAULT_TRANSLATION_SERVER_PORT,
        whisperkit_compute_profile: str = "ane",
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        self.language = language
        self.translation_enabled = translation_enabled
        self.whisper_model = whisper_model
        self.ollama_model = ollama_model
        self.local_engine = local_engine
        self.translation_target = translation_target
        self.translation_server_host = translation_server_host
        self.translation_server_port = translation_server_port
        self.whisperkit_compute_profile = whisperkit_compute_profile
        self.audio_queue = audio_queue
        self.translation_queue: queue.Queue[TranscriptSegment | None] = queue.Queue()
        self.on_transcript = on_transcript
        self.on_translation = on_translation
        self.on_status = on_status
        self.on_error = on_error
        self.on_ready = on_ready or (lambda: None)
        self._transcription_thread: threading.Thread | None = None
        self._translation_thread: threading.Thread | None = None
        self._whisperkit_process: subprocess.Popen[bytes] | None = None
        self._stop_event = threading.Event()
        self._translation_history: deque[tuple[str, str]] = deque(maxlen=3)
        self._previous_transcript = ""
        self._last_speed = ""

    def _set_idle_status(self) -> None:
        suffix = f"（{self._last_speed}）" if self._last_speed else ""
        self.on_status(f"音声を待っています…{suffix}")

    def _record_speed(self, chunk: AudioChunk, elapsed: float) -> None:
        self._last_speed = format_realtime_speed(
            chunk.samples.size / chunk.sample_rate,
            elapsed,
        )
        self._set_idle_status()

    def start(self) -> None:
        self._stop_event.clear()
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop,
            name="local-transcription",
            daemon=True,
        )
        self._transcription_thread.start()
        if self.translation_enabled:
            self._translation_thread = threading.Thread(
                target=self._translation_loop,
                name="local-translation",
                daemon=True,
            )
            self._translation_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._whisperkit_process and self._whisperkit_process.poll() is None:
            self._whisperkit_process.terminate()
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.audio_queue.put_nowait(None)
        for thread in (self._transcription_thread, self._translation_thread):
            if thread and thread.is_alive():
                thread.join(timeout=0.1)

    def _transcription_loop(self) -> None:
        use_whisperkit = self.local_engine == "whisperkit" or (
            self.local_engine == "auto" and whisperkit_available()
        )
        if use_whisperkit:
            self._transcription_loop_whisperkit_supervised()
        elif self.local_engine == "mlx" or (self.local_engine == "auto" and uses_mlx()):
            self._transcription_loop_mlx()
        else:
            self._transcription_loop_faster_whisper()

    def _transcription_loop_whisperkit_supervised(self) -> None:
        failures = 0
        try:
            while not self._stop_event.is_set():
                should_restart = self._run_whisperkit_session()
                if not should_restart:
                    break
                failures += 1
                delay = min(5.0, 0.5 * failures)
                self.on_status(f"Neural Engineを自動再起動中…（{failures}回目）")
                self._stop_event.wait(delay)
        finally:
            if self.translation_enabled:
                self.translation_queue.put(None)

    def _run_whisperkit_session(self) -> bool:
        binary = shutil.which("whisperkit-cli")
        if binary is None:
            self.on_error(
                "WhisperKitがありません。`brew install whisperkit-cli` を実行するか、"
                "Metal GPUエンジンを選んでください。"
            )
            return False

        model = WHISPERKIT_MODELS.get(self.whisper_model, self.whisper_model)
        encoder_compute, decoder_compute = WHISPERKIT_COMPUTE_UNITS.get(
            self.whisperkit_compute_profile,
            WHISPERKIT_COMPUTE_UNITS["ane"],
        )
        cache_root = Path(__file__).parents[2] / ".cache"
        model_cache = cache_root / "whisperkit"
        tokenizer_cache = cache_root / "whisperkit-tokenizers"
        model_cache.mkdir(parents=True, exist_ok=True)
        tokenizer_cache.mkdir(parents=True, exist_ok=True)
        port = _available_local_port()
        command = [
            binary,
            "serve",
            "--model",
            model,
            "--download-model-path",
            str(model_cache),
            "--download-tokenizer-path",
            str(tokenizer_cache),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--audio-encoder-compute-units",
            encoder_compute,
            "--text-decoder-compute-units",
            decoder_compute,
            "--without-timestamps",
            "--use-prefill-prompt",
            "--skip-special-tokens",
            "--chunking-strategy",
            "none",
            "--no-speech-threshold",
            "1.0",
            "--temperature-fallback-count",
            "0",
            "--concurrent-worker-count",
            "1",
        ]
        failed = False
        try:
            self.on_status(
                f"Neural Engineモデル {self.whisper_model} を準備中…（初回のみ数分かかります）"
            )
            self._whisperkit_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _wait_for_local_port(
                port,
                process=self._whisperkit_process,
                stop_event=self._stop_event,
            )
            client = OpenAI(
                api_key="whisperkit-local",
                base_url=f"http://127.0.0.1:{port}/v1",
                timeout=120,
            )
            self._set_idle_status()
            self.on_ready()
            while not self._stop_event.is_set():
                chunk = self.audio_queue.get()
                if chunk is None:
                    break
                if self._whisperkit_process.poll() is not None:
                    raise RuntimeError("WhisperKitサーバーが予期せず終了しました")
                try:
                    self.on_status("Neural Engineで文字起こし中…")
                    request: dict[str, object] = {
                        "model": model,
                        "file": (
                            "meeting-chunk.wav",
                            pcm_to_wav(chunk.samples, chunk.sample_rate),
                        ),
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["word"],
                        "temperature": 0.0,
                    }
                    if self.language:
                        request["language"] = self.language
                    started = time.perf_counter()
                    response = client.audio.transcriptions.create(**request)
                    elapsed = time.perf_counter() - started
                    self._record_speed(chunk, elapsed)
                    text = self._finalize_transcript(
                        _text_after_audio_overlap(
                            response,
                            chunk.overlap_seconds,
                            self.language,
                            self._previous_transcript,
                        )
                    )
                    if not text:
                        continue
                    result = TranscriptSegment(
                        chunk.sequence,
                        text,
                        chunk.captured_at,
                        chunk.overlap_seconds > 0,
                    )
                    self.on_transcript(result)
                    if self.translation_enabled:
                        self.translation_queue.put(result)
                    self._set_idle_status()
                except Exception as exc:
                    error_kind = type(exc).__name__.lower()
                    server_stopped = self._whisperkit_process.poll() is not None
                    connection_failed = "connection" in error_kind or "timeout" in error_kind
                    if server_stopped or connection_failed:
                        raise RuntimeError(f"WhisperKitとの接続が切れました: {exc}") from exc
                    self.on_error(f"WhisperKit文字起こしエラー: {exc}")
        except Exception as exc:
            if not self._stop_event.is_set():
                self.on_error(f"Neural Engineモデルを起動できません: {exc}")
                failed = True
        finally:
            if self._whisperkit_process and self._whisperkit_process.poll() is None:
                self._whisperkit_process.terminate()
                try:
                    self._whisperkit_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._whisperkit_process.kill()
            self._whisperkit_process = None
        return failed

    def _transcription_loop_mlx(self) -> None:
        try:
            model_path = MLX_MODELS.get(self.whisper_model, MLX_MODELS["turbo"])
            self.on_status(f"MLX Metalモデル {self.whisper_model} を読み込み中…")
            try:
                import mlx_whisper
            except ImportError:
                self.on_error(
                    "Mac用MLX文字起こしが未インストールです。"
                    'ターミナルで `python -m pip install -e ".[local]"` を実行してください。'
                )
                return
            self._set_idle_status()
            self.on_ready()
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    break
                try:
                    self.on_status("Metal GPUで文字起こし中…")
                    started = time.perf_counter()
                    result_data = mlx_whisper.transcribe(
                        resample_audio(chunk.samples, chunk.sample_rate),
                        path_or_hf_repo=model_path,
                        language=self.language,
                        condition_on_previous_text=False,
                        temperature=0.0,
                        fp16=True,
                        verbose=None,
                    )
                    elapsed = time.perf_counter() - started
                    self._record_speed(chunk, elapsed)
                    text = self._finalize_transcript(str(result_data["text"]))
                    if not text:
                        continue
                    result = TranscriptSegment(
                        chunk.sequence,
                        text,
                        chunk.captured_at,
                        chunk.overlap_seconds > 0,
                    )
                    self.on_transcript(result)
                    if self.translation_enabled:
                        self.translation_queue.put(result)
                    self._set_idle_status()
                except Exception as exc:
                    self.on_error(f"MLX文字起こしエラー: {exc}")
        except Exception as exc:
            self.on_error(f"MLX音声モデルを読み込めません: {exc}")
        finally:
            if self.translation_enabled:
                self.translation_queue.put(None)

    def _transcription_loop_faster_whisper(self) -> None:
        try:
            if self.local_engine == "cuda" and not cuda_available():
                self.on_error(
                    "NVIDIA CUDA GPUを利用できません。NVIDIAドライバーとCUDA対応の"
                    "CTranslate2を確認するか、CPUまたは自動を選んでください。"
                )
                return
            self.on_status(f"ローカル音声モデル {self.whisper_model} を読み込み中…")
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                self.on_error(
                    "ローカル文字起こしが未インストールです。"
                    'ターミナルで `python -m pip install -e ".[local]"` を実行してください。'
                )
                return

            model_name = FASTER_WHISPER_MODELS.get(self.whisper_model, self.whisper_model)
            device, compute_type = self._faster_whisper_settings()
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            self._set_idle_status()
            self.on_ready()
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    break
                try:
                    self.on_status("ローカルで文字起こし中…")
                    started = time.perf_counter()
                    segments, _info = model.transcribe(
                        io.BytesIO(pcm_to_wav(chunk.samples, chunk.sample_rate)),
                        language=self.language,
                        beam_size=1,
                        condition_on_previous_text=False,
                    )
                    text = self._finalize_transcript("".join(segment.text for segment in segments))
                    elapsed = time.perf_counter() - started
                    self._record_speed(chunk, elapsed)
                    if not text:
                        continue
                    result = TranscriptSegment(
                        chunk.sequence,
                        text,
                        chunk.captured_at,
                        chunk.overlap_seconds > 0,
                    )
                    self.on_transcript(result)
                    if self.translation_enabled:
                        self.translation_queue.put(result)
                    self._set_idle_status()
                except Exception as exc:
                    self.on_error(f"ローカル文字起こしエラー: {exc}")
        except Exception as exc:
            self.on_error(f"ローカル音声モデルを読み込めません: {exc}")
        finally:
            if self.translation_enabled:
                self.translation_queue.put(None)

    def _faster_whisper_settings(self) -> tuple[str, str]:
        if self.local_engine == "cuda":
            return "cuda", "float16"
        if self.local_engine == "cpu":
            return "cpu", "int8"
        if cuda_available():
            return "cuda", "float16"
        return "cpu", "int8"

    def _translation_loop(self) -> None:
        while True:
            segment = self.translation_queue.get()
            if segment is None:
                break
            try:
                if self.translation_target == "server":
                    self.on_status("別PCで日本語翻訳中…")
                    translated = self._translate_with_server(
                        segment.original, history=list(self._translation_history)
                    )
                else:
                    self.on_status("ローカルで日本語翻訳中…")
                    translated = self._translate_with_ollama(
                        segment.original, history=list(self._translation_history)
                    )
                if translated:
                    self.on_translation(segment.sequence, translated)
                    self._translation_history.append((segment.original, translated))
                self._set_idle_status()
            except Exception as exc:
                if self.translation_target == "server":
                    self.on_error(f"翻訳サーバーエラー: {exc}")
                else:
                    self.on_error(
                        "Ollama翻訳エラー: "
                        f"{exc}。Ollamaを起動し `ollama pull {self.ollama_model}` "
                        "を実行してください。"
                    )

    def _finalize_transcript(self, text: str) -> str:
        cleaned = remove_repeated_transcript_prefix(self._previous_transcript, text)
        if cleaned:
            self._previous_transcript = f"{self._previous_transcript} {cleaned}".strip()[-600:]
        return cleaned

    def _translate_with_ollama(
        self,
        text: str,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        messages = _translation_messages(
            model=self.ollama_model,
            text=text,
            language=self.language,
            history=history or [],
        )
        body = json.dumps(
            {
                "model": self.ollama_model,
                "stream": False,
                "think": False,
                "messages": messages,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 4096,
                    "num_predict": 384,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            OLLAMA_CHAT_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc
        return str(result["message"]["content"]).strip()

    def _translate_with_server(
        self,
        text: str,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        return translate_with_server(
            host=self.translation_server_host,
            port=self.translation_server_port,
            model=self.ollama_model,
            text=text,
            language=self.language,
            history=history or [],
        )


def _text_after_audio_overlap(
    response: object,
    overlap_seconds: float,
    language: str | None = None,
    previous_text: str = "",
) -> str:
    raw_text = str(getattr(response, "text", ""))
    contains_cjk = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", raw_text))
    if language in {"ja", "zh", "ko"} or contains_cjk:
        return raw_text
    words = getattr(response, "words", None)
    if not words or overlap_seconds <= 0:
        return raw_text
    overlapped = [
        str(getattr(word, "word", ""))
        for word in words
        if float(getattr(word, "end", 0.0)) <= overlap_seconds
    ]
    overlap_text = "".join(overlapped).strip()
    if not overlap_text:
        return raw_text
    # Only discard audio proven to be present in the preceding transcript.
    # Otherwise words first recognized by this chunk would be lost.
    if remove_repeated_transcript_prefix(previous_text, overlap_text):
        return raw_text
    retained = [
        str(getattr(word, "word", ""))
        for word in words
        if float(getattr(word, "end", 0.0)) > overlap_seconds
    ]
    return "".join(retained).strip()


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_local_port(
    port: int,
    process: subprocess.Popen[bytes],
    stop_event: threading.Event,
    timeout_seconds: float = 600,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if stop_event.is_set():
            raise RuntimeError("起動を中止しました")
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"WhisperKitが終了しました（終了コード {return_code}）")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("WhisperKitの起動が10分以内に完了しませんでした")
