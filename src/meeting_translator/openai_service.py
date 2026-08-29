from __future__ import annotations

import io
import queue
import threading
import wave
from collections import deque
from collections.abc import Callable

import numpy as np
from openai import OpenAI

from .models import AudioChunk, TranscriptSegment, remove_repeated_transcript_prefix
from .translation_client import translate_with_server
from .translation_protocol import (
    DEFAULT_TRANSLATION_SERVER_HOST,
    DEFAULT_TRANSLATION_SERVER_PORT,
)

TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
TRANSLATION_MODEL = "gpt-5.6-sol"


def pcm_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16)
    return output.getvalue()


class OpenAIProcessor:
    def __init__(
        self,
        api_key: str,
        language: str | None,
        translation_enabled: bool,
        audio_queue: queue.Queue[AudioChunk | None],
        on_transcript: Callable[[TranscriptSegment], None],
        on_translation: Callable[[int, str], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        translation_target: str = "openai",
        ollama_model: str = "translategemma:12b",
        translation_server_host: str = DEFAULT_TRANSLATION_SERVER_HOST,
        translation_server_port: int = DEFAULT_TRANSLATION_SERVER_PORT,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.language = language
        self.translation_enabled = translation_enabled
        self.translation_target = translation_target
        self.ollama_model = ollama_model
        self.translation_server_host = translation_server_host
        self.translation_server_port = translation_server_port
        self.audio_queue = audio_queue
        self.translation_queue: queue.Queue[TranscriptSegment | None] = queue.Queue()
        self.on_transcript = on_transcript
        self.on_translation = on_translation
        self.on_status = on_status
        self.on_error = on_error
        self._transcription_thread: threading.Thread | None = None
        self._translation_thread: threading.Thread | None = None
        self._translation_history: deque[tuple[str, str]] = deque(maxlen=3)
        self._previous_transcript = ""

    def start(self) -> None:
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop, name="transcription", daemon=True
        )
        self._transcription_thread.start()
        if self.translation_enabled:
            self._translation_thread = threading.Thread(
                target=self._translation_loop, name="translation", daemon=True
            )
            self._translation_thread.start()

    def stop(self) -> None:
        self.audio_queue.put(None)
        for thread in (self._transcription_thread, self._translation_thread):
            if thread and thread.is_alive():
                # API requests cannot be cancelled safely; let daemon workers drain
                # in the background so the GUI never freezes when Stop is clicked.
                thread.join(timeout=0.1)

    def _transcription_loop(self) -> None:
        try:
            while True:
                chunk = self.audio_queue.get()
                if chunk is None:
                    break
                try:
                    self.on_status("文字起こし中…")
                    request: dict[str, object] = {
                        "model": TRANSCRIPTION_MODEL,
                        "file": ("meeting-chunk.wav", pcm_to_wav(chunk.samples, chunk.sample_rate)),
                        "prompt": (
                            "Transcribe exactly in the language being spoken. "
                            "Preserve names, technical terms, and punctuation."
                        ),
                    }
                    if self.language:
                        request["language"] = self.language
                    response = self.client.audio.transcriptions.create(**request)
                    text = remove_repeated_transcript_prefix(
                        self._previous_transcript, str(response.text)
                    )
                    if not text:
                        continue
                    self._previous_transcript = f"{self._previous_transcript} {text}".strip()[-600:]
                    segment = TranscriptSegment(
                        chunk.sequence,
                        text,
                        chunk.captured_at,
                        chunk.overlap_seconds > 0,
                    )
                    self.on_transcript(segment)
                    if self.translation_enabled:
                        self.translation_queue.put(segment)
                    self.on_status("音声を待っています…")
                except Exception as exc:
                    self.on_error(f"文字起こしAPIエラー: {exc}")
        finally:
            # Translation must receive its sentinel after every accepted audio chunk.
            if self.translation_enabled:
                self.translation_queue.put(None)

    def _translation_loop(self) -> None:
        while True:
            segment = self.translation_queue.get()
            if segment is None:
                break
            try:
                if self.translation_target == "server":
                    self.on_status("別PCで日本語翻訳中…")
                    translated = translate_with_server(
                        host=self.translation_server_host,
                        port=self.translation_server_port,
                        model=self.ollama_model,
                        text=segment.original,
                        language=self.language,
                        history=list(self._translation_history),
                    )
                    if translated:
                        self.on_translation(segment.sequence, translated)
                        self._translation_history.append((segment.original, translated))
                    self.on_status("音声を待っています…")
                    continue
                context = ""
                if self._translation_history:
                    pairs = "\n".join(
                        f"- 原文: {original}\n  日本語訳: {translated}"
                        for original, translated in self._translation_history
                    )
                    context = (
                        "\n直前の発言と訳を、固有名詞・専門用語・代名詞の一貫性を"
                        f"保つためだけに参照してください。再翻訳はしないでください。\n{pairs}"
                    )
                response = self.client.responses.create(
                    model=TRANSLATION_MODEL,
                    instructions=(
                        "あなたは会議通訳者です。入力された発言を自然で正確な日本語に翻訳してください。"
                        "固有名詞、数値、単位、専門用語、曖昧さ、丁寧さを保持し、"
                        "要約・省略・情報追加をしないでください。"
                        "説明、前置き、引用符は付けず、翻訳だけを返してください。"
                        "すでに日本語なら意味を変えず、読みやすく整えてください。" + context
                    ),
                    input=segment.original,
                )
                translated = response.output_text.strip()
                if translated:
                    self.on_translation(segment.sequence, translated)
                    self._translation_history.append((segment.original, translated))
            except Exception as exc:
                if self.translation_target == "server":
                    self.on_error(f"翻訳サーバーエラー: {exc}")
                else:
                    self.on_error(f"翻訳APIエラー: {exc}")
