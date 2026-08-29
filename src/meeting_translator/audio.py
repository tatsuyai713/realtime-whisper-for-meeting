from __future__ import annotations

import platform
import queue
import subprocess
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import soundcard as sc

from .models import AudioChunk, AudioDevice

LOOPBACK_HINTS = ("loopback", "monitor", "blackhole", "vb-audio", "stereo mix")


def list_audio_devices() -> list[AudioDevice]:
    """Return capture devices, including Windows loopback and macOS virtual inputs."""
    devices: list[AudioDevice] = []
    if platform.system() == "Darwin":
        devices.append(
            AudioDevice(
                name="Macで再生中の音声（YouTube・会議・アプリ）",
                native_id="macos-system-audio",
                microphone=None,
                is_loopback=True,
                capture_kind="macos-system",
            )
        )
    include_loopback = platform.system() == "Windows"
    for index, microphone in enumerate(sc.all_microphones(include_loopback=include_loopback)):
        name = str(microphone.name)
        lowered = name.lower()
        native_id = str(getattr(microphone, "id", getattr(microphone, "_id", index)))
        is_loopback = bool(getattr(microphone, "isloopback", False)) or any(
            hint in lowered for hint in LOOPBACK_HINTS
        )
        devices.append(
            AudioDevice(
                name=name,
                native_id=native_id,
                microphone=microphone,
                is_loopback=is_loopback,
            )
        )
    return devices


def preferred_device_index(devices: list[AudioDevice]) -> int:
    if not devices:
        return -1
    for index, device in enumerate(devices):
        if device.is_loopback:
            return index
    return 0


class SpeechSegmenter:
    """Groups short PCM blocks into utterances, preferring silence boundaries."""

    def __init__(
        self,
        sample_rate: int,
        silence_threshold: float = 0.006,
        min_seconds: float = 1.0,
        max_seconds: float = 8.0,
        trailing_silence_seconds: float = 0.65,
        pre_roll_seconds: float = 0.0,
        overlap_seconds: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_samples = int(min_seconds * sample_rate)
        self.max_samples = int(max_seconds * sample_rate)
        self.trailing_silence_samples = int(trailing_silence_seconds * sample_rate)
        self.pre_roll_samples = int(pre_roll_seconds * sample_rate)
        self.overlap_samples = int(overlap_seconds * sample_rate)
        self._blocks: list[np.ndarray] = []
        self._pre_roll = np.empty(0, dtype=np.float32)
        self._total_samples = 0
        self._silence_samples = 0
        self._has_speech = False
        self._has_new_speech = False
        self._buffer_overlap_samples = 0
        self.last_output_overlap_seconds = 0.0

    def add(self, samples: np.ndarray) -> np.ndarray | None:
        mono = _to_mono_float32(samples)
        if mono.size == 0:
            return None
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
        is_speech = rms >= self.silence_threshold

        # Do not build up unbounded buffers while the meeting is silent.
        if not self._has_speech and not is_speech:
            if self.pre_roll_samples:
                self._pre_roll = np.concatenate((self._pre_roll, mono))[-self.pre_roll_samples :]
            return None

        if is_speech and not self._has_speech and self._pre_roll.size:
            self._blocks.append(self._pre_roll)
            self._total_samples += self._pre_roll.size
            self._pre_roll = np.empty(0, dtype=np.float32)
        self._blocks.append(mono)
        self._total_samples += mono.size
        if is_speech:
            self._has_speech = True
            self._has_new_speech = True
            self._silence_samples = 0
        else:
            self._silence_samples += mono.size

        silence_boundary = (
            self._total_samples >= self.min_samples
            and self._silence_samples >= self.trailing_silence_samples
        )
        if silence_boundary:
            return self.flush()
        if self._total_samples >= self.max_samples:
            return self._flush_with_overlap()
        return None

    def flush(self) -> np.ndarray | None:
        if not self._blocks or not self._has_speech or not self._has_new_speech:
            self._reset()
            return None
        result = np.concatenate(self._blocks)
        self.last_output_overlap_seconds = self._buffer_overlap_samples / self.sample_rate
        self._reset()
        return result

    def _flush_with_overlap(self) -> np.ndarray:
        result = np.concatenate(self._blocks)
        self.last_output_overlap_seconds = self._buffer_overlap_samples / self.sample_rate
        overlap = min(self.overlap_samples, result.size)
        tail = result[-overlap:].copy() if overlap else np.empty(0, dtype=np.float32)
        self._reset()
        if tail.size:
            self._blocks = [tail]
            self._total_samples = tail.size
            self._has_speech = True
            self._has_new_speech = False
            self._buffer_overlap_samples = tail.size
        return result

    def _reset(self) -> None:
        self._blocks = []
        self._total_samples = 0
        self._silence_samples = 0
        self._has_speech = False
        self._has_new_speech = False
        self._buffer_overlap_samples = 0
        self._pre_roll = np.empty(0, dtype=np.float32)


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data.reshape(-1))


class AudioCapture:
    def __init__(
        self,
        device: AudioDevice,
        output_queue: queue.Queue[AudioChunk | None],
        on_level: Callable[[float], None],
        on_error: Callable[[str], None],
        sample_rate: int = 48_000,
        max_segment_seconds: float = 8.0,
        overlap_seconds: float = 0.8,
    ) -> None:
        self.device = device
        self.output_queue = output_queue
        self.on_level = on_level
        self.on_error = on_error
        self.sample_rate = sample_rate
        self.max_segment_seconds = max_segment_seconds
        self.overlap_seconds = overlap_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        if self.device.capture_kind == "macos-system":
            self._run_macos_system_audio()
            return
        self._run_soundcard()

    def _run_soundcard(self) -> None:
        block_frames = int(self.sample_rate * 0.2)
        sequence = 1
        segmenter = SpeechSegmenter(
            self.sample_rate,
            min_seconds=0.6,
            max_seconds=self.max_segment_seconds,
            trailing_silence_seconds=0.4,
            pre_roll_seconds=0.3,
            overlap_seconds=self.overlap_seconds,
        )
        try:
            with self.device.microphone.recorder(samplerate=self.sample_rate) as recorder:
                while not self._stop_event.is_set():
                    block = recorder.record(numframes=block_frames)
                    mono = _to_mono_float32(block)
                    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
                    self.on_level(min(1.0, rms * 12.0))
                    utterance = segmenter.add(mono)
                    if utterance is not None:
                        self._put(
                            AudioChunk(
                                sequence,
                                utterance,
                                self.sample_rate,
                                datetime.now(),
                                segmenter.last_output_overlap_seconds,
                            )
                        )
                        sequence += 1
                final = segmenter.flush()
                if final is not None:
                    self._put(
                        AudioChunk(
                            sequence,
                            final,
                            self.sample_rate,
                            datetime.now(),
                            segmenter.last_output_overlap_seconds,
                        )
                    )
        except Exception as exc:  # device backends provide platform-specific exceptions
            system = platform.system()
            self.on_error(f"{system} の音声取得に失敗しました: {exc}")
        finally:
            self.on_level(0.0)

    def _run_macos_system_audio(self) -> None:
        sequence = 1
        segmenter = SpeechSegmenter(
            self.sample_rate,
            min_seconds=0.6,
            max_seconds=self.max_segment_seconds,
            trailing_silence_seconds=0.4,
            pre_roll_seconds=0.3,
            overlap_seconds=self.overlap_seconds,
        )
        process: subprocess.Popen[bytes] | None = None
        try:
            helper = _build_macos_capture_helper()
            process = subprocess.Popen(
                [str(helper)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            if process.stdout is None:
                raise RuntimeError("システム音声ヘルパーの出力を開けませんでした。")
            block_bytes = int(self.sample_rate * 0.2) * 4
            while not self._stop_event.is_set():
                raw = process.stdout.read(block_bytes)
                if not raw:
                    break
                mono = np.frombuffer(raw, dtype="<f4").copy()
                if mono.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
                self.on_level(min(1.0, rms * 12.0))
                utterance = segmenter.add(mono)
                if utterance is not None:
                    self._put(
                        AudioChunk(
                            sequence,
                            utterance,
                            self.sample_rate,
                            datetime.now(),
                            segmenter.last_output_overlap_seconds,
                        )
                    )
                    sequence += 1
            final = segmenter.flush()
            if final is not None:
                self._put(
                    AudioChunk(
                        sequence,
                        final,
                        self.sample_rate,
                        datetime.now(),
                        segmenter.last_output_overlap_seconds,
                    )
                )
            if process.poll() not in (None, 0) and not self._stop_event.is_set():
                error = process.stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(error or "システム音声キャプチャが終了しました。")
        except Exception as exc:
            self.on_error(
                "Macのシステム音声を取得できません。"
                "「システム設定 → プライバシーとセキュリティ → "
                f"画面収録とシステムオーディオ」で許可して再起動してください: {exc}"
            )
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            self.on_level(0.0)

    def _put(self, chunk: AudioChunk) -> None:
        try:
            self.output_queue.put(chunk, timeout=1.0)
        except queue.Full:
            self.on_error("処理が追いつかないため、古い音声区間を1つ破棄しました。")
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            self.output_queue.put_nowait(chunk)


def _build_macos_capture_helper() -> Path:
    if platform.system() != "Darwin":
        raise RuntimeError("ScreenCaptureKitはmacOS専用です。")
    source = Path(__file__).with_name("macos_system_audio.swift")
    if not source.exists():
        raise RuntimeError(f"macOS音声ヘルパーがありません: {source}")
    cache_dir = Path(__file__).parents[2] / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = cache_dir / "macos-system-audio-capture"
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    completed = subprocess.run(
        ["xcrun", "swiftc", "-parse-as-library", "-O", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Apple Command Line Toolsで音声ヘルパーをビルドできませんでした。"
            f" `xcode-select --install` を実行してください。\n{completed.stderr.strip()}"
        )
    return binary
