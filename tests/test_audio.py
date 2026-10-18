import platform
import queue
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from meeting_translator.audio import (
    SpeechSegmenter,
    _build_macos_capture_helper,
    list_audio_devices,
    preferred_device_index,
)
from meeting_translator.local_service import (
    WHISPERKIT_COMPUTE_UNITS,
    WHISPERKIT_MODELS,
    LocalProcessor,
    _text_after_audio_overlap,
    _translation_messages,
    check_ollama,
    resample_audio,
)
from meeting_translator.models import (
    AudioChunk,
    AudioDevice,
    TranscriptSegment,
    merge_continued_transcript,
    remove_repeated_transcript_prefix,
    split_transcript_sentences,
)
from meeting_translator.openai_service import (
    TRANSCRIPTION_MODEL,
    TRANSLATION_MODEL,
    OpenAIProcessor,
    pcm_to_wav,
)
from meeting_translator.performance import format_realtime_speed, resolve_local_performance


def test_segmenter_ignores_silence_and_flushes_after_speech() -> None:
    segmenter = SpeechSegmenter(
        sample_rate=100,
        silence_threshold=0.01,
        min_seconds=0.2,
        max_seconds=5.0,
        trailing_silence_seconds=0.2,
    )
    assert segmenter.add(np.zeros(30, dtype=np.float32)) is None
    assert segmenter.add(np.full(30, 0.2, dtype=np.float32)) is None
    result = segmenter.add(np.zeros(20, dtype=np.float32))
    assert result is not None
    assert len(result) == 50


def test_segmenter_flushes_at_max_duration() -> None:
    segmenter = SpeechSegmenter(
        sample_rate=100,
        silence_threshold=0.01,
        min_seconds=0.2,
        max_seconds=0.5,
        trailing_silence_seconds=0.2,
    )
    assert segmenter.add(np.full(30, 0.2, dtype=np.float32)) is None
    result = segmenter.add(np.full(30, 0.2, dtype=np.float32))
    assert result is not None
    assert len(result) == 60


def test_segmenter_retains_overlap_after_forced_split() -> None:
    segmenter = SpeechSegmenter(
        sample_rate=100,
        min_seconds=0.2,
        max_seconds=0.5,
        trailing_silence_seconds=0.2,
        overlap_seconds=0.1,
    )
    first = segmenter.add(np.full(50, 0.2, dtype=np.float32))
    assert first is not None
    assert segmenter.last_output_overlap_seconds == 0.0
    second = segmenter.add(np.full(40, 0.2, dtype=np.float32))
    assert second is not None
    assert len(second) == 50
    assert segmenter.last_output_overlap_seconds == 0.1


def test_segmenter_keeps_audio_just_before_speech() -> None:
    segmenter = SpeechSegmenter(
        sample_rate=100,
        min_seconds=0.2,
        max_seconds=5.0,
        trailing_silence_seconds=0.2,
        pre_roll_seconds=0.2,
    )
    assert segmenter.add(np.zeros(30, dtype=np.float32)) is None
    assert segmenter.add(np.full(30, 0.2, dtype=np.float32)) is None
    result = segmenter.add(np.zeros(20, dtype=np.float32))
    assert result is not None
    assert len(result) == 70


def test_preferred_device_uses_loopback() -> None:
    devices = [
        AudioDevice("Microphone", "1", object(), False),
        AudioDevice("Loopback", "2", object(), True),
    ]
    assert preferred_device_index(devices) == 1


def test_macos_system_audio_is_available(monkeypatch) -> None:
    monkeypatch.setattr("meeting_translator.audio.platform.system", lambda: "Darwin")
    monkeypatch.setattr("meeting_translator.audio.sc.all_microphones", lambda **_kwargs: [])
    devices = list_audio_devices()
    assert devices[0].capture_kind == "macos-system"
    assert devices[0].is_loopback is True


def test_windows_loopback_attribute_is_recognized(monkeypatch) -> None:
    microphone = SimpleNamespace(name="Speakers", id="speaker-1", isloopback=True)
    monkeypatch.setattr("meeting_translator.audio.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "meeting_translator.audio.sc.all_microphones",
        lambda **_kwargs: [microphone],
    )
    devices = list_audio_devices()
    assert devices[0].name == "Speakers"
    assert devices[0].is_loopback is True


def test_pcm_to_wav_has_riff_header() -> None:
    output = pcm_to_wav(np.zeros(100, dtype=np.float32), 48_000)
    assert output[:4] == b"RIFF"
    assert output[8:12] == b"WAVE"


def test_resample_audio_to_whisper_rate() -> None:
    source = np.linspace(-1.0, 1.0, 48_000, dtype=np.float32)
    result = resample_audio(source, 48_000)
    assert result.dtype == np.float32
    assert len(result) == 16_000


def test_mac_profiles_use_optimized_whisperkit_models() -> None:
    fast = resolve_local_performance(
        "mac-fast",
        custom_model="tiny",
        custom_engine="mlx",
        custom_compute_profile="gpu",
    )
    balanced = resolve_local_performance(
        "mac-balanced",
        custom_model="tiny",
        custom_engine="mlx",
        custom_compute_profile="gpu",
    )

    assert fast.model == "small-fast"
    assert fast.max_segment_seconds == 5.0
    assert fast.overlap_seconds == 0.4
    assert WHISPERKIT_MODELS[fast.model] == "small_216MB"
    assert WHISPERKIT_MODELS[balanced.model] == "large-v3-v20240930_turbo_632MB"
    assert WHISPERKIT_COMPUTE_UNITS[fast.compute_profile] == (
        "cpuAndNeuralEngine",
        "cpuAndNeuralEngine",
    )


def test_legacy_m4_profile_names_remain_supported() -> None:
    legacy = resolve_local_performance(
        "m4-fast",
        custom_model="tiny",
        custom_engine="mlx",
        custom_compute_profile="gpu",
    )
    assert legacy == resolve_local_performance(
        "mac-fast",
        custom_model="tiny",
        custom_engine="mlx",
        custom_compute_profile="gpu",
    )


def test_realtime_speed_format() -> None:
    assert format_realtime_speed(5.0, 1.0) == "RTF 0.20（実時間の5.0倍速）"
    assert format_realtime_speed(0.0, 1.0) == ""


def test_cuda_engine_uses_float16_and_cpu_uses_int8() -> None:
    processor = LocalProcessor(
        language="en",
        translation_enabled=False,
        whisper_model="small",
        ollama_model="test-model",
        local_engine="cuda",
        audio_queue=queue.Queue(),
        on_transcript=lambda _segment: None,
        on_translation=lambda _sequence, _text: None,
        on_status=lambda _text: None,
        on_error=lambda _text: None,
    )
    assert processor._faster_whisper_settings() == ("cuda", "float16")
    processor.local_engine = "cpu"
    assert processor._faster_whisper_settings() == ("cpu", "int8")


def test_repeated_words_from_audio_overlap_are_removed() -> None:
    previous = "We need to roll back the canary deployment"
    current = "the canary deployment before Friday."
    assert remove_repeated_transcript_prefix(previous, current) == "before Friday."


def test_repeated_japanese_text_from_audio_overlap_is_removed() -> None:
    previous = "次のリリースについて確認します"
    current = "について確認します。問題はありません"
    assert remove_repeated_transcript_prefix(previous, current) == "問題はありません"


def test_similar_repeated_prefix_is_removed() -> None:
    previous = "We discussed the general availability date"
    current = "the general-availability date before closing."
    assert remove_repeated_transcript_prefix(previous, current) == "before closing."


def test_word_timestamps_remove_only_the_overlapped_audio() -> None:
    response = SimpleNamespace(
        text="Falcon was delayed by two weeks.",
        words=[
            SimpleNamespace(word=" Falcon", end=0.25),
            SimpleNamespace(word=" was", end=0.45),
            SimpleNamespace(word=" delayed", end=0.72),
            SimpleNamespace(word=" by", end=0.88),
            SimpleNamespace(word=" two", end=1.1),
            SimpleNamespace(word=" weeks.", end=1.4),
        ],
    )
    assert (
        _text_after_audio_overlap(
            response,
            0.8,
            previous_text="Project Falcon was delayed",
        )
        == "by two weeks."
    )
    assert (
        _text_after_audio_overlap(
            response,
            0.8,
            previous_text="The previous sentence ended here.",
        )
        == "Falcon was delayed by two weeks."
    )


def test_japanese_word_timestamps_do_not_cut_subwords() -> None:
    response = SimpleNamespace(
        text="デルは内装がとにかく光っている",
        words=[SimpleNamespace(word="デル", end=0.4)],
    )
    assert (
        _text_after_audio_overlap(response, 0.8, language="ja") == "デルは内装がとにかく光っている"
    )


def test_japanese_continuation_repairs_particle_and_partial_word() -> None:
    assert (
        merge_continued_transcript(
            "最近こういう高級なモデルが",
            "デルは内装がとにかく光っている",
        )
        == "最近こういう高級なモデルは内装がとにかく光っている"
    )
    assert (
        merge_continued_transcript("最先端な近未来的な内装", "装にずっと方向性を")
        == "最先端な近未来的な内装にずっと方向性を"
    )


def test_english_continuation_repairs_partial_word_and_case_changes() -> None:
    assert (
        merge_continued_transcript(
            "We tested the deploy",
            "deployment before release.",
        )
        == "We tested the deployment before release."
    )
    assert (
        merge_continued_transcript(
            "We reviewed the Canary Deployment",
            "canary deployment before release.",
        )
        == "We reviewed the canary deployment before release."
    )


def test_english_continuation_repairs_near_duplicate_boundary_phrase() -> None:
    assert (
        merge_continued_transcript(
            "This is the new Toyota Yara",
            "Toyota Yaris GR, it's effectively a rally car.",
        )
        == "This is the new Toyota Yaris GR, it's effectively a rally car."
    )
    assert (
        merge_continued_transcript(
            "And then I'll see what they feel like when you're driving.",
            "like when you drive it? Do they add up?",
        )
        == "And then I'll see what they feel like when you drive it? Do they add up?"
    )
    assert merge_continued_transcript("The roof is 10", "10cm lower.") == (
        "The roof is 10cm lower."
    )


def test_english_continuation_preserves_different_or_intentional_phrases() -> None:
    assert (
        merge_continued_transcript(
            "We compared the Toyota Yaris",
            "Toyota Corolla is a different car.",
        )
        == "We compared the Toyota Yaris Toyota Corolla is a different car."
    )
    assert (
        merge_continued_transcript(
            "I want",
            "I won't change it.",
        )
        == "I want I won't change it."
    )


def test_display_splits_sentences_even_when_next_one_starts_lowercase() -> None:
    assert split_transcript_sentences(
        "It is a rally car for the road. of the cars I wanted to test. and then we drove it."
    ) == [
        "It is a rally car for the road.",
        "of the cars I wanted to test.",
        "and then we drove it.",
    ]


def test_display_splits_japanese_sentences_without_spaces() -> None:
    assert split_transcript_sentences("最初の文です。次の文です！最後の文です？") == [
        "最初の文です。",
        "次の文です！",
        "最後の文です？",
    ]


def test_processor_transcribes_then_translates() -> None:
    audio_queue = queue.Queue()
    transcripts = []
    translations = []
    translated = threading.Event()
    processor = OpenAIProcessor(
        api_key="test-key",
        language="en",
        translation_enabled=True,
        audio_queue=audio_queue,
        on_transcript=transcripts.append,
        on_translation=lambda sequence, text: (
            translations.append((sequence, text)),
            translated.set(),
        ),
        on_status=lambda _text: None,
        on_error=lambda message: (_ for _ in ()).throw(AssertionError(message)),
    )
    processor.client = MagicMock()
    processor.client.audio.transcriptions.create.return_value = SimpleNamespace(
        text="Hello, everyone."
    )
    processor.client.responses.create.return_value = SimpleNamespace(
        output_text="皆さん、こんにちは。"
    )

    processor.start()
    audio_queue.put(AudioChunk(1, np.ones(100, dtype=np.float32) * 0.1, 16_000, datetime.now()))
    assert translated.wait(timeout=2.0)
    processor.stop()

    assert transcripts[0].original == "Hello, everyone."
    assert translations == [(1, "皆さん、こんにちは。")]
    transcription_call = processor.client.audio.transcriptions.create.call_args.kwargs
    translation_call = processor.client.responses.create.call_args.kwargs
    assert transcription_call["model"] == TRANSCRIPTION_MODEL
    assert transcription_call["language"] == "en"
    assert translation_call["model"] == TRANSLATION_MODEL


def test_openai_processor_can_disable_translation() -> None:
    audio_queue = queue.Queue()
    transcribed = threading.Event()
    processor = OpenAIProcessor(
        api_key="test-key",
        language=None,
        translation_enabled=False,
        audio_queue=audio_queue,
        on_transcript=lambda _segment: transcribed.set(),
        on_translation=lambda _sequence, _text: None,
        on_status=lambda _text: None,
        on_error=lambda message: (_ for _ in ()).throw(AssertionError(message)),
    )
    processor.client = MagicMock()
    processor.client.audio.transcriptions.create.return_value = SimpleNamespace(text="Hello.")
    processor.start()
    audio_queue.put(AudioChunk(1, np.ones(100, dtype=np.float32) * 0.1, 16_000, datetime.now()))
    assert transcribed.wait(timeout=2.0)
    processor.stop()
    processor.client.responses.create.assert_not_called()


def test_openai_transcription_can_use_remote_translation(monkeypatch) -> None:
    translations = []
    requests = []
    monkeypatch.setattr(
        "meeting_translator.openai_service.translate_with_server",
        lambda **kwargs: (requests.append(kwargs), "こんにちは")[1],
    )
    processor = OpenAIProcessor(
        api_key="test-key",
        language="en",
        translation_enabled=True,
        translation_target="server",
        ollama_model="test-model",
        translation_server_host="192.168.1.50",
        translation_server_port=9000,
        audio_queue=queue.Queue(),
        on_transcript=lambda _segment: None,
        on_translation=lambda sequence, text: translations.append((sequence, text)),
        on_status=lambda _text: None,
        on_error=lambda message: (_ for _ in ()).throw(AssertionError(message)),
    )
    processor.client = MagicMock()
    processor.translation_queue.put(TranscriptSegment(1, "Hello", datetime.now()))
    processor.translation_queue.put(None)
    processor._translation_loop()

    assert translations == [(1, "こんにちは")]
    assert requests[0]["host"] == "192.168.1.50"
    assert requests[0]["port"] == 9000
    processor.client.responses.create.assert_not_called()


def test_local_translation_uses_ollama(monkeypatch) -> None:
    response = MagicMock()
    response.read.return_value = b'{"message":{"content":"\\u3053\\u3093\\u306b\\u3061\\u306f"}}'
    context = MagicMock()
    context.__enter__.return_value = response
    monkeypatch.setattr(
        "meeting_translator.local_service.urllib.request.urlopen", lambda *_a, **_k: context
    )
    processor = LocalProcessor(
        language="en",
        translation_enabled=True,
        whisper_model="small",
        ollama_model="qwen3:4b-instruct",
        audio_queue=queue.Queue(),
        on_transcript=lambda _segment: None,
        on_translation=lambda _sequence, _text: None,
        on_status=lambda _text: None,
        on_error=lambda _text: None,
    )
    assert processor._translate_with_ollama("Hello") == "こんにちは"


def test_check_ollama_detects_model(monkeypatch) -> None:
    response = MagicMock()
    response.read.return_value = b'{"models":[{"name":"qwen3:4b-instruct"}]}'
    context = MagicMock()
    context.__enter__.return_value = response
    monkeypatch.setattr(
        "meeting_translator.local_service.urllib.request.urlopen",
        lambda *_args, **_kwargs: context,
    )
    assert check_ollama("qwen3:4b-instruct") == (True, "")


def test_translategemma_prompt_preserves_context_and_output_contract() -> None:
    messages = _translation_messages(
        model="translategemma:12b",
        text="It slipped, so move the GA date.",
        language="en",
        history=[
            (
                "Project Falcon will reach general availability in September.",
                "Project Falconは9月に一般提供を開始します。",
            )
        ],
    )

    assert len(messages) == 1
    prompt = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert "Project Falcon" in prompt
    assert "Do not translate them again" in prompt
    assert "without explanations" in prompt
    assert prompt.endswith("\n\nIt slipped, so move the GA date.")


def test_macos_helper_compiles() -> None:
    if platform.system() != "Darwin":
        return
    assert _build_macos_capture_helper().exists()
