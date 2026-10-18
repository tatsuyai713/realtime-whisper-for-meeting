from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalPerformanceProfile:
    model: str
    engine: str
    compute_profile: str
    max_segment_seconds: float
    overlap_seconds: float


LOCAL_PERFORMANCE_PROFILES = {
    "mac-fast": LocalPerformanceProfile(
        model="small-fast",
        engine="whisperkit",
        compute_profile="ane",
        max_segment_seconds=5.0,
        overlap_seconds=0.4,
    ),
    "mac-balanced": LocalPerformanceProfile(
        model="turbo",
        engine="whisperkit",
        compute_profile="ane",
        max_segment_seconds=5.0,
        overlap_seconds=0.5,
    ),
    "mac-accuracy": LocalPerformanceProfile(
        model="large-v3",
        engine="whisperkit",
        compute_profile="ane",
        max_segment_seconds=4.5,
        overlap_seconds=0.6,
    ),
}

LEGACY_PERFORMANCE_PROFILES = {
    "m4-fast": "mac-fast",
    "m4-balanced": "mac-balanced",
    "m4-accuracy": "mac-accuracy",
}


def resolve_local_performance(
    profile: str,
    *,
    custom_model: str,
    custom_engine: str,
    custom_compute_profile: str,
) -> LocalPerformanceProfile:
    selected = LOCAL_PERFORMANCE_PROFILES.get(LEGACY_PERFORMANCE_PROFILES.get(profile, profile))
    if selected is not None:
        return selected
    return LocalPerformanceProfile(
        model=custom_model,
        engine=custom_engine,
        compute_profile=custom_compute_profile,
        max_segment_seconds=3.5,
        overlap_seconds=0.8,
    )


def format_realtime_speed(audio_seconds: float, elapsed_seconds: float) -> str:
    if audio_seconds <= 0 or elapsed_seconds <= 0:
        return ""
    realtime_factor = elapsed_seconds / audio_seconds
    speed = audio_seconds / elapsed_seconds
    return f"RTF {realtime_factor:.2f}（実時間の{speed:.1f}倍速）"
