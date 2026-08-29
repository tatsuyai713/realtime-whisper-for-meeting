from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any


@dataclass(frozen=True)
class AudioDevice:
    name: str
    native_id: str
    microphone: Any
    is_loopback: bool = False
    capture_kind: str = "soundcard"


@dataclass(frozen=True)
class AudioChunk:
    sequence: int
    samples: Any
    sample_rate: int
    captured_at: datetime
    overlap_seconds: float = 0.0


@dataclass(frozen=True)
class TranscriptSegment:
    sequence: int
    original: str
    captured_at: datetime
    continues_previous: bool = False


def remove_repeated_transcript_prefix(previous: str, current: str) -> str:
    """Remove text repeated because adjacent audio chunks intentionally overlap."""
    previous = previous.strip()
    current = current.strip()
    if not previous or not current:
        return current

    # Languages without spaces (Japanese/Chinese) need character-level matching.
    max_chars = min(60, len(previous), len(current))
    for size in range(max_chars, 3, -1):
        if previous[-size:] == current[:size]:
            return current[size:].lstrip(" 、，。,.")

    previous_words = list(re.finditer(r"\S+", previous))
    current_words = list(re.finditer(r"\S+", current))

    def normalized(match: re.Match[str]) -> str:
        return re.sub(r"[^\w]+", "", match.group(0).casefold())

    max_words = min(16, len(previous_words), len(current_words))
    for size in range(max_words, 1, -1):
        left = [normalized(match) for match in previous_words[-size:]]
        right = [normalized(match) for match in current_words[:size]]
        if left == right and all(left):
            end = current_words[size - 1].end()
            return current[end:].lstrip(" 、，。,.")

    # The same overlapped audio can produce small spelling or punctuation changes.
    best_prefix_words = 0
    best_score = 0.0
    for previous_size in range(2, min(12, len(previous_words)) + 1):
        left = " ".join(normalized(match) for match in previous_words[-previous_size:])
        for current_size in range(
            max(2, previous_size - 2),
            min(len(current_words), previous_size + 2) + 1,
        ):
            right = " ".join(normalized(match) for match in current_words[:current_size])
            if min(len(left), len(right)) < 8:
                continue
            score = SequenceMatcher(None, left, right).ratio()
            if score >= 0.88 and score > best_score:
                best_score = score
                best_prefix_words = current_size
    if best_prefix_words:
        end = current_words[best_prefix_words - 1].end()
        return current[end:].lstrip(" 、，。,.")

    if previous_words and current_words:
        left = normalized(previous_words[-1])
        right = normalized(current_words[0])
        if left == right and len(left) >= 5:
            return current[current_words[0].end() :].lstrip(" 、，。,.")
    return current


def merge_continued_transcript(previous: str, current: str) -> str:
    """Join two chunks from one utterance while reconciling their audio overlap."""
    previous = previous.rstrip()
    current = current.lstrip()
    if not previous:
        return current
    if not current:
        return previous

    contains_cjk = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", previous + current))
    if not contains_cjk:
        previous_words = list(re.finditer(r"\w+(?:[-']\w+)*", previous))
        current_words = list(re.finditer(r"\w+(?:[-']\w+)*", current))

        def word(match: re.Match[str]) -> str:
            return re.sub(r"[^\w]+", "", match.group(0).casefold())

        max_words = min(12, len(previous_words), len(current_words))
        for size in range(max_words, 0, -1):
            left = [word(match) for match in previous_words[-size:]]
            right = [word(match) for match in current_words[:size]]
            if left == right:
                return previous[: previous_words[-size].start()] + current

        if previous_words and current_words:
            left = word(previous_words[-1])
            right = word(current_words[0])
            is_numeric_extension = left.isdigit() and right.startswith(left)
            if (len(left) >= 3 and right.startswith(left)) or is_numeric_extension:
                return previous[: previous_words[-1].start()] + current

        # A forced audio overlap can recognize the same phrase twice with one
        # unfinished/misspelled word at the boundary, e.g.
        # "Toyota Yara" + "Toyota Yaris GR".  Only repair aligned multi-word
        # phrases with an exact anchor word and a substantial shared word stem.
        # This deliberately avoids general fuzzy matching, which could erase an
        # intentional correction or a genuinely different phrase.
        for size in range(max_words, 1, -1):
            left = [word(match) for match in previous_words[-size:]]
            right = [word(match) for match in current_words[:size]]
            exact_matches = sum(a == b for a, b in zip(left, right, strict=True))
            if exact_matches == 0:
                continue
            differing = [(a, b) for a, b in zip(left, right, strict=True) if a != b]
            if not differing:
                continue
            compatible = True
            for a, b in differing:
                common = next(
                    (
                        index
                        for index, pair in enumerate(zip(a, b, strict=False))
                        if pair[0] != pair[1]
                    ),
                    min(len(a), len(b)),
                )
                if (
                    min(len(a), len(b)) < 3
                    or common < 3
                    or SequenceMatcher(None, a, b).ratio() < 0.65
                ):
                    compatible = False
                    break
            if compatible:
                return previous[: previous_words[-size].start()] + current

        separator = "" if previous.endswith((" ", "\n")) else " "
        return previous + separator + current

    minimum_overlap = 1
    best: tuple[int, int] | None = None
    # Allow the new recognition to correct one or two trailing particles/characters.
    for dropped in range(0, min(3, len(previous)) + 1):
        base = previous[:-dropped] if dropped else previous
        max_size = min(40, len(base), len(current))
        for size in range(max_size, minimum_overlap - 1, -1):
            left = base[-size:]
            right = current[:size]
            if (
                left.casefold() == right.casefold()
                or SequenceMatcher(None, left.casefold(), right.casefold()).ratio() >= 0.9
            ):
                candidate = (dropped, size)
                if best is None or size > best[1]:
                    best = candidate
                break
    if best:
        dropped, size = best
        base = previous[:-dropped] if dropped else previous
        return base[:-size] + current

    separator = "" if contains_cjk or previous.endswith((" ", "\n")) else " "
    return previous + separator + current


def split_transcript_sentences(text: str) -> list[str]:
    """Create readable display paragraphs without changing transcript wording."""
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]
