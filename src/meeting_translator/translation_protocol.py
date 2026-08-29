from __future__ import annotations

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
}
DEFAULT_TRANSLATION_SERVER_HOST = "127.0.0.1"
DEFAULT_TRANSLATION_SERVER_PORT = 8765


def translation_server_base_url(host: str, port: int) -> str:
    cleaned = host.strip().strip("[]")
    if not cleaned or "/" in cleaned or "://" in cleaned:
        raise ValueError("host must be an IP address or host name")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    display_host = f"[{cleaned}]" if ":" in cleaned else cleaned
    return f"http://{display_host}:{port}"


def translation_messages(
    model: str,
    text: str,
    language: str | None,
    history: list[tuple[str, str]],
) -> list[dict[str, str]]:
    source_code = language or "auto"
    source_name = LANGUAGE_NAMES.get(source_code, "automatically detected language")
    context = ""
    if history:
        pairs = "\n".join(
            f"- Source: {original}\n  Japanese: {translated}"
            for original, translated in history[-3:]
        )
        context = (
            "\nFor terminology and pronoun consistency, refer to these immediately preceding "
            "utterances and translations. Do not translate them again:\n"
            f"{pairs}\n"
        )

    if model.lower().startswith("translategemma"):
        prompt = (
            f"You are a professional {source_name} ({source_code}) to Japanese (ja) translator. "
            "Your goal is to accurately convey the meaning and nuances of the original text "
            "while adhering to natural Japanese grammar, vocabulary, and cultural sensitivities. "
            "Preserve names, product names, numbers, units, technical terminology, uncertainty, "
            "and the speaker's degree of politeness. Never summarize or add information."
            f"{context}"
            "Produce only the Japanese translation, without explanations, labels, quotation "
            "marks, or commentary. Please translate the following meeting utterance into Japanese:"
            f"\n\n{text}"
        )
        return [{"role": "user", "content": prompt}]

    system = (
        "You are a professional meeting interpreter translating into natural, accurate Japanese. "
        "Preserve names, numbers, technical terms, nuance, uncertainty, and politeness. "
        "Never summarize, omit, or add information. Return only the Japanese translation."
        f"{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
