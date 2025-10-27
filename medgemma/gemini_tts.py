import base64
import os
from typing import Dict, Optional

import google.generativeai as genai

from cache import PersistentCache, cache as default_cache

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GENERATE_SPEECH = os.environ.get("GENERATE_SPEECH", "false").lower() == "true"
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.0-flash-exp")
GEMINI_TTS_LANGUAGE = os.environ.get("GEMINI_TTS_LANGUAGE", "bn-BD")
DEFAULT_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Kore")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def _cache_context(voice: str) -> str:
    return f"tts_bangla::{voice}"


def _extract_audio_from_cache(entry: object) -> tuple[Optional[str], Optional[str]]:
    if isinstance(entry, dict):
        return entry.get("audio"), entry.get("mime")
    if isinstance(entry, str):
        return entry, None
    return None, None


def _extract_audio_from_response(response: object) -> tuple[Optional[bytes], Optional[str]]:
    audio_bytes: Optional[bytes] = getattr(response, "audio", None)
    mime_type: Optional[str] = getattr(response, "mime_type", None)

    if audio_bytes:
        return audio_bytes, mime_type

    try:
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline:
                data = getattr(inline, "data", None)
                mime_type = getattr(inline, "mime_type", None)
                if isinstance(data, str):
                    data = base64.b64decode(data)
                return data, mime_type
    except Exception:
        return None, None

    return None, None


def _ensure_audio_payload(
    text: str,
    voice: str,
    cache: Optional[PersistentCache] = None,
) -> tuple[Optional[str], Optional[str]]:
    cache = cache or default_cache
    clean_text = (text or "").strip()
    if not clean_text:
        return None, None

    cached_entry = cache.get(clean_text, context=_cache_context(voice), cache_type="audio")
    audio_base64, mime_type = _extract_audio_from_cache(cached_entry) if cached_entry else (None, None)
    if audio_base64:
        return audio_base64, mime_type

    if not GENERATE_SPEECH or not GEMINI_API_KEY:
        return None, None

    try:
        model = genai.GenerativeModel(GEMINI_TTS_MODEL)
        response = model.generate_content(
            clean_text,
            generation_config=genai.GenerationConfig(
                response_modalities=["AUDIO"],
                speech_config={
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": voice},
                    },
                    "language_code": GEMINI_TTS_LANGUAGE,
                },
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"TTS generation error: {exc}")
        return None, None

    audio_bytes, mime_type = _extract_audio_from_response(response)
    if not audio_bytes:
        return None, None

    if isinstance(audio_bytes, str):
        audio_bytes = base64.b64decode(audio_bytes)

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    payload: Dict[str, str] = {"audio": audio_base64}
    if mime_type:
        payload["mime"] = mime_type

    cache.set(clean_text, payload, context=_cache_context(voice), cache_type="audio")
    return audio_base64, mime_type


def generate_speech(
    text: str,
    cache: Optional[PersistentCache] = None,
    voice: str = DEFAULT_VOICE,
) -> Optional[str]:
    """
    Return Bangla speech as base64-encoded audio payload.
    """

    audio_base64, _ = _ensure_audio_payload(text, voice, cache)
    return audio_base64


def generate_speech_batch(
    texts: list[str],
    cache: Optional[PersistentCache] = None,
    voice: str = DEFAULT_VOICE,
) -> dict[str, Optional[str]]:
    return {text: generate_speech(text, cache=cache, voice=voice) for text in texts}


def get_available_voices() -> list[str]:
    return ["Kore", "Puck", "Charon"]


def clear_audio_cache(cache: Optional[PersistentCache] = None) -> None:
    (cache or default_cache).clear(cache_type="audio")


def synthesize_gemini_tts(
    text: str,
    gemini_voice_name: str = DEFAULT_VOICE,
    cache: Optional[PersistentCache] = None,
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Compatibility wrapper returning raw audio bytes and mime-type.
    """

    audio_base64, mime_type = _ensure_audio_payload(text, gemini_voice_name, cache)
    if not audio_base64:
        return None, None

    audio_bytes = base64.b64decode(audio_base64.encode("utf-8"))
    return audio_bytes, mime_type or "audio/mpeg"

