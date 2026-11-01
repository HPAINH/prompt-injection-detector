from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")
_CYR = re.compile(r"[а-яё]", re.IGNORECASE)
_LAT = re.compile(r"[a-z]", re.IGNORECASE)
_SENT = re.compile(r"(?<=[.!?…])\s+|\n+")


def normalize(text: str, max_len: int = 2000) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _WS.sub(" ", text).strip()
    return text[:max_len]


def text_id(text: str) -> str:
    key = normalize(text).lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def detect_lang(text: str) -> str:
    cyr = len(_CYR.findall(text))
    lat = len(_LAT.findall(text))
    return "ru" if cyr >= lat else "en"


def fragments(text: str, window: int = 12, stride: int = 6) -> list[str]:
    parts = [p.strip() for p in _SENT.split(text) if p and len(p.strip()) > 2]
    if len(parts) >= 2:
        return parts
    words = text.split()
    if len(words) <= window:
        return [text]
    out = []
    for start in range(0, len(words) - window + stride, stride):
        chunk = words[start : start + window]
        if chunk:
            out.append(" ".join(chunk))
        if start + window >= len(words):
            break
    return out
