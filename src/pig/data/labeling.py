from __future__ import annotations

import base64
import re

_LEAK = re.compile(
    r"(system ?prompt|your (instructions|rules|prompt)|initial (instructions|prompt)|"
    r"(reveal|print|show|repeat|output|tell me) .{0,40}(above|instructions|prompt|password|secret)|"
    r"системн\w* (промпт|инструкц|подсказк)|твои (инструкции|правила)|"
    r"(выведи|покажи|повтори|раскрой|напиши) .{0,40}(инструкц|промпт|выше|пароль|секрет))",
    re.IGNORECASE,
)
_ROLE = re.compile(
    r"(\bDAN\b|do anything now|developer mode|jailbreak|pretend (you are|to be)|act as|roleplay|"
    r"you are now|from now on you|unfiltered|no restrictions|evil|"
    r"представь,? что ты|ты теперь|притворись|режим разработчика|без (ограничений|цензуры|фильтров)|"
    r"играй роль|веди себя как)",
    re.IGNORECASE,
)
_OVERRIDE = re.compile(
    r"(ignore|disregard|forget|override|bypass) .{0,30}(previous|above|prior|all|earlier|instructions|rules)|"
    r"(игнорир|забудь|отмен|не обращай внимания|пренебреги|перестань следовать).{0,40}"
    r"(предыдущ|указани|инструкц|правил|выше|все)",
    re.IGNORECASE,
)
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_SPACED = re.compile(r"(?:\b\w\s){6,}")
_LEET = re.compile(r"\b\w*[0-9@$]\w*[a-z]\w*[0-9@$]\w*\b", re.IGNORECASE)


def _looks_obfuscated(text: str) -> bool:
    for m in _B64.findall(text):
        try:
            base64.b64decode(m + "=" * (-len(m) % 4), validate=True).decode("utf-8")
            return True
        except Exception:
            continue
    if _SPACED.search(text):
        return True
    return len(_LEET.findall(text)) >= 3


def infer_attack_type(text: str) -> str:
    if _looks_obfuscated(text):
        return "obfuscation"
    if _LEAK.search(text):
        return "leak"
    if _ROLE.search(text):
        return "roleplay"
    return "override"
