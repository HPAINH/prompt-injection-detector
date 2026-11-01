from __future__ import annotations

import re

import numpy as np

from pig.models.base import Detector

PATTERNS: list[tuple[str, float]] = [
    (r"\b(ignore|disregard|forget|override|bypass)\b.{0,30}\b(previous|above|prior|all|instructions|rules)\b", 0.9),
    (r"(игнорир\w*|забудь|не учитывай|отмени|проигнорируй|не обращай внимания).{0,40}(предыдущ|указани|инструкц|правил|выше)", 0.9),
    (r"\b(system prompt|initial instructions|your instructions)\b", 0.6),
    (r"(системн\w* (промпт|инструкц)|твои (инструкции|правила)|скрыт\w* правил)", 0.6),
    (r"\b(DAN|do anything now|developer mode|jailbreak)\b", 0.8),
    (r"(режим разработчика|без (цензуры|ограничений|фильтров)|ты теперь|представь,? что ты)", 0.5),
    (r"\b(pretend (you are|to be)|you are now|act as|stay in character)\b", 0.5),
    (r"(выведи|покажи|повтори|раскрой).{0,30}(дословно|целиком|выше|промпт)", 0.6),
    (r"\b(print|reveal|repeat|show)\b.{0,30}\b(verbatim|above|prompt|password)\b", 0.6),
    (r"(base64|декодируй|decode)", 0.4),
    (r"(###\s*(system|система)|\[system\]|<\|im_start\|>)", 0.7),
    (r"(new (task|instructions|rules)|новая задача|новые правила|настройки обновлены)", 0.5),
]


class RuleDetector(Detector):
    name = "M0_rules"

    def __init__(self) -> None:
        super().__init__()
        self.patterns = [(re.compile(p, re.IGNORECASE | re.DOTALL), w) for p, w in PATTERNS]

    def _score(self, text: str) -> float:
        keep = 1.0
        for rx, w in self.patterns:
            if rx.search(text):
                keep *= 1 - w
        return 1 - keep

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.array([self._score(t) for t in texts], dtype=np.float32)
