from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from pig.text import text_id


class IncidentLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, text: str, result: dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "text_hash": text_id(text),
            "text_len": len(text),
            "verdict": result["verdict"],
            "score": round(float(result["score"]), 4),
            "attack_type": result["attack_type"],
            "model": result["model"],
        }
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
