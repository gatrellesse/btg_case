"""On-disk response cache.

Two jobs. It keeps development from re-billing the same eight documents, and —
committed to the repository — it lets an evaluator reproduce the whole batch
with ``--offline`` and no API key at all. Reproducibility of a run that
involved a model is otherwise not something a reader can check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResponseCache:
    def __init__(self, directory: str | Path, enabled: bool = True) -> None:
        self.dir = Path(directory)
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, prompt: str, system: str | None, schema_name: str) -> str:
        digest = hashlib.sha256()
        for part in (model, schema_name, system or "", prompt):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:32]

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self.dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, key: str, payload: dict) -> None:
        if not self.enabled:
            return
        try:
            (self.dir / f"{key}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            # A cache that cannot write must not take the run down with it.
            pass
