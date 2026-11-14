"""Keep a short history of brief metrics so each new brief can be compared with the previous one."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

HISTORY_FILE = os.path.join(tempfile.gettempdir(), "airbrief_history.json")
MAX_HISTORY = 10


def load_history(path: str = HISTORY_FILE) -> list[dict[str, Any]]:
    """Return saved briefs (oldest first); an empty list if none or unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    return [d for d in data if isinstance(d, dict) and "metrics" in d] if isinstance(data, list) else []


def save_history(history: list[dict[str, Any]], path: str = HISTORY_FILE) -> None:
    """Persist the latest briefs. Failure to save must never break the app."""
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(history[-MAX_HISTORY:], handle)
    except OSError:
        pass


def parse_saved_brief(raw: bytes) -> dict[str, Any]:
    """Read a brief file downloaded from this app.

    Raises:
        ValueError: if the file is not a brief exported by this app.
    """
    try:
        data = json.loads(raw.decode("utf-8"))
        return {"created": str(data["created"]), "metrics": {"stations": dict(data["metrics"]["stations"])}}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("This file is not a brief exported from this app.") from exc
