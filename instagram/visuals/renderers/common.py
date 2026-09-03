from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_palette(template: dict[str, Any]) -> dict[str, str]:
    palette = template.get("palette", {}) or {}
    return {
        "background": str(palette.get("background", "#0f2f24")),
        "panel": str(palette.get("panel", "#0f2f24")),
        "panel_alt": str(palette.get("panel_alt", "#214a3b")),
        "text": str(palette.get("text", "#f4ead7")),
        "muted": str(palette.get("muted", "#c8bda8")),
        "accent": str(palette.get("accent", "#d8b45f")),
        "accent_2": str(palette.get("accent_2", "#9ec5a2")),
        "grid": str(palette.get("grid", "#f4ead7")),
        "warning": str(palette.get("warning", "#b55b5b")),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
