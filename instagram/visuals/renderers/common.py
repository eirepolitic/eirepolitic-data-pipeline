from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_palette(template: dict[str, Any]) -> dict[str, str]:
    palette = template.get("palette", {}) or {}
    return {
        "background": str(palette.get("background", "#0f2f24")),
        "panel": str(palette.get("panel", "#173d30")),
        "panel_alt": str(palette.get("panel_alt", "#214a3b")),
        "text": str(palette.get("text", "#f4ead7")),
        "muted": str(palette.get("muted", "#cbbf9f")),
        "accent": str(palette.get("accent", "#d8b45f")),
        "grid": str(palette.get("grid", "#cbbf9f")),
        "warning": str(palette.get("warning", "#b55b5b")),
    }


def rows_from_sample(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_cfg = sample.get("input", {}) or {}
    mode = str(input_cfg.get("mode", "inline"))
    if mode == "inline":
        return list(input_cfg.get("rows", []) or []), {
            "input_mode": "inline",
            "source": "inline",
        }
    if mode == "local_csv":
        csv_path = resolve_repo_path(str(input_cfg["path"]))
        rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
        return rows, {
            "input_mode": "local_csv",
            "source": str(input_cfg["path"]),
            "resolved_source": str(csv_path),
        }
    raise ValueError(f"Unsupported visual input mode: {mode}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
