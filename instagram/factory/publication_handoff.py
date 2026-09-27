from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from instagram.factory.contract import RenderResult

HANDOFF_FILENAME = "publication_handoff.json"
HANDOFF_SCHEMA_VERSION = 1


def _relative(path: str | None, root: Path) -> str | None:
    if not path:
        return None
    value = Path(path).resolve()
    try:
        return value.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"publication handoff path is outside artifact root: {path}") from exc


def build_publication_handoff(result: RenderResult, *, artifact_root: str | Path) -> dict[str, Any]:
    root = Path(artifact_root)
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "project_id": result.project_id,
        "period_key": result.period_key,
        "source_batch_id": result.source_batch_id,
        "qa_all_passed": result.qa.all_passed,
        "review_state": result.review_state,
        "factory_publication_enabled": result.publication_enabled,
        "slides": [
            {
                "id": slide.id,
                "path": _relative(slide.path, root),
                "width": slide.width,
                "height": slide.height,
            }
            for slide in result.slides
        ],
        "caption_path": _relative(result.caption_path, root),
        "source_manifest_path": _relative(result.manifest_path, root),
        "operator_contract": {
            "requires_explicit_publish_approval": True,
            "factory_does_not_publish": True,
            "caption_override_required_when_caption_path_is_null": result.caption_path is None,
        },
    }


def write_publication_handoff(result: RenderResult, *, artifact_root: str | Path) -> Path:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / HANDOFF_FILENAME
    value = build_publication_handoff(result, artifact_root=root)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
