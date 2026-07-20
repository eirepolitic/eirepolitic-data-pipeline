from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import get_adapter
from .common import source_batch_id
from .project import load_project


def evaluate_readiness(
    project_path: str | Path,
    *,
    data_source: str = "s3",
    latest_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project = load_project(project_path)
    adapter = get_adapter(project)
    records, source_manifest, join_manifest = adapter.load_records(data_source)
    batch_id = source_batch_id(source_manifest)
    previous_batch_id = (latest_manifest or {}).get("source_batch_id")
    reasons: list[str] = []

    if not records:
        reasons.append(f"no {project['granularity']['grain']} items were produced")
    if join_manifest.get("matched_speeches") is not None and join_manifest.get("matched_speeches", 0) <= 0:
        reasons.append("adapter join produced no matched source rows")
    if batch_id == "local-fixture" and data_source == "s3":
        reasons.append("production batch ID was not resolved")

    duplicate = bool(previous_batch_id and previous_batch_id == batch_id)
    if duplicate:
        reasons.append(f"source batch {batch_id} has already been generated")

    return {
        "ready": not reasons,
        "project_id": project["project_id"],
        "adapter_id": adapter.adapter_id,
        "grain": project["granularity"]["grain"],
        "data_source": data_source,
        "source_batch_id": batch_id,
        "previous_source_batch_id": previous_batch_id,
        "duplicate_source_batch": duplicate,
        "source_manifest": source_manifest,
        "join_manifest": join_manifest,
        "expected_item_count": len(records),
        "reasons": reasons,
    }


def load_latest_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))
