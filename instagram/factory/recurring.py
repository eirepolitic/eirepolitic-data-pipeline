from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constituency_batch import _batch_id
from .constituency_pilot import build_constituency_records, load_source_rows


def evaluate_readiness(
    *,
    data_source: str = "s3",
    latest_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    members, speeches, source_manifest = load_source_rows(data_source)
    records, join_manifest = build_constituency_records(members, speeches)
    batch_id = _batch_id(source_manifest)
    previous_batch_id = (latest_manifest or {}).get("source_batch_id")
    reasons: list[str] = []

    if not members:
        reasons.append("member source is empty")
    if not speeches:
        reasons.append("speech source is empty")
    if join_manifest.get("matched_speeches", 0) <= 0:
        reasons.append("no speeches matched current members")
    if join_manifest.get("constituency_count", 0) <= 0:
        reasons.append("no constituency records were produced")
    if batch_id == "local-fixture" and data_source == "s3":
        reasons.append("production batch ID was not resolved")

    duplicate = bool(previous_batch_id and previous_batch_id == batch_id)
    if duplicate:
        reasons.append(f"source batch {batch_id} has already been generated")

    return {
        "ready": not reasons,
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
