from __future__ import annotations

from pathlib import Path
from typing import Any

from instagram.visuals.renderers.common import write_json

from .review import load_json, utc_now


def mark_ready_for_posting(
    run_root: str | Path,
    *,
    reviewer: str,
    note: str | None = None,
) -> dict[str, Any]:
    root = Path(run_root)
    manifest_path = root / "run_manifest.json"
    state_path = root / "review/review_state.json"
    manifest = load_json(manifest_path)
    state = load_json(state_path)

    blocking: list[str] = []
    for slug, item in sorted(state.get("items", {}).items()):
        if item.get("status") != "approved":
            blocking.append(f"{slug}: item status is {item.get('status', 'unreviewed')}")
        for slide_id, status in sorted(item.get("slides", {}).items()):
            if status != "approved":
                blocking.append(f"{slug}/{slide_id}: slide status is {status}")
    if blocking:
        raise ValueError("Run is not fully approved:\n" + "\n".join(blocking))

    timestamp = utc_now()
    manifest["review_state"] = "approved"
    manifest["approved"] = True
    manifest["ready_for_posting"] = True
    manifest["ready_for_posting_at"] = timestamp
    manifest["ready_for_posting_by"] = reviewer
    manifest["ready_for_posting_note"] = note
    manifest["publishing_allowed"] = False
    write_json(manifest_path, manifest)

    state["overall_status"] = "approved"
    state["ready_for_posting"] = True
    state["publishing_allowed"] = False
    state["updated_at"] = timestamp
    state.setdefault("history", []).append(
        {
            "timestamp": timestamp,
            "event": "ready_for_posting",
            "reviewer": reviewer,
            "note": note,
        }
    )
    write_json(state_path, state)
    return {
        "run_id": manifest["run_id"],
        "approved": True,
        "ready_for_posting": True,
        "publishing_allowed": False,
        "reviewer": reviewer,
        "timestamp": timestamp,
    }
