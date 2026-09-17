from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from instagram.visuals.renderers.common import write_json

VALID_STATES = {"unreviewed", "approved", "changes_requested", "rejected"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mark_review(
    run_root: str | Path,
    *,
    item_slug: str,
    status: str,
    slide_id: str | None = None,
    note: str | None = None,
    reviewer: str = "human",
) -> dict[str, Any]:
    if status not in VALID_STATES:
        raise ValueError(f"Invalid review status: {status}. Allowed: {sorted(VALID_STATES)}")
    root = Path(run_root)
    state_path = root / "review/review_state.json"
    state = load_json(state_path)
    if item_slug not in state.get("items", {}):
        raise ValueError(f"Unknown item slug: {item_slug}")
    item = state["items"][item_slug]
    if slide_id:
        if slide_id not in item.get("slides", {}):
            raise ValueError(f"Unknown slide ID for {item_slug}: {slide_id}")
        item["slides"][slide_id] = status
        slide_states = set(item["slides"].values())
        if slide_states == {"approved"}:
            item["status"] = "approved"
        elif "changes_requested" in slide_states:
            item["status"] = "changes_requested"
        elif "rejected" in slide_states:
            item["status"] = "rejected"
        else:
            item["status"] = "unreviewed"
    else:
        item["status"] = status
        for key in item.get("slides", {}):
            item["slides"][key] = status

    event = {
        "timestamp": utc_now(),
        "reviewer": reviewer,
        "item_slug": item_slug,
        "slide_id": slide_id,
        "status": status,
        "note": note,
    }
    state.setdefault("history", []).append(event)
    item_states = [value["status"] for value in state["items"].values()]
    if item_states and all(value == "approved" for value in item_states):
        state["overall_status"] = "approved"
    elif any(value == "changes_requested" for value in item_states):
        state["overall_status"] = "changes_requested"
    elif any(value == "rejected" for value in item_states):
        state["overall_status"] = "rejected"
    else:
        state["overall_status"] = "in_review"
    state["publishing_allowed"] = False
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    return state


def create_derived_run(
    source_run_root: str | Path,
    destination_run_root: str | Path,
    *,
    new_run_id: str,
    reason: str,
    item_slugs: list[str],
    slide_ids: list[str] | None = None,
) -> dict[str, Any]:
    source = Path(source_run_root)
    destination = Path(destination_run_root)
    if destination.exists():
        raise ValueError(f"Destination run already exists: {destination}")
    shutil.copytree(source, destination)
    manifest_path = destination / "run_manifest.json"
    manifest = load_json(manifest_path)
    parent_run_id = manifest["run_id"]
    manifest["run_id"] = new_run_id
    manifest["parent_run_id"] = parent_run_id
    manifest["derived_at"] = utc_now()
    manifest["regeneration"] = {
        "reason": reason,
        "item_slugs": sorted(item_slugs),
        "slide_ids": sorted(slide_ids or []),
    }
    manifest["review_state"] = "unreviewed"
    manifest["approved"] = False
    manifest["publishing_allowed"] = False
    write_json(manifest_path, manifest)

    state_path = destination / "review/review_state.json"
    state = load_json(state_path)
    state["run_id"] = new_run_id
    state["overall_status"] = "unreviewed"
    state["publishing_allowed"] = False
    state.setdefault("history", []).append(
        {
            "timestamp": utc_now(),
            "event": "derived_run_created",
            "parent_run_id": parent_run_id,
            "reason": reason,
            "item_slugs": sorted(item_slugs),
            "slide_ids": sorted(slide_ids or []),
        }
    )
    for slug in item_slugs:
        if slug not in state.get("items", {}):
            raise ValueError(f"Unknown item slug: {slug}")
        state["items"][slug]["status"] = "unreviewed"
        target_slides = slide_ids or list(state["items"][slug].get("slides", {}))
        for slide_id in target_slides:
            if slide_id not in state["items"][slug].get("slides", {}):
                raise ValueError(f"Unknown slide ID for {slug}: {slide_id}")
            state["items"][slug]["slides"][slide_id] = "unreviewed"
    write_json(state_path, state)
    return manifest
