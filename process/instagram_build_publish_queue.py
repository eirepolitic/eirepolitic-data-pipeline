from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_COPY_CSV = "generated_posts/member_profile_batch_v1_fixture/copy/captions.csv"
DEFAULT_QUEUE_DIR = "generated_posts/member_profile_batch_v1_fixture/queue"
APPROVED_STATUSES = {"approved", "ready", "ready_to_publish", "publish_ready"}


def clean(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value or "").strip()
    return text if text else default


def is_yes(value: Any) -> bool:
    return clean(value).lower() in {"yes", "y", "true", "1"}


def is_approved(value: Any) -> bool:
    return clean(value).lower() in APPROVED_STATUSES


def build_publish_queue(copy_csv: str | Path, queue_dir: str | Path) -> dict[str, Any]:
    copy_csv = Path(copy_csv)
    queue_dir = Path(queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(copy_csv)
    queue_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        publish_ready = is_yes(row.get("publish_ready"))
        approved = is_approved(row.get("review_status"))
        safety_notes = clean(row.get("safety_notes"))
        has_blocking_notes = bool(safety_notes)
        item = {
            "slug": clean(row.get("slug")),
            "full_name": clean(row.get("full_name")),
            "output_file_rel": clean(row.get("output_file_rel")),
            "caption_file": clean(row.get("caption_file")),
            "alt_text_file": clean(row.get("alt_text_file")),
            "caption": clean(row.get("caption")),
            "alt_text": clean(row.get("alt_text")),
            "hashtags": clean(row.get("hashtags")),
            "publish_ready": clean(row.get("publish_ready"), "no"),
            "review_status": clean(row.get("review_status"), "needs_review"),
            "safety_notes": safety_notes,
        }
        if publish_ready and approved and not has_blocking_notes:
            queue_rows.append(item)
        else:
            reasons = []
            if not publish_ready:
                reasons.append("publish_ready is not yes")
            if not approved:
                reasons.append("review_status is not approved/ready")
            if has_blocking_notes:
                reasons.append("safety_notes is not empty")
            blocked_rows.append({**item, "blocked_reasons": "; ".join(reasons)})

    queue_csv = queue_dir / "publish_queue.csv"
    blocked_csv = queue_dir / "blocked_items.csv"
    manifest_path = queue_dir / "publish_queue_manifest.json"

    pd.DataFrame(queue_rows).to_csv(queue_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(blocked_rows).to_csv(blocked_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "success": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "copy_csv": str(copy_csv),
        "queue_dir": str(queue_dir),
        "publish_queue_csv": str(queue_csv),
        "blocked_items_csv": str(blocked_csv),
        "queued_count": len(queue_rows),
        "blocked_count": len(blocked_rows),
        "gate_rules": {
            "publish_ready": "must equal yes/true/1",
            "review_status": sorted(APPROVED_STATUSES),
            "safety_notes": "must be empty",
        },
        "notes": [
            "This creates a queue file only; it does not publish posts.",
            "Fixture runs should normally produce an empty publish queue because generated review tables default to needs_review and publish_ready=no.",
            "Publishing must remain a separate explicit step after manual review.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review-gated Instagram publish queue from copy-pack outputs.")
    parser.add_argument("--copy-csv", default=DEFAULT_COPY_CSV)
    parser.add_argument("--queue-dir", default=DEFAULT_QUEUE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_publish_queue(args.copy_csv, args.queue_dir)
    print(json.dumps({
        "success": True,
        "queued_count": result["queued_count"],
        "blocked_count": result["blocked_count"],
        "queue_dir": result["queue_dir"],
    }, indent=2))


if __name__ == "__main__":
    main()
