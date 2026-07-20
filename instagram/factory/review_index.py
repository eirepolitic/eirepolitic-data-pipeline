from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .review import load_json, utc_now


def _status_counts(items: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items.values():
        status = str(item.get("status", "unreviewed"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_review_index(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    manifest = load_json(root / "run_manifest.json")
    state = load_json(root / "review/review_state.json")
    item_rows: list[str] = []

    for slug in manifest.get("item_order", sorted(state.get("items", {}))):
        item_state = state.get("items", {}).get(slug, {})
        item_entry = manifest.get("items", {}).get(slug, {})
        label = html.escape(str(item_entry.get("label", slug)))
        status = html.escape(str(item_state.get("status", "unreviewed")))
        slide_cells: list[str] = []
        item_manifest_path = root / str(item_entry.get("manifest", f"generated/{slug}/item_manifest.json"))
        item_manifest = load_json(item_manifest_path)
        for slide in item_manifest.get("slides", []):
            slide_id = str(slide["slide_id"])
            slide_status = html.escape(str(item_state.get("slides", {}).get(slide_id, "unreviewed")))
            rel_path = html.escape(str(slide["path"]))
            slide_cells.append(
                f'<div class="slide"><a href="../{rel_path}">{html.escape(slide_id)}</a>'
                f'<span class="status">{slide_status}</span></div>'
            )
        item_rows.append(
            f'<section class="item"><h2>{label}</h2><p class="item-status">{status}</p>'
            f'<div class="slides">{"".join(slide_cells)}</div></section>'
        )

    counts = _status_counts(state.get("items", {}))
    count_text = " · ".join(f"{html.escape(key)}: {value}" for key, value in sorted(counts.items()))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review index — {html.escape(str(manifest['run_id']))}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 24px; background: #f6f6f3; color: #1d2b24; }}
header {{ margin-bottom: 24px; }}
.summary {{ background: white; padding: 16px; border-radius: 12px; }}
.item {{ background: white; padding: 16px; margin: 16px 0; border-radius: 12px; }}
.item h2 {{ margin: 0; }}
.item-status, .status {{ font-weight: 600; }}
.slides {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.slide {{ border: 1px solid #ccc; border-radius: 8px; padding: 10px; min-width: 180px; display: flex; justify-content: space-between; gap: 12px; }}
a {{ color: #185f45; }}
</style>
</head>
<body>
<header>
<h1>Instagram Content Factory review</h1>
<div class="summary">
<p><strong>Run:</strong> {html.escape(str(manifest['run_id']))}</p>
<p><strong>Overall review:</strong> {html.escape(str(state.get('overall_status', 'unreviewed')))}</p>
<p><strong>Ready for posting:</strong> {str(bool(manifest.get('ready_for_posting', False))).lower()}</p>
<p><strong>Publishing allowed:</strong> false</p>
<p>{count_text}</p>
</div>
</header>
<main>{''.join(item_rows)}</main>
</body>
</html>
"""
    output = root / "review/review_index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    result = {
        "run_id": manifest["run_id"],
        "output": str(output),
        "item_count": len(item_rows),
        "status_counts": counts,
        "overall_status": state.get("overall_status", "unreviewed"),
        "ready_for_posting": bool(manifest.get("ready_for_posting", False)),
        "publishing_allowed": False,
        "generated_at": utc_now(),
    }
    (root / "review/review_index_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
