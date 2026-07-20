from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers.common import write_json

from .adapters import get_adapter
from .catalogues import REPO_ROOT, load_catalogues
from .common import file_sha256, replace_tokens, slugify, source_batch_id, stable_run_id
from .project import load_project, validate_project


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_project_batch(
    project_path: str | Path,
    *,
    data_source: str = "s3",
    output_root: str | Path | None = None,
    git_sha: str | None = None,
    workflow_run_id: str | None = None,
) -> dict[str, Any]:
    project = load_project(project_path)
    validation = validate_project(project=project, catalogues=load_catalogues())
    if not validation["success"]:
        raise ValueError("Invalid project:\n" + "\n".join(validation["errors"]))

    adapter = get_adapter(project)
    records, source_manifest, join_manifest = adapter.load_records(data_source)
    grain = project["granularity"]
    key_fields = list(grain["key_fields"])
    label_field = str(grain["label_field"])
    order_field = str(grain.get("ordering", {}).get("field") or label_field)
    reverse = str(grain.get("ordering", {}).get("direction", "ascending")) == "descending"
    records = sorted(records, key=lambda row: str(row.get(order_field, "")), reverse=reverse)

    batch_id = source_batch_id(source_manifest)
    resolved_git_sha = git_sha or os.getenv("GITHUB_SHA") or "local"
    run_id = stable_run_id(project["project_id"], int(project["version"]), batch_id, resolved_git_sha)
    root = Path(output_root or f"generated_factory_batches/{project['project_id']}/{run_id}")
    if not root.is_absolute():
        root = REPO_ROOT / root
    generated_root = root / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    template_cache: dict[str, dict[str, Any]] = {}
    item_manifests: dict[str, Any] = {}
    failures: list[dict[str, str]] = []

    for record in records:
        key_values = [str(record.get(field, "")) for field in key_fields]
        item_key = "|".join(key_values)
        item_label = str(record.get(label_field) or item_key)
        item_slug = slugify(item_key)
        item_dir = generated_root / item_slug
        (item_dir / "slides").mkdir(parents=True, exist_ok=True)
        context = adapter.build_context(record, project)
        slides: list[dict[str, Any]] = []
        errors: list[str] = []
        status = "succeeded"
        visual_manifest: dict[str, Any] | None = None

        try:
            asset_result = adapter.render_assets(item_dir, context, project)
            visual_manifest = asset_result.get("visual_manifest")
            for slide in sorted(project["slides"], key=lambda value: value["order"]):
                post_type = next(
                    entry for entry in load_catalogues()["post_types"]["post_types"]
                    if entry["post_type_id"] == slide["post_type_id"]
                )
                layout_path = str(post_type["layout_path"])
                if layout_path not in template_cache:
                    template_cache[layout_path] = json.loads((REPO_ROOT / layout_path).read_text(encoding="utf-8"))
                bindings = {key: replace_tokens(value, context) for key, value in slide.get("text", {}).items()}
                bindings["main_media"] = str(adapter.media_for_slide(slide, asset_result["paths"]))
                output_path = item_dir / "slides" / f"{slide['order']:02d}_{slide['slide_id']}.png"
                result = render_template(template_cache[layout_path], bindings, output_path)
                if result.warnings:
                    raise ValueError(f"Render warnings for {item_label}/{slide['slide_id']}: {result.warnings}")
                with Image.open(output_path) as image:
                    width, height = image.size
                slides.append({
                    "slide_id": slide["slide_id"],
                    "order": slide["order"],
                    "layout_id": slide["post_type_id"],
                    "path": str(output_path.relative_to(root)),
                    "sha256": file_sha256(output_path),
                    "width": width,
                    "height": height,
                    "warnings": [],
                })
        except Exception as exc:
            status = "failed"
            errors.append(str(exc))
            failures.append({"item_slug": item_slug, "item_label": item_label, "error": str(exc)})

        item_manifest = {
            "project_id": project["project_id"],
            "project_version": project["version"],
            "run_id": run_id,
            "grain": grain["grain"],
            "item_key": item_key,
            "item_key_fields": dict(zip(key_fields, key_values)),
            "item_label": item_label,
            "item_slug": item_slug,
            "status": status,
            "review_state": "unreviewed",
            "no_publication": True,
            "source_batch_id": batch_id,
            "slides": slides,
            "errors": errors,
            "generated_at": utc_now(),
        }
        if visual_manifest is not None:
            item_manifest["visual_manifest"] = visual_manifest
        write_json(item_dir / "item_manifest.json", item_manifest)
        item_manifests[item_slug] = item_manifest

    succeeded = sum(1 for item in item_manifests.values() if item["status"] == "succeeded")
    failed = len(item_manifests) - succeeded
    state = "succeeded" if failed == 0 else ("partially_failed" if succeeded else "failed")
    run_manifest = {
        "project_id": project["project_id"],
        "project_version": project["version"],
        "run_id": run_id,
        "adapter_id": adapter.adapter_id,
        "grain": grain["grain"],
        "state": state,
        "draft": True,
        "approved": False,
        "publishing_allowed": False,
        "review_state": "unreviewed",
        "created_at": utc_now(),
        "git_sha": resolved_git_sha,
        "workflow_run_id": workflow_run_id or os.getenv("GITHUB_RUN_ID"),
        "data_source": data_source,
        "source_batch_id": batch_id,
        "source_manifest": source_manifest,
        "join_manifest": join_manifest,
        "expected_item_count": len(records),
        "succeeded_item_count": succeeded,
        "failed_item_count": failed,
        "item_order": list(item_manifests),
        "items": {
            slug: {
                "status": manifest["status"],
                "label": manifest["item_label"],
                "manifest": f"generated/{slug}/item_manifest.json",
            }
            for slug, manifest in item_manifests.items()
        },
        "failures": failures,
        "warnings": validation["warnings"],
    }
    write_json(root / "run_manifest.json", run_manifest)
    write_json(root / "review/review_state.json", {
        "project_id": project["project_id"],
        "run_id": run_id,
        "overall_status": "unreviewed",
        "publishing_allowed": False,
        "items": {
            slug: {
                "status": "unreviewed" if item["status"] == "succeeded" else "generation_failed",
                "slides": {slide["slide_id"]: "unreviewed" for slide in item["slides"]},
            }
            for slug, item in item_manifests.items()
        },
    })
    return {**run_manifest, "output_root": str(root)}
