from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers.common import write_json

from .adapters import get_adapter
from .catalogues import REPO_ROOT, load_catalogues
from .common import file_sha256, replace_tokens, slugify
from .project import load_project, validate_project
from .review import create_derived_run, load_json, utc_now


def regenerate_project_items(
    project_path: str | Path,
    source_run_root: str | Path,
    destination_run_root: str | Path,
    *,
    new_run_id: str,
    item_slugs: list[str],
    slide_ids: list[str] | None = None,
    reason: str,
    data_source: str = "s3",
) -> dict[str, Any]:
    project = load_project(project_path)
    catalogues = load_catalogues()
    validation = validate_project(project=project, catalogues=catalogues)
    if not validation["success"]:
        raise ValueError("Invalid project:\n" + "\n".join(validation["errors"]))

    adapter = get_adapter(project)
    destination = Path(destination_run_root)
    manifest = create_derived_run(
        source_run_root,
        destination,
        new_run_id=new_run_id,
        reason=reason,
        item_slugs=item_slugs,
        slide_ids=slide_ids,
    )

    records, source_manifest, join_manifest = adapter.load_records(data_source)
    grain = project["granularity"]
    key_fields = list(grain["key_fields"])
    records_by_slug = {
        slugify("|".join(str(record.get(field, "")) for field in key_fields)): record
        for record in records
    }
    slides_by_id = {slide["slide_id"]: slide for slide in project["slides"]}
    selected_slide_ids = slide_ids or [slide["slide_id"] for slide in sorted(project["slides"], key=lambda row: row["order"])]
    unknown = sorted(set(selected_slide_ids) - set(slides_by_id))
    if unknown:
        raise ValueError(f"Unknown slide IDs: {unknown}")

    template_cache: dict[str, dict[str, Any]] = {}
    regenerated: list[dict[str, Any]] = []

    for item_slug in item_slugs:
        if item_slug not in records_by_slug:
            raise ValueError(f"No source record found for item slug: {item_slug}")
        record = records_by_slug[item_slug]
        context = adapter.build_context(record, project)
        item_dir = destination / "generated" / item_slug
        item_manifest_path = item_dir / "item_manifest.json"
        item_manifest = load_json(item_manifest_path)
        current_slides = {slide["slide_id"]: slide for slide in item_manifest.get("slides", [])}
        asset_result = adapter.render_assets(item_dir, context, project)

        for slide_id in selected_slide_ids:
            slide = slides_by_id[slide_id]
            post_type = catalogues.post_types[slide["post_type_id"]]
            layout_path = str(post_type["layout_path"])
            if layout_path not in template_cache:
                template_cache[layout_path] = json.loads((REPO_ROOT / layout_path).read_text(encoding="utf-8"))
            bindings = {key: replace_tokens(value, context) for key, value in slide.get("text", {}).items()}
            bindings["main_media"] = str(adapter.media_for_slide(slide, asset_result["paths"]))
            output_path = item_dir / "slides" / f"{slide['order']:02d}_{slide_id}.png"
            result = render_template(template_cache[layout_path], bindings, output_path)
            if result.warnings:
                raise ValueError(f"Render warnings for {item_slug}/{slide_id}: {result.warnings}")
            with Image.open(output_path) as image:
                width, height = image.size
            current_slides[slide_id] = {
                "slide_id": slide_id,
                "order": slide["order"],
                "layout_id": slide["post_type_id"],
                "path": str(output_path.relative_to(destination)),
                "sha256": file_sha256(output_path),
                "width": width,
                "height": height,
                "warnings": [],
                "regenerated_at": utc_now(),
                "regeneration_reason": reason,
            }
            regenerated.append({"item_slug": item_slug, "slide_id": slide_id, "path": str(output_path)})

        item_manifest["run_id"] = new_run_id
        item_manifest["review_state"] = "unreviewed"
        item_manifest["approved"] = False
        item_manifest["no_publication"] = True
        item_manifest["slides"] = sorted(current_slides.values(), key=lambda entry: entry["order"])
        if asset_result.get("visual_manifest") is not None:
            item_manifest["visual_manifest"] = asset_result["visual_manifest"]
        item_manifest.setdefault("regeneration_history", []).append({
            "timestamp": utc_now(),
            "parent_run_id": manifest["parent_run_id"],
            "reason": reason,
            "slide_ids": sorted(selected_slide_ids),
        })
        write_json(item_manifest_path, item_manifest)

    manifest_path = destination / "run_manifest.json"
    manifest = load_json(manifest_path)
    manifest["adapter_id"] = adapter.adapter_id
    manifest["grain"] = grain["grain"]
    manifest["source_manifest"] = source_manifest
    manifest["join_manifest"] = join_manifest
    manifest["regenerated_outputs"] = regenerated
    manifest["state"] = "succeeded"
    manifest["review_state"] = "unreviewed"
    manifest["approved"] = False
    manifest["publishing_allowed"] = False
    write_json(manifest_path, manifest)
    return {**manifest, "output_root": str(destination)}
