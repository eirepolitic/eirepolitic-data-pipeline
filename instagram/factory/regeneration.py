from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers.common import write_json

from .catalogues import REPO_ROOT, load_catalogues
from .constituency_batch import _bindings, _item_context, file_sha256, slugify
from .constituency_pilot import build_constituency_records, load_source_rows, render_visual, write_cover_asset
from .project import load_project, validate_project
from .review import create_derived_run, load_json, utc_now


def regenerate_selected(
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
    validation = validate_project(project=project, catalogues=load_catalogues())
    if not validation["success"]:
        raise ValueError("Invalid project:\n" + "\n".join(validation["errors"]))

    source_root = Path(source_run_root)
    destination_root = Path(destination_run_root)
    manifest = create_derived_run(
        source_root,
        destination_root,
        new_run_id=new_run_id,
        reason=reason,
        item_slugs=item_slugs,
        slide_ids=slide_ids,
    )

    members, speeches, source_manifest = load_source_rows(data_source)
    records, join_manifest = build_constituency_records(members, speeches)
    records_by_slug = {slugify(record["constituency"]): record for record in records}
    slides_by_id = {slide["slide_id"]: slide for slide in project["slides"]}
    selected_slide_ids = slide_ids or sorted(slides_by_id)
    unknown_slides = sorted(set(selected_slide_ids) - set(slides_by_id))
    if unknown_slides:
        raise ValueError(f"Unknown slide IDs: {unknown_slides}")

    template = json.loads(
        (REPO_ROOT / "instagram/templates/layouts/title_text_media_v1.json").read_text(encoding="utf-8")
    )
    regenerated: list[dict[str, Any]] = []

    for item_slug in item_slugs:
        if item_slug not in records_by_slug:
            raise ValueError(f"No live constituency record for item slug: {item_slug}")
        record = records_by_slug[item_slug]
        constituency = record["constituency"]
        context = _item_context(record)
        item_dir = destination_root / "generated" / item_slug
        assets_dir = item_dir / "assets"
        slides_dir = item_dir / "slides"
        item_manifest_path = item_dir / "item_manifest.json"
        item_manifest = load_json(item_manifest_path)
        slide_manifest_by_id = {entry["slide_id"]: entry for entry in item_manifest.get("slides", [])}

        cover_asset = assets_dir / "constituency_cover.png"
        visual_asset = assets_dir / "issue_profile.png"
        if "cover" in selected_slide_ids:
            write_cover_asset(cover_asset, context)
        if "issue_profile" in selected_slide_ids:
            item_manifest["visual_manifest"] = render_visual(
                visual_asset,
                item_dir / "metadata/issue_profile.visual.json",
                item_dir / "manifests/issue_profile.visual_manifest.json",
                context,
            )

        for slide_id in selected_slide_ids:
            slide = slides_by_id[slide_id]
            media_path = cover_asset if slide_id == "cover" else visual_asset
            output_path = slides_dir / f"{slide['order']:02d}_{slide_id}.png"
            result = render_template(template, _bindings(slide, constituency, media_path), output_path)
            if result.warnings:
                raise ValueError(f"Render warnings for {constituency}/{slide_id}: {result.warnings}")
            with Image.open(output_path) as rendered_image:
                width, height = rendered_image.size
            slide_manifest_by_id[slide_id] = {
                "slide_id": slide_id,
                "order": slide["order"],
                "layout_id": slide["post_type_id"],
                "path": str(output_path.relative_to(destination_root)),
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
        item_manifest["slides"] = sorted(slide_manifest_by_id.values(), key=lambda entry: entry["order"])
        item_manifest.setdefault("regeneration_history", []).append(
            {
                "timestamp": utc_now(),
                "parent_run_id": manifest["parent_run_id"],
                "reason": reason,
                "slide_ids": sorted(selected_slide_ids),
            }
        )
        write_json(item_manifest_path, item_manifest)

    manifest_path = destination_root / "run_manifest.json"
    manifest = load_json(manifest_path)
    manifest["source_manifest"] = source_manifest
    manifest["join_manifest"] = join_manifest
    manifest["regenerated_outputs"] = regenerated
    manifest["state"] = "succeeded"
    manifest["review_state"] = "unreviewed"
    manifest["approved"] = False
    manifest["publishing_allowed"] = False
    write_json(manifest_path, manifest)
    return {**manifest, "output_root": str(destination_root)}
