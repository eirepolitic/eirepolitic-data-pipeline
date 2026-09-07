from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from instagram.factory.package import deterministic_zip
from instagram.factory.render_primitives import contact_sheet
from instagram.renderer.template_renderer import render_template
from instagram.projects.bill_tracker_factory_v1.renderers import (
    render_bill_media,
    render_cover_media,
    render_methodology,
)

PROJECT_ID = "bill_tracker_factory_v1"
FACTORY_REFERENCE_COMMIT = "386b933"
FACTORY_REFERENCE_WORKFLOW_RUN = 33894430571


def _render_outer(project: dict[str, Any], *, title: str, visual_path: Path, output_path: Path) -> dict[str, Any]:
    layout_path = Path(str((project.get("render") or {})["outer_layout"]))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    result = render_template(layout, {"slide_title": title, "main_media": str(visual_path)}, output_path)
    if result.warnings:
        raise RuntimeError(f"Approved outer layout warnings for {title}: {result.warnings}")
    return {"outer_layout": str(layout_path), "text_metrics": result.text_metrics}


def _assert_slide(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Slide was not created: {path}")
    with Image.open(path) as image:
        if image.size != (1080, 1350):
            raise RuntimeError(f"Unexpected slide dimensions for {path}: {image.size}")


def _load_content() -> dict[str, Any]:
    path = Path("instagram/projects/bill_tracker_factory_v1/content.yml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bills = payload.get("bills") or []
    if len(bills) != 6:
        raise RuntimeError(f"Bill Tracker prototype requires exactly 6 bills; found {len(bills)}")
    methodology = payload.get("methodology") or {}
    if len(methodology.get("entries") or []) < 3:
        raise RuntimeError("Bill Tracker methodology requires at least three entries")
    return payload


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    content = _load_content()
    edition = content["edition"]
    bills = content["bills"]
    methodology = content["methodology"]
    edition_id = str(edition["id"])

    period_root = output_root / f"period={edition_id}"
    if period_root.exists():
        shutil.rmtree(period_root)
    slides_dir = period_root / "slides"
    assets_dir = period_root / "assets"
    metadata_dir = period_root / "metadata"
    contact_dir = period_root / "contact_sheets"
    for directory in (slides_dir, assets_dir, metadata_dir, contact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    slide_paths: list[Path] = []
    outer_layouts: list[dict[str, Any]] = []
    media_manifests: dict[str, Any] = {}

    cover_media = assets_dir / "01_cover_media.png"
    media_manifests["cover"] = render_cover_media(edition, bills, cover_media)
    cover_slide = slides_dir / "01_cover.png"
    outer_layouts.append(_render_outer(project, title=str(edition["cover_title"]), visual_path=cover_media, output_path=cover_slide))
    slide_paths.append(cover_slide)

    for index, bill in enumerate(bills, start=2):
        media_path = assets_dir / f"{index:02d}_bill_media.png"
        slide_path = slides_dir / f"{index:02d}_bill.png"
        manifest = render_bill_media(bill, media_path)
        media_manifests[f"bill_{index-1}"] = manifest
        outer_layouts.append(_render_outer(project, title=str(bill["display_title"]), visual_path=media_path, output_path=slide_path))
        slide_paths.append(slide_path)

    methodology_entries = [(str(item["term"]), str(item["body"])) for item in methodology["entries"]]
    methodology_slide = slides_dir / "08_about_bill_tracker.png"
    methodology_manifest = render_methodology(methodology_entries, methodology_slide, title=str(methodology["title"]))
    media_manifests["methodology"] = methodology_manifest
    slide_paths.append(methodology_slide)

    for path in slide_paths:
        _assert_slide(path)

    contact_path = contact_dir / "eight_slide_overview.jpg"
    labels = ["Cover"] + [str(bill["display_title"]) for bill in bills] + ["About"]
    contact_sheet(list(zip(labels, slide_paths)), contact_path, columns=4)

    caption = "\n".join([
        "Bills of the Current Session — Enacted, Part 1.",
        "",
        "This first part covers six enacted Bills from the current parliamentary session. Each slide gives a short description of what the law does, who introduced it, and the main debate around it.",
        "",
        "Recorded divisions are only described as support/opposition where the exact proposition supports that interpretation. A linked vote may instead concern an amendment or procedural question.",
        "",
        f"Source: {edition['source_footer']}",
    ]).strip() + "\n"
    caption_path = period_root / "caption.txt"
    caption_path.write_text(caption, encoding="utf-8")

    manifest = {
        "project_id": PROJECT_ID,
        "edition_id": edition_id,
        "review_state": "pending_human_review",
        "publication_enabled": False,
        "factory_reference_commit": FACTORY_REFERENCE_COMMIT,
        "factory_reference_workflow_run": FACTORY_REFERENCE_WORKFLOW_RUN,
        "approved_reference_post_id": "2026-09-06-ipi-polling-carousel",
        "source_label": str(edition["source_footer"]),
        "slides": [str(path) for path in slide_paths],
        "contact_sheet": str(contact_path),
        "caption": str(caption_path),
        "media_manifests": media_manifests,
        "outer_layouts": outer_layouts,
        "qa": {
            "expected_slide_count": 8,
            "actual_slide_count": len(slide_paths),
            "dimensions": [1080, 1350],
            "approved_factory_commit": FACTORY_REFERENCE_COMMIT,
            "approved_outer_layout": "instagram/templates/layouts/title_text_media_v1.json",
            "approved_methodology_component": "draw_glossary",
            "source_footer_required": True,
            "bill_count": len(bills),
            "publication_enabled": False,
        },
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    package = deterministic_zip(period_root, period_root / "bill_tracker_factory_review.zip")
    manifest["package"] = package
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
