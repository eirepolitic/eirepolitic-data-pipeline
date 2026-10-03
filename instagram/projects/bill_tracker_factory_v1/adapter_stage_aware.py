from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from instagram.factory.oireachtas_source import load_csv_tables, resolve_validated_production_batch
from instagram.factory.package import deterministic_zip
from instagram.factory.render_primitives import contact_sheet
from instagram.projects.bill_tracker_factory_v1 import adapter as enacted_adapter
from instagram.projects.bill_tracker_factory_v1.first_stage_renderers import render_first_stage_bill

PROJECT_ID = "bill_tracker_factory_v1"
FIRST_STAGE_PROTOTYPE = "first_stage_prototype"


def _assert_image(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing rendered slide: {path}")
    with Image.open(path) as image:
        if image.size != (1080, 1350):
            raise RuntimeError(f"Unexpected dimensions for {path}: {image.size}")


def _generate_first_stage_prototype(*, output_root: Path) -> dict[str, Any]:
    content_path = Path("instagram/projects/bill_tracker_factory_v1/first_stage_content.yml")
    payload = yaml.safe_load(content_path.read_text(encoding="utf-8")) or {}
    prototype = payload.get("prototype") or {}
    bill = prototype.get("bill") or {}
    if not bill:
        raise RuntimeError("First Stage prototype content is missing its Bill payload")

    batch = resolve_validated_production_batch()
    frames, lineage = load_csv_tables(
        batch,
        ["silver_bills", "silver_bill_stages", "silver_bill_sponsors", "silver_bill_debates"],
    )
    bill_uri = f"https://data.oireachtas.ie/ie/oireachtas/bill/2026/44"
    production_rows = {}
    for table_name, frame in frames.items():
        if "bill_id" in frame.columns:
            production_rows[table_name] = int(frame[frame["bill_id"].astype(str).eq(bill_uri)].shape[0])
        else:
            production_rows[table_name] = 0
    if production_rows.get("silver_bills", 0) != 1:
        raise RuntimeError(f"Expected exactly one production Bill row for {bill_uri}: {production_rows}")

    period = FIRST_STAGE_PROTOTYPE
    root = output_root / f"period={period}"
    if root.exists():
        shutil.rmtree(root)
    slides_dir = root / "slides"
    metadata_dir = root / "metadata"
    contact_dir = root / "contact_sheets"
    for directory in (slides_dir, metadata_dir, contact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    slide = slides_dir / "01_adult_safeguarding_first_stage.png"
    render_manifest = render_first_stage_bill(bill, slide)
    _assert_image(slide)

    contact_path = contact_dir / "first_stage_prototype_contact_sheet.jpg"
    contact_sheet([("Adult Safeguarding · First Stage", slide)], contact_path, columns=1)

    caption_path = root / "caption.txt"
    caption_path.write_text(
        "REVIEW-ONLY PROTOTYPE — not publication copy.\n",
        encoding="utf-8",
    )
    manifest = {
        "project_id": PROJECT_ID,
        "period_key": period,
        "review_state": "pending_human_review",
        "publication_enabled": False,
        "publishing_allowed": False,
        "source_batch_id": batch.batch_id,
        "source_pointer": batch.pointer,
        "source_lineage": lineage,
        "production_bill_row_counts": production_rows,
        "editorial_verification_date": "2026-10-03",
        "editorial_sources": bill.get("sources") or [],
        "slides": [str(slide)],
        "contact_sheets": {period: str(contact_path)},
        "caption": str(caption_path),
        "render_manifests": {slide.stem: render_manifest},
        "qa": {
            "expected_slide_count": 1,
            "actual_slide_count": 1,
            "dimensions": [1080, 1350],
            "source_footer_required": True,
            "publication_enabled": False,
            "publishing_allowed": False,
            "overflow_assertions_passed": True,
        },
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    package = deterministic_zip(root, root / "bill_tracker_first_stage_prototype_review.zip")
    manifest["package"] = package
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    period = (period_spec or "post1").strip().lower()
    if period == FIRST_STAGE_PROTOTYPE:
        return _generate_first_stage_prototype(output_root=output_root)
    return enacted_adapter.generate(project=project, period_spec=period, output_root=output_root)
