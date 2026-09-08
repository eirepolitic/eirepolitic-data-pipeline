from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from instagram.factory.package import deterministic_zip
from instagram.factory.render_primitives import contact_sheet
from instagram.renderer.template_renderer import render_template
from instagram.projects.bill_tracker_vote_options_v1.renderers import render_option_a, render_option_b

PROJECT_ID = "bill_tracker_vote_options_v1"
FACTORY_REFERENCE_COMMIT = "386b933"


def _outer(project: dict[str, Any], *, title: str, visual: Path, output: Path) -> dict[str, Any]:
    layout_path = Path(str((project.get("render") or {})["outer_layout"]))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    result = render_template(layout, {"slide_title": title, "main_media": str(visual)}, output)
    if result.warnings:
        raise RuntimeError(f"outer layout warnings: {result.warnings}")
    return {"layout": str(layout_path), "text_metrics": result.text_metrics}


def _check(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing slide: {path}")
    with Image.open(path) as im:
        if im.size != (1080, 1350):
            raise RuntimeError(f"bad dimensions {im.size} for {path}")


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    root = output_root / "period=sample"
    if root.exists(): shutil.rmtree(root)
    slides = root / "slides"; assets = root / "assets"; metadata = root / "metadata"; contact = root / "contact_sheets"
    for d in (slides, assets, metadata, contact): d.mkdir(parents=True, exist_ok=True)

    a_media = assets / "01_option_a_media.png"
    b_media = assets / "02_option_b_media.png"
    a_manifest = render_option_a(a_media)
    b_manifest = render_option_b(b_media)

    a_slide = slides / "01_option_a_overall_vote.png"
    b_slide = slides / "02_option_b_party_breakdown.png"
    a_outer = _outer(project, title="Strategic Gas Reserve · Vote", visual=a_media, output=a_slide)
    b_outer = _outer(project, title="Strategic Gas Reserve · Party Split", visual=b_media, output=b_slide)
    for p in (a_slide, b_slide): _check(p)

    contact_path = contact / "vote_options.jpg"
    contact_sheet([("A · Overall vote", a_slide), ("B · Party breakdown", b_slide)], contact_path, columns=2)

    manifest = {
        "project_id": PROJECT_ID,
        "review_state": "pending_human_review",
        "publication_enabled": False,
        "factory_reference_commit": FACTORY_REFERENCE_COMMIT,
        "source_division_id": "https://data.oireachtas.ie/ie/oireachtas/division/house/dail/34/2026-06-30/vote_162",
        "source_proposition": "Remaining sections and Title agreed; Fourth Stage completed; Bill passed",
        "overall": {"eligible": 174, "for": 90, "against": 57, "abstain": 0, "no_recorded_vote": 27},
        "slides": [str(a_slide), str(b_slide)],
        "contact_sheet": str(contact_path),
        "renderers": {"option_a": a_manifest, "option_b": b_manifest},
        "outer": {"option_a": a_outer, "option_b": b_outer},
        "notes": [
            "No recorded vote is not equivalent to absent.",
            "Party attribution is reconstructed from date-correct silver_member_parties history because party_name_at_vote is blank in these vote rows.",
            "No ambiguous party histories were found for the vote date.",
        ],
    }
    (metadata / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    deterministic_zip(root, root / "bill_vote_options_review.zip")
    return manifest
