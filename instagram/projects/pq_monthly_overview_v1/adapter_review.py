from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from instagram.projects.pq_monthly_overview_v1 import adapter as base
from instagram.renderer.template_renderer import render_template

HEADLINE_LAYOUT = Path("instagram/projects/pq_monthly_overview_v1/headline_layout_v1.json")
DESCRIPTOR_LAYOUT = Path("instagram/projects/pq_monthly_overview_v1/descriptor_layout_v1.json")
_original_render_text_slide = base._render_text_slide


def _first_int(text: str) -> str:
    match = re.search(r"[+-]?[\d,]+", text)
    return match.group(0) if match else ""


def _render_text_slide_review(
    *,
    slide_id: str,
    slide_title: str,
    lines: list[str],
    footer_text: str,
    period_root: Path,
    layout: dict[str, Any],
    slide_index: int,
) -> dict[str, Any]:
    if slide_id != "headline":
        return _original_render_text_slide(
            slide_id=slide_id,
            slide_title=slide_title,
            lines=lines,
            footer_text=footer_text,
            period_root=period_root,
            layout=layout,
            slide_index=slide_index,
        )

    period_label = slide_title.split(":", 1)[1].strip() if ":" in slide_title else ""
    hero_value = _first_int(lines[0]) if len(lines) > 0 else ""

    written_value = ""
    oral_value = ""
    if len(lines) > 1:
        parts = [part.strip() for part in lines[1].split("·")]
        if parts:
            written_value = _first_int(parts[0])
        if len(parts) > 1:
            oral_value = _first_int(parts[1])

    askers_value = _first_int(lines[2]) if len(lines) > 2 else ""
    days_value = _first_int(lines[3]) if len(lines) > 3 else ""

    delta_text = lines[4] if len(lines) > 4 else ""
    delta_value = _first_int(delta_text)
    previous_month = ""
    previous_total = ""
    match = re.search(r"vs\s+(.+?)\s+\(([\d,]+)\s+questions\)", delta_text)
    if match:
        previous_month = match.group(1).strip()
        previous_total = match.group(2)

    custom_layout = json.loads(HEADLINE_LAYOUT.read_text(encoding="utf-8"))
    slides_dir = period_root / "slides"
    slide_path = slides_dir / f"{slide_index:02d}_{slide_id}.png"
    bindings = {
        "eyebrow": "MONTHLY QUESTIONS OVERVIEW",
        "title": "Parliamentary Questions",
        "period": period_label,
        "hero_value": hero_value,
        "hero_label": "PARLIAMENTARY QUESTIONS",
        "written_value": written_value,
        "written_label": "Written",
        "oral_value": oral_value,
        "oral_label": "Oral",
        "askers_value": askers_value,
        "askers_label": "TDs asked questions",
        "days": f"{days_value} sitting days",
        "delta": f"{delta_value} vs {previous_month}" if previous_month else delta_text,
        "comparison_label": "Change in total questions from previous month",
        "previous_total": f"Previous month total: {previous_total} questions" if previous_total else "",
        "footer_text": footer_text,
    }
    rendered = render_template(custom_layout, bindings, slide_path)
    if rendered.warnings:
        raise RuntimeError(f"Headline layout warnings for {slide_id}: {rendered.warnings}")
    return {
        "id": slide_id,
        "title": slide_title,
        "path": str(slide_path.relative_to(period_root)),
        "slide_path_abs": str(slide_path),
    }


base._render_text_slide = _render_text_slide_review


def _insert_descriptor(period_root: Path, raw: dict[str, Any]) -> None:
    slides_dir = period_root / "slides"
    existing = sorted(slides_dir.glob("*.png"))

    # Shift existing slides 2..N upward by one, moving from the end so names never collide.
    for path in reversed(existing):
        match = re.match(r"(\d{2})_(.+)\.png$", path.name)
        if not match:
            continue
        index = int(match.group(1))
        if index >= 2:
            path.rename(slides_dir / f"{index + 1:02d}_{match.group(2)}.png")

    layout = json.loads(DESCRIPTOR_LAYOUT.read_text(encoding="utf-8"))
    descriptor_path = slides_dir / "02_descriptor.png"
    bindings = {
        "eyebrow": "HOW TO READ THIS POST",
        "title": "What is a parliamentary question?",
        "intro": (
            "Parliamentary questions are one of the main ways TDs ask Government ministers "
            "for information and explanations about public matters."
        ),
        "written_heading": "WRITTEN QUESTIONS",
        "written_body": (
            "Submitted in writing to a minister. The reply is provided in writing and published "
            "in the official parliamentary record."
        ),
        "oral_heading": "ORAL QUESTIONS",
        "oral_body": (
            "Selected for oral answer in the Dáil. The minister answers in the chamber and TDs "
            "may ask supplementary questions."
        ),
        "scope_heading": "WHAT “QUESTIONS” MEANS IN THIS CAROUSEL",
        "scope_body": (
            "Dáil parliamentary questions recorded by the Houses of the Oireachtas during the month."
        ),
        "footer_text": "Source: Houses of the Oireachtas — Parliamentary Questions procedure guidance",
    }
    rendered = render_template(layout, bindings, descriptor_path)
    if rendered.warnings:
        raise RuntimeError(f"Descriptor layout warnings: {rendered.warnings}")

    raw["slides"] = [str(p) for p in sorted(slides_dir.glob("*.png"))]
    raw["slide_count"] = len(raw["slides"])

    manifest_path = Path(raw.get("manifest_path") or period_root / "run_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["slide_count"] = raw["slide_count"]
        manifest["slides"] = [str(Path(p).relative_to(period_root)) for p in raw["slides"]]
        manifest.setdefault("review_notes", []).append(
            "Review branch inserts a plain-English parliamentary-question descriptor slide at position 2."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    raw = base.generate(project=project, period_spec=period_spec, output_root=output_root)
    _insert_descriptor(Path(raw["output_root"]), raw)
    return raw
