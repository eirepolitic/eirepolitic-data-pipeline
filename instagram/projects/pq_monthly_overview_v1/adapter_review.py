from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from instagram.projects.pq_monthly_overview_v1 import adapter as base
from instagram.renderer.template_renderer import render_template

PROJECT_LAYOUT = Path("instagram/projects/pq_monthly_overview_v1/headline_layout_v1.json")
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

    custom_layout = json.loads(PROJECT_LAYOUT.read_text(encoding="utf-8"))
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


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    return base.generate(project=project, period_spec=period_spec, output_root=output_root)
