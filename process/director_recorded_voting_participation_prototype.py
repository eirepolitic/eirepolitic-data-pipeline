#!/usr/bin/env python3
"""Render one representative Director-review slide for recorded voting participation.

This is intentionally a single-slide prototype for the visual-direction gate.
It reads validated evidence already committed to the Director session branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from instagram.renderer.template_renderer import render_template
from instagram.visuals.renderers import horizontal_bar

SESSION_ROOT = Path("director/sessions/2026-09-27-recorded-voting-participation")
EVIDENCE = SESSION_ROOT / "evidence"
OUT = SESSION_ROOT / "prototype"

PALETTE = {
    "background": "#0f2f24",
    "panel": "#0f2f24",
    "text": "#f4ead7",
    "muted": "#c8bda8",
    "accent": "#d8b45f",
    "grid": "#f4ead7",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parties = pd.read_csv(EVIDENCE / "party_participation.csv")
    parties = parties.sort_values(["party_name", "party_uri"], kind="stable").reset_index(drop=True)

    rows = []
    for row in parties.itertuples(index=False):
        numerator = int(row.recorded_participation_opportunities)
        denominator = int(row.eligible_division_opportunities)
        rows.append(
            {
                "label": f"{row.party_name} · {numerator:,}/{denominator:,}",
                "value": float(row.recorded_participation_pct),
            }
        )

    chart_template = {
        "template_id": "recorded_voting_participation_party_prototype_v1",
        "params": {
            "width": 928,
            "height": 760,
            "max_items": 11,
            "sort": "input",
            "value_format": "percent",
            "min_visual_rows": 11,
        },
        "palette": PALETTE,
    }
    sample = {
        "visual_id": "recorded-voting-participation-parties-prototype",
        "bindings": {"label": "label", "value": "value"},
        "source_note": "Labels show recorded participation opportunities / eligible division opportunities",
    }

    visual_path = OUT / "parties_visual.png"
    visual_metadata_path = OUT / "parties_visual_metadata.json"
    visual_manifest_path = OUT / "parties_visual_manifest.json"
    manifest = horizontal_bar.render(
        chart_template,
        sample,
        rows,
        visual_path,
        visual_metadata_path,
        visual_manifest_path,
        {
            "source": str(EVIDENCE / "party_participation.csv"),
            "period_start": "2026-02-28",
            "period_end": "2026-08-28",
            "division_count": 136,
            "ordering": "alphabetical",
            "metric": "recorded participation opportunities / eligible division opportunities",
            "prototype_only": True,
        },
    )
    warnings = manifest.get("warnings") or []
    if warnings:
        raise RuntimeError(f"Prototype chart QA warnings: {warnings}")

    layout = json.loads(Path("instagram/templates/layouts/title_text_media_v1.json").read_text(encoding="utf-8"))
    slide_path = OUT / "prototype_party_slide.png"
    rendered = render_template(
        layout,
        {
            "slide_title": "Recorded voting participation — parties",
            "body_text": (
                "Share of eligible Dáil division opportunities with a recorded Tá, Níl or formal abstention. "
                "Alphabetical order — this is not an attendance measure."
            ),
            "main_media": str(visual_path),
            "footer_text": "28 Feb–28 Aug 2026 · 136 divisions · Houses of the Oireachtas / EirePolitic analysis",
        },
        slide_path,
    )
    if rendered.warnings:
        raise RuntimeError(f"Prototype outer-layout warnings: {rendered.warnings}")

    prototype_manifest = {
        "status": "PASS",
        "prototype_only": True,
        "visual_direction_gate": "pending_human_review",
        "slide": str(slide_path),
        "period": {"start": "2026-02-28", "end": "2026-08-28"},
        "division_count": 136,
        "party_count": len(rows),
        "ordering": "alphabetical",
        "metric": "recorded participation opportunities / eligible division opportunities",
        "denominator_display": "numerator/denominator embedded beside each party label",
        "canonical_reference": "party_issue_monthly_profile_v2 July 2026 analytical horizontal-bar treatment",
        "renderer": "instagram.visuals.renderers.horizontal_bar",
        "outer_layout": "instagram/templates/layouts/title_text_media_v1.json",
        "chart_readability": manifest.get("readability") or {},
        "chart_warnings": warnings,
        "layout_warnings": rendered.warnings,
        "editorial_caveat": "Recorded voting participation does not by itself measure a TD's attendance, workload, effectiveness, or overall job performance.",
        "publication_enabled": False,
    }
    (OUT / "prototype_manifest.json").write_text(json.dumps(prototype_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(prototype_manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
